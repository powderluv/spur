use std::collections::HashSet;
use std::sync::Arc;

use chrono::Utc;
use tracing::{debug, error, info, warn};

use spur_proto::proto::slurm_agent_client::SlurmAgentClient;
use spur_proto::proto::{
    AgentCancelJobRequest, JobSpec as ProtoJobSpec, LaunchJobRequest,
    ResourceSet as ProtoResourceSet,
};
use spur_sched::backfill::{self, BackfillScheduler};
use spur_sched::traits::{ClusterState, Scheduler};

use crate::cluster::ClusterManager;

/// Spawn the time-limit enforcement watchdog alongside the scheduler loop.
pub async fn run(cluster: Arc<ClusterManager>) {
    let enforcer_cluster = cluster.clone();
    tokio::spawn(async move {
        enforce_time_limits(enforcer_cluster).await;
    });
    let interval_secs = cluster.config.scheduler.interval_secs.max(1) as u64;
    let max_jobs = cluster.config.scheduler.max_jobs_per_cycle as usize;

    let mut scheduler = BackfillScheduler::new(max_jobs);

    info!(
        interval_secs,
        max_jobs,
        plugin = scheduler.name(),
        "scheduler loop started"
    );

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));

    loop {
        interval.tick().await;

        let pending = cluster.pending_jobs();
        if pending.is_empty() {
            continue;
        }

        let nodes = cluster.get_nodes();
        let partitions = cluster.get_partitions();

        if nodes.is_empty() {
            debug!("no nodes registered, skipping scheduling cycle");
            continue;
        }

        let cluster_state = ClusterState {
            nodes: &nodes,
            partitions: &partitions,
        };

        let assignments = scheduler.schedule(&pending, &cluster_state);

        // Preemption: if high-priority jobs couldn't be scheduled,
        // cancel lower-priority running jobs to free resources.
        if assignments.len() < pending.len() {
            let unscheduled: Vec<_> = pending
                .iter()
                .filter(|p| !assignments.iter().any(|a| a.job_id == p.job_id))
                .collect();

            if !unscheduled.is_empty() {
                try_preempt(&cluster, &unscheduled);
            }
        }

        for assignment in assignments {
            let job = match cluster.get_job(assignment.job_id) {
                Some(j) => j,
                None => continue,
            };

            // Compute per-node resources (including GPUs from GRES)
            let per_node = backfill::job_resource_request(&job);
            let node_count = assignment.nodes.len() as u32;
            let resources = spur_core::resource::ResourceSet {
                cpus: per_node.cpus * node_count,
                memory_mb: per_node.memory_mb * node_count as u64,
                gpus: per_node.gpus.clone(),
                generic: per_node
                    .generic
                    .iter()
                    .map(|(k, v)| (k.clone(), v * node_count as u64))
                    .collect(),
            };

            // Transition job to Running
            if let Err(e) =
                cluster.start_job(assignment.job_id, assignment.nodes.clone(), resources)
            {
                debug!(
                    job_id = assignment.job_id,
                    error = %e,
                    "failed to start job"
                );
                continue;
            }

            // Dispatch job to ALL assigned nodes
            let job_id = assignment.job_id;
            let spec = job.spec.clone();
            let all_nodes = assignment.nodes.clone();
            let per_node_alloc = per_node.clone();

            // Build peer_nodes list with addresses for cross-node communication
            let peer_addrs: Vec<String> = all_nodes
                .iter()
                .filter_map(|name| {
                    cluster
                        .get_node(name)
                        .and_then(|n| n.address.as_ref().map(|a| format!("{}:{}", a, n.port)))
                })
                .collect();

            let tasks_per_node = if let Some(tpn) = spec.tasks_per_node {
                tpn
            } else {
                (spec.num_tasks / spec.num_nodes.max(1)).max(1)
            };

            for (node_idx, node_name) in all_nodes.iter().enumerate() {
                let node_info = cluster.get_node(node_name);
                let (addr, port) = match node_info {
                    Some(ref n) if n.address.is_some() => (n.address.clone().unwrap(), n.port),
                    _ => {
                        warn!(
                            job_id,
                            node = %node_name,
                            "no agent address for node, skipping dispatch"
                        );
                        continue;
                    }
                };

                let agent_addr = format!("http://{}:{}", addr, port);
                let spec = spec.clone();
                let peer_addrs = peer_addrs.clone();
                let task_offset = node_idx as u32 * tasks_per_node;
                let target_node = node_name.clone();
                let allocated = per_node_alloc.clone();
                tokio::spawn(async move {
                    if let Err(e) = dispatch_to_agent(
                        &agent_addr,
                        job_id,
                        &spec,
                        &peer_addrs,
                        task_offset,
                        &target_node,
                        &allocated,
                    )
                    .await
                    {
                        // Log but do NOT mark job as Failed — that breaks afterok
                        // dependencies. The job stays Running; the time-limit
                        // enforcer will eventually clean it up if it has a
                        // --time set. Agent completion reporting handles the
                        // normal path.
                        error!(
                            job_id,
                            agent = %agent_addr,
                            error = %e,
                            "failed to dispatch job to agent"
                        );
                    }
                });
            }
        }
    }
}

/// Try to preempt lower-priority running jobs to make room for higher-priority pending jobs.
fn try_preempt(cluster: &Arc<ClusterManager>, unscheduled: &[&spur_core::job::Job]) {
    use spur_core::job::JobState;

    // Get running jobs sorted by priority (lowest first = best preemption candidates)
    let mut running: Vec<spur_core::job::Job> = cluster
        .get_jobs(&[JobState::Running], None, None, None, &[])
        .into_iter()
        .collect();
    running.sort_by_key(|j| j.priority);

    for pending in unscheduled {
        // Only preempt if pending job has significantly higher priority
        for candidate in &running {
            if candidate.priority < pending.priority / 2 {
                // Preempt: cancel the lower-priority job
                info!(
                    preempted_job = candidate.job_id,
                    preempted_priority = candidate.priority,
                    pending_job = pending.job_id,
                    pending_priority = pending.priority,
                    "preempting lower-priority job"
                );
                if let Err(e) = cluster.complete_job(candidate.job_id, -1, JobState::Preempted) {
                    warn!(
                        job_id = candidate.job_id,
                        error = %e,
                        "failed to preempt job"
                    );
                }
                break; // One preemption per cycle, re-evaluate next cycle
            }
        }
    }
}

/// Send a LaunchJob RPC to a node agent.
async fn dispatch_to_agent(
    agent_addr: &str,
    job_id: u32,
    spec: &spur_core::job::JobSpec,
    peer_nodes: &[String],
    task_offset: u32,
    target_node: &str,
    allocated: &spur_core::resource::ResourceSet,
) -> anyhow::Result<()> {
    let mut client = SlurmAgentClient::connect(agent_addr.to_string()).await?;

    let proto_spec = ProtoJobSpec {
        name: spec.name.clone(),
        partition: spec.partition.clone().unwrap_or_default(),
        account: spec.account.clone().unwrap_or_default(),
        user: spec.user.clone(),
        uid: spec.uid,
        gid: spec.gid,
        num_nodes: spec.num_nodes,
        num_tasks: spec.num_tasks,
        tasks_per_node: spec.tasks_per_node.unwrap_or(0),
        cpus_per_task: spec.cpus_per_task,
        memory_per_node_mb: spec.memory_per_node_mb.unwrap_or(0),
        memory_per_cpu_mb: spec.memory_per_cpu_mb.unwrap_or(0),
        gres: spec.gres.clone(),
        script: spec.script.clone().unwrap_or_default(),
        argv: spec.argv.clone(),
        work_dir: spec.work_dir.clone(),
        stdout_path: spec.stdout_path.clone().unwrap_or_default(),
        stderr_path: spec.stderr_path.clone().unwrap_or_default(),
        environment: spec.environment.clone(),
        time_limit: spec.time_limit.map(|d| prost_types::Duration {
            seconds: d.num_seconds(),
            nanos: 0,
        }),
        time_min: None,
        qos: spec.qos.clone().unwrap_or_default(),
        priority: spec.priority.unwrap_or(0),
        reservation: spec.reservation.clone().unwrap_or_default(),
        dependency: spec.dependency.clone(),
        nodelist: spec.nodelist.clone().unwrap_or_default(),
        exclude: spec.exclude.clone().unwrap_or_default(),
        array_spec: spec.array_spec.clone().unwrap_or_default(),
        requeue: spec.requeue,
        exclusive: spec.exclusive,
        hold: spec.hold,
        comment: spec.comment.clone().unwrap_or_default(),
        wckey: spec.wckey.clone().unwrap_or_default(),
        container_image: spec.container_image.clone().unwrap_or_default(),
        container_mounts: spec.container_mounts.clone(),
        container_workdir: spec.container_workdir.clone().unwrap_or_default(),
        container_name: spec.container_name.clone().unwrap_or_default(),
        container_readonly: spec.container_readonly,
        container_mount_home: spec.container_mount_home,
        container_env: spec.container_env.clone(),
        container_entrypoint: spec.container_entrypoint.clone().unwrap_or_default(),
        container_remap_root: spec.container_remap_root,
    };

    let response = client
        .launch_job(LaunchJobRequest {
            job_id,
            spec: Some(proto_spec),
            allocated: Some(core_resource_to_proto(allocated)),
            peer_nodes: peer_nodes.to_vec(),
            task_offset,
            target_node: target_node.to_string(),
        })
        .await?;

    let inner = response.into_inner();
    if inner.success {
        info!(job_id, "job dispatched to agent successfully");
    } else {
        anyhow::bail!("agent rejected job: {}", inner.error);
    }

    Ok(())
}

/// Convert a core ResourceSet to proto ResourceSet.
fn core_resource_to_proto(r: &spur_core::resource::ResourceSet) -> ProtoResourceSet {
    use spur_core::resource::GpuLinkType;
    ProtoResourceSet {
        cpus: r.cpus,
        memory_mb: r.memory_mb,
        gpus: r
            .gpus
            .iter()
            .map(|g| spur_proto::proto::GpuResource {
                device_id: g.device_id,
                gpu_type: g.gpu_type.clone(),
                memory_mb: g.memory_mb,
                peer_gpus: g.peer_gpus.clone(),
                link_type: match g.link_type {
                    GpuLinkType::XGMI => spur_proto::proto::GpuLinkType::GpuLinkXgmi as i32,
                    GpuLinkType::NVLink => spur_proto::proto::GpuLinkType::GpuLinkNvlink as i32,
                    GpuLinkType::PCIe => spur_proto::proto::GpuLinkType::GpuLinkPcie as i32,
                },
            })
            .collect(),
        generic: r.generic.clone(),
    }
}

/// Watchdog: gracefully terminate running jobs that exceed their time limit.
///
/// Two-phase shutdown:
///   1. **Warning phase**: When `start_time + time_limit < now`, send SIGTERM
///      (signal 15) to all agents and record the job as "warned".
///   2. **Kill phase**: 30 seconds after warning, if the job is still running,
///      mark it as Timeout and send SIGKILL (signal 9).
///
/// Runs every 10 seconds.
async fn enforce_time_limits(cluster: Arc<ClusterManager>) {
    const GRACE_PERIOD_SECS: i64 = 30;

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
    // Jobs that have been warned (SIGTERM sent) — maps job_id → warn timestamp
    let mut warned_jobs: HashSet<spur_core::job::JobId> = HashSet::new();
    let mut warn_times: std::collections::HashMap<spur_core::job::JobId, chrono::DateTime<Utc>> =
        std::collections::HashMap::new();

    loop {
        interval.tick().await;

        let now = Utc::now();
        let running = cluster.get_jobs(&[spur_core::job::JobState::Running], None, None, None, &[]);

        for job in &running {
            let (Some(time_limit), Some(start_time)) = (job.spec.time_limit, job.start_time) else {
                continue;
            };
            let deadline = start_time + time_limit;
            if now < deadline {
                continue;
            }

            if warned_jobs.contains(&job.job_id) {
                // Already warned — check if grace period has elapsed
                let warn_time = warn_times
                    .get(&job.job_id)
                    .copied()
                    .unwrap_or(now - chrono::Duration::seconds(GRACE_PERIOD_SECS + 1));
                if (now - warn_time).num_seconds() < GRACE_PERIOD_SECS {
                    continue; // Still in grace period
                }

                // Grace period expired — force kill
                info!(
                    job_id = job.job_id,
                    "grace period expired — force-killing job"
                );

                if let Err(e) =
                    cluster.complete_job(job.job_id, -1, spur_core::job::JobState::Timeout)
                {
                    warn!(job_id = job.job_id, error = %e, "failed to mark job as timed out");
                    continue;
                }

                send_cancel_to_agents(&cluster, job, 9).await; // SIGKILL
                warned_jobs.remove(&job.job_id);
                warn_times.remove(&job.job_id);
            } else {
                // First time past deadline — send SIGTERM (graceful warning)
                info!(
                    job_id = job.job_id,
                    elapsed_secs = (now - start_time).num_seconds(),
                    limit_secs = time_limit.num_seconds(),
                    grace_secs = GRACE_PERIOD_SECS,
                    "time limit exceeded — sending SIGTERM, grace period starts"
                );

                send_cancel_to_agents(&cluster, job, 15).await; // SIGTERM
                warned_jobs.insert(job.job_id);
                warn_times.insert(job.job_id, now);
            }
        }

        // Clean up warned_jobs for jobs that are no longer running
        // (e.g., they exited cleanly during grace period)
        let running_ids: HashSet<_> = running.iter().map(|j| j.job_id).collect();
        warned_jobs.retain(|id| running_ids.contains(id));
        warn_times.retain(|id, _| running_ids.contains(id));
    }
}

/// Send CancelJob RPC to all agents for a job with a specific signal.
async fn send_cancel_to_agents(
    cluster: &Arc<ClusterManager>,
    job: &spur_core::job::Job,
    signal: i32,
) {
    for node_name in &job.allocated_nodes {
        let node_info = cluster.get_node(node_name);
        let (addr, port) = match node_info {
            Some(ref n) if n.address.is_some() => (n.address.clone().unwrap(), n.port),
            _ => {
                warn!(
                    job_id = job.job_id,
                    node = %node_name,
                    "no agent address — cannot cancel job on node"
                );
                continue;
            }
        };

        let agent_addr = format!("http://{}:{}", addr, port);
        let job_id = job.job_id;
        tokio::spawn(async move {
            match SlurmAgentClient::connect(agent_addr.clone()).await {
                Ok(mut client) => {
                    if let Err(e) = client
                        .cancel_job(AgentCancelJobRequest { job_id, signal })
                        .await
                    {
                        warn!(
                            job_id,
                            signal,
                            agent = %agent_addr,
                            error = %e,
                            "CancelJob RPC failed"
                        );
                    } else {
                        info!(job_id, signal, agent = %agent_addr, "sent CancelJob");
                    }
                }
                Err(e) => {
                    warn!(
                        job_id,
                        agent = %agent_addr,
                        error = %e,
                        "failed to connect to agent for cancel"
                    );
                }
            }
        });
    }
}
