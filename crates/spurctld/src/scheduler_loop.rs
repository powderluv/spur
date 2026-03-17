use std::sync::Arc;

use tracing::{debug, error, info, warn};

use spur_proto::proto::slurm_agent_client::SlurmAgentClient;
use spur_proto::proto::{JobSpec as ProtoJobSpec, LaunchJobRequest, ResourceSet as ProtoResourceSet};
use spur_sched::backfill::BackfillScheduler;
use spur_sched::traits::{ClusterState, Scheduler};

use crate::cluster::ClusterManager;

/// Main scheduler loop. Runs periodically, matching pending jobs to available nodes.
pub async fn run(cluster: Arc<ClusterManager>) {
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

        for assignment in assignments {
            let job = match cluster.get_job(assignment.job_id) {
                Some(j) => j,
                None => continue,
            };

            let per_node_cpus = if let Some(tpn) = job.spec.tasks_per_node {
                tpn * job.spec.cpus_per_task
            } else {
                (job.spec.num_tasks / job.spec.num_nodes.max(1)) * job.spec.cpus_per_task
            };

            let resources = spur_core::resource::ResourceSet {
                cpus: per_node_cpus * assignment.nodes.len() as u32,
                memory_mb: job.spec.memory_per_node_mb.unwrap_or(0)
                    * assignment.nodes.len() as u64,
                ..Default::default()
            };

            // Transition job to Running
            if let Err(e) = cluster.start_job(
                assignment.job_id,
                assignment.nodes.clone(),
                resources,
            ) {
                debug!(
                    job_id = assignment.job_id,
                    error = %e,
                    "failed to start job"
                );
                continue;
            }

            // Dispatch job to the first assigned node's agent
            let first_node = &assignment.nodes[0];
            let node_addr = cluster
                .get_node(first_node)
                .and_then(|n| n.address.clone());

            if let Some(addr) = node_addr {
                let job_id = assignment.job_id;
                let spec = job.spec.clone();
                let agent_addr = format!("http://{}:6818", addr);

                tokio::spawn(async move {
                    if let Err(e) = dispatch_to_agent(&agent_addr, job_id, &spec).await {
                        error!(job_id, agent = %agent_addr, error = %e, "failed to dispatch job to agent");
                    }
                });
            } else {
                warn!(
                    job_id = assignment.job_id,
                    node = %first_node,
                    "no agent address for node, job will not execute"
                );
            }
        }
    }
}

/// Send a LaunchJob RPC to a node agent.
async fn dispatch_to_agent(
    agent_addr: &str,
    job_id: u32,
    spec: &spur_core::job::JobSpec,
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
    };

    let response = client
        .launch_job(LaunchJobRequest {
            job_id,
            spec: Some(proto_spec),
            allocated: None,
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
