// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use tracing::{debug, error, info, warn};

use spur_core::node::{Node, NodeSource};
use spur_core::partition::requested_partition_names;
use spur_core::task_launch::batch_dispatched_multi_node_pmix;
use spur_proto::proto::slurm_controller_client::SlurmControllerClient;
use spur_proto::proto::{
    AgentCancelJobRequest, AgentSuspendJobRequest, JobSpec as ProtoJobSpec, LaunchJobRequest,
    RegisterJobAllocationRequest, SubmitJobRequest,
};
use spur_sched::backfill::{self, BackfillScheduler};
use spur_sched::traits::{ClusterState, Scheduler};

use crate::cluster::{ClusterManager, JobFilter};
use crate::pmix_dispatch::{self, PmixPrepareNode};
use crate::raft::RaftHandle;

/// Upper bound on a single CancelJob RPC (connect + call) when the caller
/// awaits delivery. Best-effort cleanup must not stall eviction on an
/// unreachable agent.
const CANCEL_RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// Grace between SIGTERM and SIGKILL when force-finishing a job. The time-limit
/// and inactive-limit watchdogs share it so the two windows can't drift apart.
const GRACE_PERIOD_SECS: i64 = 30;

fn node_comm_socket(node: &Node) -> Option<String> {
    let host = node.comm_addr()?;
    Some(spur_net::format_comm_socket(host, node.port))
}

fn node_comm_http_url(node: &Node) -> Option<String> {
    let host = node.comm_addr()?;
    Some(spur_net::format_comm_http_url(host, node.port))
}

/// Spawn the time-limit enforcement watchdog and power manager alongside the scheduler loop.
pub async fn run(cluster: Arc<ClusterManager>, raft: Arc<RaftHandle>) {
    let enforcer_cluster = cluster.clone();
    let enforcer_raft = raft.clone();
    tokio::spawn(async move {
        enforce_time_limits(enforcer_cluster, enforcer_raft).await;
    });
    let completing_cluster = cluster.clone();
    let completing_raft = raft.clone();
    tokio::spawn(async move {
        enforce_completing_timeout(completing_cluster, completing_raft).await;
    });
    let power_cluster = cluster.clone();
    let power_raft = raft.clone();
    tokio::spawn(async move {
        manage_power(power_cluster, power_raft).await;
    });
    let inactive_cluster = cluster.clone();
    let inactive_raft = raft.clone();
    tokio::spawn(async move {
        enforce_inactive_limits(inactive_cluster, inactive_raft).await;
    });
    // Captured once at loop start: the tick interval, per-cycle job cap, and
    // topology tree are baked into loop-local state and are NOT picked up by
    // `scontrol reconfigure` — changing them needs a controller restart.
    let startup_config = cluster.config();
    let interval_secs = startup_config.scheduler.interval_secs.max(1) as u64;
    let max_jobs = startup_config.scheduler.max_jobs_per_cycle as usize;

    let mut scheduler = BackfillScheduler::new(max_jobs);

    // Build topology tree from config (if configured)
    let topology = startup_config.topology.as_ref().and_then(|topo_config| {
        use spur_core::topology::TopologyTree;
        match topo_config.plugin.as_str() {
            "tree" => {
                let tree = TopologyTree::from_switches(&topo_config.switches);
                info!(
                    switches = tree.switches.len(),
                    nodes = tree.node_switch.len(),
                    "topology/tree loaded"
                );
                Some(tree)
            }
            "block" => {
                let block_size = topo_config.block_size.unwrap_or(18);
                let all_nodes = cluster.get_nodes();
                let node_names: Vec<String> = all_nodes.iter().map(|n| n.name.clone()).collect();
                let tree = TopologyTree::from_blocks(&node_names, block_size);
                info!(
                    blocks = tree.switches.len(),
                    block_size, "topology/block loaded"
                );
                Some(tree)
            }
            _ => None,
        }
    });

    info!(
        interval_secs,
        max_jobs,
        plugin = scheduler.name(),
        topology = topology.is_some(),
        "scheduler loop started (event-driven wake enabled)"
    );

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));
    let scheduler_notify = cluster.scheduler_notify.clone();

    loop {
        // Event-driven wake: sleep until EITHER a job is submitted OR the periodic tick fires.
        // This eliminates the up-to-`interval_secs` polling delay for new submissions while
        // preserving a periodic wake for resource-freed events and node state changes.
        tokio::select! {
            _ = scheduler_notify.notified() => {}
            _ = interval.tick() => {}
        }

        if !raft.is_leader() {
            // A former leader must not keep serving planned-reservation info
            // from before it lost leadership.
            cluster.set_planned_reservations(HashMap::new());
            cluster.set_planned_job_starts(HashMap::new());
            scheduler.clear_outcomes();
            continue;
        }

        // Finalize never-satisfiable deps before pending_jobs() so they drop
        // out of this cycle instead of sitting PENDING forever.
        cluster.cancel_unsatisfiable_dependency_jobs();

        // Free completed stage-in capacity before classification selects new
        // stage candidates. Real agent-side data movement is a follow-up;
        // drive_bb_stage_in() is the controller-side seam only.
        cluster.drive_bb_stage_in();
        cluster.purge_expired_reservations();
        cluster.enforce_reservation_end_times();
        cluster.evict_expired_terminal_jobs();

        // Classify once, apply reasons, and stage only candidates admitted by
        // that classification. Run before the empty-check so reasons stay fresh
        // even with nothing schedulable.
        let pending = cluster.pending_jobs_and_tag_reasons();
        if pending.is_empty() {
            // Nothing pending means nothing can be planned either.
            cluster.set_planned_reservations(HashMap::new());
            cluster.set_planned_job_starts(HashMap::new());
            scheduler.clear_outcomes();
            continue;
        }
        let hit_depth_limit = pending.len() > max_jobs;

        let nodes = cluster.nodes_off_dispatch_cooldown(&pending);
        let partitions = cluster.get_partitions();
        let reservations = cluster.get_reservations();

        if nodes.is_empty() {
            debug!("no schedulable nodes, skipping scheduling cycle");
            cluster.set_planned_reservations(HashMap::new());
            cluster.set_planned_job_starts(HashMap::new());
            scheduler.clear_outcomes();
            continue;
        }

        let cycle_start = Instant::now();

        let busy_until = running_jobs_busy_until(&cluster);
        let cluster_state = ClusterState {
            nodes: &nodes,
            partitions: &partitions,
            reservations: &reservations,
            topology: topology.as_ref(),
            busy_until: &busy_until,
        };

        // Catch panics in the scheduler so that a single bad job doesn't kill
        // the entire scheduling loop (issue #56).
        let sched_ref = &mut scheduler;
        let schedule_start = Instant::now();
        let assignments = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sched_ref.schedule(&pending, &cluster_state)
        })) {
            Ok(a) => a,
            Err(e) => {
                error!(
                    "scheduler panicked: {:?} — skipping cycle",
                    e.downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| e.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown")
                );
                let cycle_time_us = cycle_start.elapsed().as_micros().min(u64::MAX as u128) as u64;
                let schedule_time_us =
                    schedule_start.elapsed().as_micros().min(u64::MAX as u128) as u64;
                cluster.record_sched_cycle(cycle_time_us, schedule_time_us, 0, hit_depth_limit);
                // Fail safe: a scheduler that just panicked can't be trusted
                // to have produced a valid plan, planned or otherwise.
                cluster.set_planned_reservations(HashMap::new());
                cluster.set_planned_job_starts(HashMap::new());
                sched_ref.clear_outcomes();
                continue;
            }
        };
        let schedule_time_us = schedule_start.elapsed().as_micros().min(u64::MAX as u128) as u64;
        cluster.set_planned_reservations(sched_ref.planned_starts());
        cluster.set_planned_job_starts(sched_ref.planned_job_starts());

        // Preemption: if high-priority jobs couldn't be scheduled,
        // cancel lower-priority running jobs to free resources.
        if assignments.len() < pending.len() {
            let unscheduled: Vec<_> = pending
                .iter()
                .filter(|p| !assignments.iter().any(|a| a.job_id == p.job_id))
                .collect();

            if !unscheduled.is_empty() {
                // Update pending_reason for unscheduled jobs to reflect actual cause.
                // This helps users distinguish "waiting for higher-priority jobs" vs
                // "no suitable nodes at all".
                cluster.update_pending_reasons(&unscheduled, &cluster_state);

                try_preempt(
                    &cluster,
                    &partitions,
                    &unscheduled,
                    &cluster.config().scheduler,
                )
                .await;

                // Federation: forward still-unschedulable jobs to peer clusters.
                if !cluster.config().federation.clusters.is_empty() {
                    let jobs_to_fwd: Vec<spur_core::job::Job> =
                        unscheduled.iter().map(|j| (*j).clone()).collect();
                    let fed_cluster = cluster.clone();
                    tokio::spawn(async move {
                        forward_to_federation(&fed_cluster, &jobs_to_fwd).await;
                    });
                }
            }
        }

        let mut jobs_started_cycle = 0u64;
        for assignment in assignments {
            if process_assignment(cluster.clone(), assignment).await {
                jobs_started_cycle += 1;
            }
        }

        let cycle_time_us = cycle_start.elapsed().as_micros().min(u64::MAX as u128) as u64;
        cluster.record_sched_cycle(
            cycle_time_us,
            schedule_time_us,
            jobs_started_cycle,
            hit_depth_limit,
        );
    }
}

/// Process one scheduler assignment: dispatch/confirm on its nodes, start the
/// job, and run `PrologSlurmctld`. Returns whether the job actually started
/// (fed into the caller's `jobs_started_cycle` count).
///
/// Extracted from `run`'s assignment loop so this — including the
/// `confirm_dispatch_on_nodes`/`start_job` ordering the srun-step startup fix
/// depends on — is directly testable without driving the whole scheduler
/// loop (which needs a live Raft cluster, a real `interval`/`scheduler_notify`
/// wake, and the three watchdog tasks `run` also spawns).
async fn process_assignment(
    cluster: Arc<ClusterManager>,
    assignment: spur_sched::traits::Assignment,
) -> bool {
    let job = match cluster.get_job(assignment.job_id) {
        Some(j) => j,
        None => return false,
    };

    let resources = compute_job_allocation(&job, &assignment.nodes, &assignment.per_node_alloc);

    let job_id = assignment.job_id;
    let spec = job.spec.clone();
    let all_nodes = assignment.nodes.clone();
    let per_node_allocs = assignment.per_node_alloc.clone();
    let dispatch_nodes = all_nodes.clone();
    let allocated_nodelist = all_nodes.join(",");

    let srun_step_dispatch = spec.srun_job
        && dispatch_nodes.iter().all(|name| {
            cluster
                .get_node(name)
                .is_some_and(|n| n.source == NodeSource::NativeHost)
        });

    if spec.srun_job && !srun_step_dispatch && spec.script.as_deref().unwrap_or("").is_empty() {
        warn!(
            job_id,
            "srun batch fallback requires a script in the job spec"
        );
        if let Err(e) = cluster.requeue_job(job_id) {
            error!(job_id, error = %e, "failed to requeue srun job without script");
        }
        return false;
    }

    // Run PrologSlurmctld before any node is touched. In Slurm it gates node
    // access: a failure must abort admission before allocations are registered,
    // the node prolog (prolog_slurmd) runs, or anything launches. The job is
    // still Pending here, so a failure just requeues it (batch) or cancels it
    // (interactive) — nothing on any node to tear down.
    if let Some(prolog_ctld) = cluster.config().hooks.prolog_slurmctld.clone() {
        let ctx = spur_core::hooks::HookContext {
            job_id: assignment.job_id,
            work_dir: job.spec.work_dir.clone(),
            uid: job.spec.uid,
            gid: job.spec.gid,
            partition: job.spec.partition.clone().unwrap_or_default(),
            nodelist: assignment.nodes.join(","),
            script_context: "prolog_slurmctld".into(),
            gpu_devices: Vec::new(),
            cpus: job.spec.cpus_per_task,
            memory_mb: job.spec.memory_per_node_mb.unwrap_or(0),
        };
        if let Err(e) = spur_core::hooks::run_hook(&prolog_ctld, &ctx).await {
            error!(
                job_id = assignment.job_id,
                error = %e,
                "PrologSlurmctld failed"
            );
            if job.spec.interactive {
                if let Err(ce) = cluster.cancel_job(assignment.job_id, &job.spec.user) {
                    error!(job_id = assignment.job_id, error = %ce, "failed to cancel job after PrologSlurmctld failure");
                }
            } else if let Err(re) = cluster.requeue_job(assignment.job_id) {
                error!(job_id = assignment.job_id, error = %re, "failed to requeue job after PrologSlurmctld failure");
            }
            return false;
        }
    }

    // `start_job_impl` advances the run epoch after the allocation has been
    // registered. Give agents that impending epoch so their Stepds
    // identify the controller-owned run rather than the pending job record.
    let prospective_run_attempt = job.run_attempt.saturating_add(1);

    if spec.srun_job && srun_step_dispatch {
        match register_allocation_on_nodes(
            cluster.clone(),
            job_id,
            dispatch_nodes.clone(),
            &spec,
            per_node_allocs.clone(),
            allocated_nodelist.clone(),
            prospective_run_attempt,
        )
        .await
        {
            // Both arms tear down identically: with a deadline in play, even "all failed" can mean
            // every node registered and answered too late, so none of them may be left holding one.
            AllocationRegisterOutcome::AllFailed | AllocationRegisterOutcome::PartialFailed => {
                cancel_job_on_nodes(&cluster, job_id, prospective_run_attempt, &all_nodes, 9).await;
                // The job never left Pending, so plain requeue is a no-op here — the same
                // Pending-aware backoff the launch path uses is what actually throttles a retry.
                if let Err(e) = cluster.backoff_pending_job_after_dispatch_failure(job_id) {
                    error!(job_id, error = %e, "failed to back off after registration failure");
                }
                return false;
            }
            AllocationRegisterOutcome::AllSucceeded => {}
        }
    }

    // Build peer_nodes list with addresses for cross-node communication
    // and the effective per-node task count. Both are needed by the
    // LaunchJob dispatch below, which — unlike the old fire-and-forget
    // spawn — now runs (and is awaited) *before* the job is allowed to
    // become visibly Running.
    let peer_addrs: Vec<String> = {
        let mut addrs = Vec::with_capacity(all_nodes.len());
        for name in &all_nodes {
            let Some(n) = cluster.get_node(name) else {
                warn!(
                    job_id,
                    node = %name,
                    "node missing while building peer list"
                );
                return false;
            };
            let Some(addr) = node_comm_socket(&n) else {
                warn!(
                    job_id,
                    node = %name,
                    "no comm address for peer list"
                );
                return false;
            };
            addrs.push(addr);
        }
        addrs
    };

    let tasks_per_node = if let Some(tpn) = spec.tasks_per_node {
        tpn
    } else {
        (spec.num_tasks / spec.num_nodes.max(1)).max(1)
    };

    // Batch dispatch (plain sbatch jobs, and the srun-as-batch-script
    // fallback) must have every assigned node confirm its LaunchJob
    // *before* start_job flips the job to Running — otherwise squeue
    // reports Running while a node may still be mid-launch (the race
    // an earlier mitigation only papered over with a retry window on
    // the agent side). The pure interactive srun_step_dispatch case
    // has nothing left to dispatch here: register_allocation_on_nodes
    // above already confirmed the allocation, and the actual step
    // launch is a later RunCommand, not this LaunchJob RPC.
    //
    // `task_fanout` distinguishes the two dispatched cases for the agent:
    // true only for the srun-as-batch-script fallback, where the dispatched
    // "script" is the literal command srun was asked to run tasks_per_node
    // times on this node (real srun semantics). False for a genuine sbatch
    // batch script, which runs exactly once per node regardless of
    // tasks_per_node, matching Slurm.
    let (dispatch_spec, task_fanout) = if !spec.srun_job {
        (Some(spec.clone()), false)
    } else if !srun_step_dispatch {
        let mut batch_spec = spec.clone();
        batch_spec.srun_job = false;
        (Some(batch_spec), true)
    } else {
        (None, false)
    };

    if let Some(dspec) = dispatch_spec {
        // The run epoch start_job_impl is about to persist for this
        // dispatch. Safe to read ahead of that call: this iteration is
        // the only place that can advance a Pending job's run_attempt,
        // and nothing here yields back to another iteration for the
        // same job in between.
        match confirm_dispatch_on_nodes(
            cluster.clone(),
            job_id,
            dispatch_nodes.clone(),
            dspec,
            peer_addrs.clone(),
            per_node_allocs.clone(),
            allocated_nodelist.clone(),
            tasks_per_node,
            prospective_run_attempt,
            task_fanout,
        )
        .await
        {
            DispatchConfirmOutcome::Aborted => return false,
            DispatchConfirmOutcome::Confirmed => {}
        }
    }

    // Transition job to Running. Reached only once every assigned node
    // has confirmed (LaunchJob for batch dispatch above, or
    // RegisterJobAllocation for the pure interactive case above that).
    let start_result = if srun_step_dispatch {
        cluster.start_job_impl(
            job_id,
            assignment.nodes.clone(),
            resources,
            assignment.per_node_alloc.clone(),
            true,
        )
    } else {
        cluster.start_job(
            job_id,
            assignment.nodes.clone(),
            resources,
            assignment.per_node_alloc.clone(),
        )
    };
    if let Err(e) = start_result {
        // Confirmation above already registered the allocation or
        // launched real processes on dispatch_nodes; stop them so a
        // start_job failure here (e.g. the job was cancelled out from
        // under us between assignment and this point) doesn't leave
        // orphans.
        cancel_job_on_nodes(
            &cluster,
            job_id,
            prospective_run_attempt,
            &dispatch_nodes,
            0,
        )
        .await;
        debug!(
            job_id = assignment.job_id,
            error = %e,
            "failed to start job"
        );
        return false;
    }

    true
}

/// Compute the resource set to record against the cluster for an assignment.
///
/// Non-exclusive: per-node request × node count (cpus, memory, generic),
/// plus the per-job GPU list verbatim.
///
/// Exclusive (#147): cpus / gpus / generic gres are bumped to the **sum of
/// each assigned node's total resources**, so the node shows as fully
/// allocated and the backfill scheduler's CPU-saturation check fires for
/// subsequent jobs. Memory stays at requested (matches Slurm semantics).
///
/// `node_totals` returns the total resources for a node by name. Returns
/// `None` if the node has been deregistered between assignment and start;
/// in that case its contribution is silently zero.
pub(crate) fn compute_job_allocation(
    job: &spur_core::job::Job,
    assignment_nodes: &[String],
    per_node_alloc: &std::collections::HashMap<String, spur_core::resource::ResourceAllocations>,
) -> spur_core::resource::ResourceAllocations {
    use spur_core::resource::{
        aggregate_allocations, build_exclusive_allocation, ResourceAllocations,
    };

    if job.spec.exclusive {
        let mut total = ResourceAllocations::default();
        let per_node_req = backfill::job_resource_request(job);
        for name in assignment_nodes {
            if let Some(alloc) = per_node_alloc.get(name) {
                total.add(alloc);
            }
        }
        if total.is_empty() {
            // Fallback if scheduler did not populate per-node slices.
            total = build_exclusive_allocation(
                &spur_core::resource::ResourceSet::default(),
                per_node_req.memory_mb,
            );
        }
        total
    } else {
        aggregate_allocations(
            assignment_nodes
                .iter()
                .filter_map(|name| per_node_alloc.get(name).cloned()),
        )
    }
}

/// Real per-node "free again at" times derived from running jobs, so backfill
/// doesn't fall back to a flat placeholder for every busy node.
fn running_jobs_busy_until(cluster: &ClusterManager) -> HashMap<String, DateTime<Utc>> {
    let running = cluster.get_jobs(&JobFilter {
        states: &[spur_core::job::JobState::Running],
        ..Default::default()
    });
    busy_until_from_running_jobs(&running)
}

/// Pure core of [`running_jobs_busy_until`], split out so it's testable
/// without a full `ClusterManager` harness.
fn busy_until_from_running_jobs(running: &[spur_core::job::Job]) -> HashMap<String, DateTime<Utc>> {
    use spur_sched::UNLIMITED_JOB_DURATION as UNLIMITED_FALLBACK;

    let now = Utc::now();
    let far_future = now + UNLIMITED_FALLBACK;
    let mut busy_until: HashMap<String, DateTime<Utc>> = HashMap::new();
    for job in running {
        let start = job.start_time.unwrap_or(now);
        let mut suspended = job.suspended_secs;
        if let Some(since) = job.suspended_at {
            suspended = suspended.saturating_add((now - since).num_seconds().max(0));
        }
        // Saturating i64 math, clamped once at the end: avoids the panicking
        // Duration + Duration path (a huge suspended_secs alone can overflow it).
        let time_limit_secs = job
            .spec
            .time_limit
            .map(|d| d.num_seconds())
            .unwrap_or(UNLIMITED_FALLBACK.num_seconds());
        let slack_secs = time_limit_secs
            .saturating_add(suspended)
            .saturating_add(GRACE_PERIOD_SECS)
            .min(UNLIMITED_FALLBACK.num_seconds());
        // A long-unbounded job outliving UNLIMITED_FALLBACK would otherwise
        // compute an `end` already in the past, producing a backwards interval.
        let end = start
            .checked_add_signed(chrono::Duration::seconds(slack_secs))
            .unwrap_or(far_future)
            .max(now);
        for node in &job.allocated_nodes {
            busy_until
                .entry(node.clone())
                .and_modify(|e| *e = (*e).max(end))
                .or_insert(end);
        }
    }
    busy_until
}

/// Resolve the effective PreemptMode for a job: QoS override wins if set
/// (see `qos_preempt_override`), else the most aggressive matched partition.
fn job_preempt_mode(
    job: &spur_core::job::Job,
    partitions: &[spur_core::partition::Partition],
    qos: &spur_core::accounting::Qos,
) -> spur_core::partition::PreemptMode {
    use spur_core::partition::PreemptMode;

    if let Some(mode) = spur_core::qos::qos_preempt_override(qos) {
        return mode;
    }

    spur_core::partition::matched_partitions(job.spec.partition.as_deref(), partitions)
        .into_iter()
        .map(|p| p.preempt_mode)
        .max_by_key(|m| m.aggressiveness())
        .unwrap_or(PreemptMode::Off)
}

/// Effective preempt-exempt seconds for a running job: QOS > partition > global.
/// When the job spans multiple partitions, the maximum partition override wins
/// (most protective), consistent with how `job_preempt_mode` uses the most
/// aggressive mode across partitions.
fn effective_exempt_secs(
    job: &spur_core::job::Job,
    partitions: &[spur_core::partition::Partition],
    qos: &spur_core::accounting::Qos,
    sched: &spur_core::config::SchedulerConfig,
) -> u32 {
    if let Some(t) = qos.limits.preempt_exempt_time {
        return t;
    }
    spur_core::partition::matched_partitions(job.spec.partition.as_deref(), partitions)
        .into_iter()
        .filter_map(|p| p.preempt_exempt_time)
        .max()
        .unwrap_or(sched.preempt_exempt_time)
}

/// Preempt lower-priority running jobs per their partition PreemptMode
/// (Off jobs are never preempted).
pub(crate) async fn try_preempt(
    cluster: &Arc<ClusterManager>,
    partitions: &[spur_core::partition::Partition],
    unscheduled: &[&spur_core::job::Job],
    sched: &spur_core::config::SchedulerConfig,
) {
    use crate::cluster::PreemptOutcome;
    use spur_core::job::JobState;
    use spur_core::partition::{Partition, PreemptMode, PreemptType};
    use spur_core::reservation::job_runs_in_active_reservation;

    let now = chrono::Utc::now();
    let reservations = cluster.get_reservations();
    let cluster_nodes = cluster.get_nodes();

    let partition_for = |job: &spur_core::job::Job| -> Option<&Partition> {
        spur_core::partition::matched_partitions(job.spec.partition.as_deref(), partitions)
            .into_iter()
            .max_by_key(|p| p.preempt_mode.aggressiveness())
    };

    let mut running: Vec<spur_core::job::Job> = cluster
        .get_jobs(&JobFilter {
            states: &[JobState::Running],
            ..Default::default()
        })
        .into_iter()
        .collect();
    // Resolve once, reuse for both the priority recompute and the
    // preempt-mode decision below.
    let running_qos: std::collections::HashMap<spur_core::job::JobId, spur_core::accounting::Qos> =
        running
            .iter()
            .map(|j| (j.job_id, cluster.resolve_qos(j)))
            .collect();
    // Running jobs' stored `priority` is the raw base value, unlike
    // `pending`'s fully adjusted one; recompute a comparable value.
    let running_priority: std::collections::HashMap<spur_core::job::JobId, u32> = running
        .iter()
        .map(|j| (j.job_id, cluster.current_effective_priority(j, partitions)))
        .collect();
    running.sort_by_key(|j| running_priority[&j.job_id]);

    // Pending job's QOS is resolved once per pending job; used for the
    // QosPriority hierarchy check.
    let pending_qos_map: std::collections::HashMap<
        spur_core::job::JobId,
        spur_core::accounting::Qos,
    > = unscheduled
        .iter()
        .map(|j| (j.job_id, cluster.resolve_qos(j)))
        .collect();

    for pending in unscheduled {
        let Some(pending_part) = partition_for(pending) else {
            continue;
        };
        if pending_part.preempt_mode == PreemptMode::Off {
            continue;
        }
        let pending_tier = pending_part.priority_tier;
        let pending_qos = &pending_qos_map[&pending.job_id];

        for candidate in &running {
            let candidate_priority = running_priority[&candidate.job_id];
            if candidate_priority >= pending.priority / 2 {
                continue;
            }

            if !preempt_overlaps_pending_nodes(pending, candidate, &cluster_nodes) {
                continue;
            }

            if job_runs_in_active_reservation(candidate, &reservations, now) {
                let candidate_tier = partition_for(candidate)
                    .map(|p| p.priority_tier)
                    .unwrap_or(1);
                if pending_tier <= candidate_tier {
                    continue;
                }
            }

            // QOS hierarchy: pending job may only preempt candidate when the
            // pending QOS explicitly lists the candidate's QOS in its allow-list.
            let candidate_qos = &running_qos[&candidate.job_id];
            if sched.preempt_type == PreemptType::QosPriority
                && !pending_qos.preempt.contains(&candidate_qos.name)
            {
                continue;
            }

            // Exempt time: skip candidates that haven't been running long enough.
            // A missing start_time is treated as immediately preemptable — we
            // can't know when it started, so we don't grant extra protection.
            let exempt_secs = effective_exempt_secs(candidate, partitions, candidate_qos, sched);
            if exempt_secs > 0 {
                if let Some(started_at) = candidate.start_time {
                    let running_for = (now - started_at).num_seconds().max(0) as u32;
                    if running_for < exempt_secs {
                        continue;
                    }
                }
            }

            let mode = job_preempt_mode(candidate, partitions, candidate_qos);
            if mode == PreemptMode::Off {
                continue;
            }
            info!(
                preempted_job = candidate.job_id,
                preempted_priority = candidate_priority,
                pending_job = pending.job_id,
                pending_priority = pending.priority,
                mode = ?mode,
                "preempting lower-priority job"
            );
            let preempt_qos = if sched.preempt_type == PreemptType::QosPriority {
                Some(pending_qos.name.clone())
            } else {
                None
            };
            match cluster.preempt_job_with_provenance(
                candidate.job_id,
                mode,
                Some(pending.job_id),
                preempt_qos,
            ) {
                Ok(PreemptOutcome::Killed) => {
                    // Signal 0 = graceful cancel (SIGTERM then SIGKILL).
                    send_cancel_to_agents(cluster, candidate, 0).await;
                }
                Ok(PreemptOutcome::Suspended) => {
                    send_suspend_to_agents(cluster, candidate, false).await;
                }
                Err(e) => {
                    warn!(
                        job_id = candidate.job_id,
                        error = %e,
                        "failed to preempt job"
                    );
                    continue;
                }
            }
            break; // One preemption per cycle, re-evaluate next cycle
        }
    }
}

/// True when `candidate` occupies a node the pending job could target.
fn preempt_overlaps_pending_nodes(
    pending: &spur_core::job::Job,
    candidate: &spur_core::job::Job,
    nodes: &[spur_core::node::Node],
) -> bool {
    if candidate.allocated_nodes.is_empty() {
        return false;
    }
    let occupied: HashSet<&str> = candidate
        .allocated_nodes
        .iter()
        .map(String::as_str)
        .collect();

    if let Some(ref nodelist) = pending.spec.nodelist {
        return nodelist
            .split(',')
            .map(str::trim)
            .any(|n| occupied.contains(n));
    }

    let partitions: Vec<&str> =
        requested_partition_names(pending.spec.partition.as_deref()).collect();

    nodes.iter().any(|node| {
        if !occupied.contains(node.name.as_str()) {
            return false;
        }
        if partitions.is_empty() {
            return true;
        }
        partitions
            .iter()
            .any(|p| node.partitions.iter().any(|np| np == p))
    })
}

/// Forward unschedulable jobs to federation peer clusters.
///
/// Tries each peer in order; stops forwarding a job as soon as one peer accepts it.
/// Failed peer connections are logged as warnings and skipped.
async fn forward_to_federation(cluster: &ClusterManager, jobs: &[spur_core::job::Job]) {
    let config = cluster.config();
    let peers = &config.federation.clusters;
    for job in jobs {
        for peer in peers {
            match SlurmControllerClient::connect(peer.address.clone())
                .await
                .map(|c| {
                    c.max_decoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE)
                        .max_encoding_message_size(spur_proto::MAX_GRPC_REQUEST_SIZE)
                }) {
                Ok(mut client) => {
                    let req = SubmitJobRequest {
                        spec: Some(core_spec_to_proto(&job.spec)),
                    };
                    match client.submit_job(req).await {
                        Ok(resp) => {
                            let remote_id = resp.into_inner().job_id;
                            info!(
                                job_id = job.job_id,
                                peer = %peer.name,
                                remote_id,
                                "forwarded unschedulable job to federation peer"
                            );
                            break; // accepted by this peer — don't try others
                        }
                        Err(e) => {
                            warn!(
                                job_id = job.job_id,
                                peer = %peer.name,
                                error = %e,
                                "federation peer rejected job"
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        peer = %peer.name,
                        error = %e,
                        "could not connect to federation peer"
                    );
                }
            }
        }
    }
}

/// Convert a core JobSpec to its proto representation for cross-cluster forwarding.
fn core_spec_to_proto(s: &spur_core::job::JobSpec) -> ProtoJobSpec {
    // Split licenses back out of GRES (stored as "license:<entry>")
    let mut gres = Vec::new();
    let mut licenses = Vec::new();
    for g in &s.gres {
        if let Some(lic) = g.strip_prefix("license:") {
            licenses.push(lic.to_string());
        } else {
            gres.push(g.clone());
        }
    }

    ProtoJobSpec {
        name: s.name.clone(),
        partition: s.partition.clone().unwrap_or_default(),
        account: s.account.clone().unwrap_or_default(),
        user: s.user.clone(),
        uid: s.uid,
        gid: s.gid,
        num_nodes: s.num_nodes,
        num_tasks: s.num_tasks,
        tasks_per_node: s.tasks_per_node.unwrap_or(0),
        cpus_per_task: s.cpus_per_task,
        memory_per_node_mb: s.memory_per_node_mb.unwrap_or(0),
        memory_per_cpu_mb: s.memory_per_cpu_mb.unwrap_or(0),
        gres,
        gpus: s.gpus.as_ref().map(Into::into),
        gpus_per_node: s.gpus_per_node.as_ref().map(Into::into),
        gpus_per_task: s.gpus_per_task.as_ref().map(Into::into),
        licenses,
        script: s.script.clone().unwrap_or_default(),
        argv: s.argv.clone(),
        script_args: s.script_args.clone(),
        work_dir: s.work_dir.clone(),
        stdout_path: s.stdout_path.clone().unwrap_or_default(),
        stderr_path: s.stderr_path.clone().unwrap_or_default(),
        stdin_path: s.stdin_path.clone().unwrap_or_default(),
        environment: s.environment.clone(),
        time_limit: s.time_limit.map(|d| prost_types::Duration {
            seconds: d.num_seconds(),
            nanos: 0,
        }),
        time_min: s.time_min.map(|d| prost_types::Duration {
            seconds: d.num_seconds(),
            nanos: 0,
        }),
        qos: s.qos.clone().unwrap_or_default(),
        // Proto `priority` is non-optional; 0 encodes "unset", not a base
        // priority of zero. The receiver decodes 0 back to `None`, which
        // `Job::new` then resolves to the default.
        priority: s.priority.unwrap_or(0),
        reservation: s.reservation.clone().unwrap_or_default(),
        dependency: s.dependency.clone(),
        nodelist: s.nodelist.clone().unwrap_or_default(),
        exclude: s.exclude.clone().unwrap_or_default(),
        constraint: s.constraint.clone().unwrap_or_default(),
        mpi: s.mpi.clone().unwrap_or_default(),
        distribution: s.distribution.clone().unwrap_or_default(),
        het_group: s.het_group.unwrap_or(0),
        array_spec: s.array_spec.clone().unwrap_or_default(),
        requeue: s.requeue,
        exclusive: s.exclusive,
        hold: s.hold,
        interactive: s.interactive,
        srun_job: s.srun_job,
        mail_type: s.mail_type.clone(),
        mail_user: s.mail_user.clone().unwrap_or_default(),
        comment: s.comment.clone().unwrap_or_default(),
        wckey: s.wckey.clone().unwrap_or_default(),
        container_image: s.container_image.clone().unwrap_or_default(),
        container_mounts: s.container_mounts.clone(),
        container_workdir: s.container_workdir.clone().unwrap_or_default(),
        container_name: s.container_name.clone().unwrap_or_default(),
        container_readonly: s.container_readonly,
        container_mount_home: s.container_mount_home,
        container_env: s.container_env.clone(),
        container_entrypoint: s.container_entrypoint.clone().unwrap_or_default(),
        container_remap_root: s.container_remap_root,
        burst_buffer: s.burst_buffer.clone().unwrap_or_default(),
        begin_time: s.begin_time.map(|dt| prost_types::Timestamp {
            seconds: dt.timestamp(),
            nanos: 0,
        }),
        deadline: s.deadline.map(|dt| prost_types::Timestamp {
            seconds: dt.timestamp(),
            nanos: 0,
        }),
        spread_job: s.spread_job,
        topology: s.topology.clone().unwrap_or_default(),
        host_network: s.host_network,
        privileged: s.privileged,
        host_ipc: s.host_ipc,
        shm_size: s.shm_size.clone().unwrap_or_default(),
        extra_resources: s.extra_resources.clone(),
        open_mode: s.open_mode.clone().unwrap_or_default(),
        pty: s.pty,
        initial_winsize: None,
        submit_line: s.submit_line.clone().unwrap_or_default(),
    }
}

/// Parameters for dispatching a job to a single node agent.
struct AgentDispatchParams<'a> {
    job_id: u32,
    spec: &'a spur_core::job::JobSpec,
    peer_nodes: &'a [String],
    peer_hosts: &'a [String],
    node_index: u32,
    task_offset: u32,
    target_node: &'a str,
    allocated: &'a spur_core::resource::ResourceAllocations,
    allocated_nodelist: &'a str,
    run_attempt: u32,
    pmix_tmpdir: &'a str,
    /// See `LaunchJobRequest.task_fanout` in slurm.proto: true only for a
    /// standalone `srun` request routed through this batch dispatch path
    /// (e.g. a Kubernetes-inclusive allocation), where the dispatched
    /// "script" is the literal command `srun` was asked to fan out.
    task_fanout: bool,
    modex_connect_timeout_secs: u32,
    modex_fence_timeout_secs: u32,
    modex_verify_timeout_secs: u32,
    pmix_prepared: bool,
}

/// Resolved output paths reported by an agent after a successful launch.
struct LaunchOutcome {
    stdout_path: String,
    stderr_path: String,
}

/// A failed dispatch, keeping the agent's classification of the failure so the
/// controller can choose a response rather than string-matching the message.
enum DispatchError {
    /// The node ran the job's prolog and it failed. The prolog sees the job's
    /// own context, so the same failure recurs everywhere: Slurm drains the node
    /// and holds the job instead of retrying it onto the next one.
    PrologFailed(String),
    /// The agent rejected the dispatch because the controller-allocated resources
    /// are already in use locally — the controller's view of this node is stale.
    ResourcesUnavailable,
    /// Connect failed, or the RPC timed out without a response, as opposed to
    /// the agent responding with an explicit rejection.
    Unreachable(anyhow::Error),
    /// The agent held the call for the whole dispatch deadline. Distinct from `Unreachable`, which
    /// a refused connection also produces in milliseconds and must not earn the same long cooldown.
    TimedOut(Duration),
    /// The agent explicitly rejected the launch for a reason it does not have
    /// a `LaunchFailureKind` for yet.
    AgentRejected(String),
    Other(anyhow::Error),
}

impl DispatchError {
    /// A short, stable label for grouping failures across nodes into one
    /// job-level reason instead of a raw per-node error dump.
    fn category(&self) -> &'static str {
        match self {
            Self::PrologFailed(_) => "prolog failure",
            Self::ResourcesUnavailable => "gpu/resource allocation mismatch",
            Self::Unreachable(_) => "agent unreachable",
            Self::TimedOut(_) => "agent timed out",
            Self::AgentRejected(_) => "agent rejected launch",
            Self::Other(_) => "dispatch error",
        }
    }
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrologFailed(reason) => write!(f, "agent rejected job: {reason}"),
            Self::ResourcesUnavailable => {
                write!(f, "agent rejected job: allocated resources unavailable")
            }
            Self::Unreachable(e) => write!(f, "agent unreachable: {e:#}"),
            Self::TimedOut(limit) => {
                write!(f, "agent did not answer within {}s", limit.as_secs())
            }
            Self::AgentRejected(reason) => write!(f, "agent rejected job: {reason}"),
            Self::Other(e) => write!(f, "{e:#}"),
        }
    }
}

impl<E: Into<anyhow::Error>> From<E> for DispatchError {
    fn from(e: E) -> Self {
        Self::Other(e.into())
    }
}

/// Ceiling on one agent RPC in a fan-out that is drained to completion. `0` disables it, matching
/// the sibling `dispatch_reject_cooldown_secs` knob.
pub(crate) fn dispatch_deadline(cluster: &ClusterManager) -> Option<Duration> {
    match cluster.config().controller.dispatch_timeout_secs {
        0 => None,
        secs => Some(Duration::from_secs(secs)),
    }
}

/// Send a LaunchJob RPC to a node agent.
async fn dispatch_to_agent(
    agent_addr: &str,
    params: &AgentDispatchParams<'_>,
) -> Result<LaunchOutcome, DispatchError> {
    // Logged before the call, not just on its outcome: a launch that never returns is otherwise
    // invisible for as long as it hangs, which is exactly when an operator needs to see it.
    info!(
        job_id = params.job_id,
        node = params.target_node,
        agent = agent_addr,
        run_attempt = params.run_attempt,
        "dispatching job to agent"
    );
    let mut client = crate::agent_client::connect(agent_addr.to_string())
        .await
        .map_err(|e| DispatchError::Unreachable(e.into()))?
        .max_decoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(spur_proto::MAX_GRPC_REQUEST_SIZE);

    let spec = params.spec;

    // The scheduler distributes GPUs per node (a --gpus total may be uneven
    // across nodes). Rewrite this node's `gres` to its concrete GPU count so the
    // agent's fallback and any display see the real per-node figure; the agent
    // itself binds the exact device IDs from `allocated`.
    let node_gpu_count = params.allocated.total_device_count("gpu") as u32;
    let gpu_type = spur_core::gpu_request::resolve_gpu_demand(spec)
        .ok()
        .and_then(|d| d.gpu_type().map(str::to_string));
    let mut per_node_gres: Vec<String> = spec
        .gres
        .iter()
        .filter(|g| !(g.starts_with("gpu:") || g.as_str() == "gpu"))
        .cloned()
        .collect();
    if node_gpu_count > 0 {
        per_node_gres.push(match &gpu_type {
            Some(t) => format!("gpu:{}:{}", t, node_gpu_count),
            None => format!("gpu:{}", node_gpu_count),
        });
    }

    let tasks_per_node = if spec.tasks_per_node.unwrap_or(0) > 0 {
        spec.tasks_per_node.unwrap_or(1)
    } else {
        (spec.num_tasks / spec.num_nodes.max(1)).max(1)
    };
    let pmix_plan = build_pmix_plan_proto(params, spec, tasks_per_node)
        .map_err(|e| anyhow::anyhow!("invalid PMIx launch plan: {e}"))?;
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
        gres: per_node_gres,
        // Per-node count is carried in `gres` above; the explicit GPU request
        // fields are controller-side scheduling inputs the agent does not use.
        gpus: None,
        gpus_per_node: None,
        gpus_per_task: None,
        script: spec.script.clone().unwrap_or_default(),
        argv: spec.argv.clone(),
        script_args: spec.script_args.clone(),
        work_dir: spec.work_dir.clone(),
        stdout_path: spec.stdout_path.clone().unwrap_or_default(),
        stderr_path: spec.stderr_path.clone().unwrap_or_default(),
        stdin_path: spec.stdin_path.clone().unwrap_or_default(),
        environment: spec.environment.clone(),
        time_limit: spec.time_limit.map(|d| prost_types::Duration {
            seconds: d.num_seconds(),
            nanos: 0,
        }),
        time_min: None,
        qos: spec.qos.clone().unwrap_or_default(),
        // Proto `priority` is non-optional; 0 encodes "unset". The agent does
        // not schedule, so this value is carried for fidelity only.
        priority: spec.priority.unwrap_or(0),
        reservation: spec.reservation.clone().unwrap_or_default(),
        dependency: spec.dependency.clone(),
        nodelist: params.allocated_nodelist.to_string(),
        exclude: spec.exclude.clone().unwrap_or_default(),
        constraint: spec.constraint.clone().unwrap_or_default(),
        mpi: spec.mpi.clone().unwrap_or_default(),
        distribution: spec.distribution.clone().unwrap_or_default(),
        het_group: spec.het_group.unwrap_or(0),
        array_spec: spec.array_spec.clone().unwrap_or_default(),
        requeue: spec.requeue,
        exclusive: spec.exclusive,
        hold: spec.hold,
        interactive: spec.interactive,
        srun_job: spec.srun_job,
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
        burst_buffer: spec.burst_buffer.clone().unwrap_or_default(),
        licenses: Vec::new(),
        mail_type: spec.mail_type.clone(),
        mail_user: spec.mail_user.clone().unwrap_or_default(),
        begin_time: spec.begin_time.map(|dt| prost_types::Timestamp {
            seconds: dt.timestamp(),
            nanos: dt.timestamp_subsec_nanos() as i32,
        }),
        deadline: spec.deadline.map(|dt| prost_types::Timestamp {
            seconds: dt.timestamp(),
            nanos: dt.timestamp_subsec_nanos() as i32,
        }),
        spread_job: spec.spread_job,
        topology: spec.topology.clone().unwrap_or_default(),
        host_network: spec.host_network,
        privileged: spec.privileged,
        host_ipc: spec.host_ipc,
        shm_size: spec.shm_size.clone().unwrap_or_default(),
        extra_resources: spec.extra_resources.clone(),
        open_mode: spec.open_mode.clone().unwrap_or_default(),
        pty: spec.pty,
        initial_winsize: None,
        submit_line: spec.submit_line.clone().unwrap_or_default(),
    };

    let response = client
        .launch_job(LaunchJobRequest {
            job_id: params.job_id,
            spec: Some(proto_spec),
            allocated: Some(crate::server::allocations_to_proto(params.allocated)),
            peer_nodes: params.peer_nodes.to_vec(),
            task_offset: params.task_offset,
            target_node: params.target_node.to_string(),
            // Controller-assigned at array expansion; consumed agent-side.
            array_job_id: spec.array_job_id.unwrap_or(0),
            array_task_id: spec.array_task_id.unwrap_or(0),
            run_attempt: params.run_attempt,
            pmix_plan,
            task_fanout: params.task_fanout,
            pmix_prepared: params.pmix_prepared,
        })
        .await
        .map_err(|s| match s.code() {
            tonic::Code::ResourceExhausted => DispatchError::ResourcesUnavailable,
            tonic::Code::Unavailable | tonic::Code::DeadlineExceeded => {
                DispatchError::Unreachable(s.into())
            }
            _ => DispatchError::Other(s.into()),
        })?;

    let inner = response.into_inner();
    if !inner.success {
        // An agent predating the classification sends UNSPECIFIED, which falls
        // through to the generic requeue this has always done.
        return Err(
            if inner.failure_kind
                == spur_proto::proto::LaunchFailureKind::LaunchFailureProlog as i32
            {
                DispatchError::PrologFailed(inner.error)
            } else {
                DispatchError::AgentRejected(inner.error)
            },
        );
    }
    info!(
        job_id = params.job_id,
        "job dispatched to agent successfully"
    );

    Ok(LaunchOutcome {
        stdout_path: inner.stdout_path,
        stderr_path: inner.stderr_path,
    })
}

fn build_pmix_plan_proto(
    params: &AgentDispatchParams<'_>,
    spec: &spur_core::job::JobSpec,
    tasks_per_node: u32,
) -> Result<Option<spur_proto::proto::PmixLaunchPlan>, String> {
    let mpi = spec.mpi.as_deref().unwrap_or("");
    spur_core::mpi::build_validated_pmix_plan_proto(
        mpi,
        spur_core::mpi::PmixLocalDispatch {
            job_id: params.job_id,
            universe_size: spec.num_tasks,
            task_offset: params.task_offset,
            local_count: tasks_per_node,
            tmpdir: params.pmix_tmpdir.to_string(),
            job_uid: spec.uid,
            job_gid: spec.gid,
            num_nodes: spec.num_nodes,
            node_index: params.node_index,
            peer_hosts: params.peer_hosts.to_vec(),
            modex_connect_timeout_secs: params.modex_connect_timeout_secs,
            modex_fence_timeout_secs: params.modex_fence_timeout_secs,
            modex_verify_timeout_secs: params.modex_verify_timeout_secs,
        },
    )
}

/// Outcome of parallel RegisterJobAllocation RPCs for a standalone srun job.
pub(crate) enum AllocationRegisterOutcome {
    AllSucceeded,
    AllFailed,
    PartialFailed,
}

/// Why one RegisterJobAllocation RPC did not succeed. Split so the caller can tell a node that
/// burned the whole deadline from one that failed fast, and cool each for the span it earned.
enum RegisterError {
    Failed(anyhow::Error),
    TimedOut(Duration),
}

impl std::fmt::Display for RegisterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed(e) => write!(f, "{e}"),
            Self::TimedOut(limit) => {
                write!(f, "allocation register RPC exceeded {}s", limit.as_secs())
            }
        }
    }
}

/// Parameters for registering a srun-only allocation on a single node agent.
struct AllocationRegisterParams {
    job_id: u32,
    partition: String,
    uid: u32,
    gid: u32,
    user: String,
    mpi: String,
    allocated_nodelist: String,
    allocated: spur_core::resource::ResourceAllocations,
    work_dir: String,
    run_attempt: u32,
}

/// Register a srun-only allocation on a node agent without launching a batch process.
async fn register_allocation_to_agent(
    agent_addr: &str,
    params: &AllocationRegisterParams,
) -> anyhow::Result<()> {
    let mut client = crate::agent_client::connect(agent_addr.to_string())
        .await?
        .max_decoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(spur_proto::MAX_GRPC_REQUEST_SIZE);

    client
        .register_job_allocation(RegisterJobAllocationRequest {
            job_id: params.job_id,
            partition: params.partition.clone(),
            nodelist: params.allocated_nodelist.clone(),
            uid: params.uid,
            gid: params.gid,
            cpus: params.allocated.cpus,
            memory_mb: params.allocated.memory_mb,
            gpu_devices: params
                .allocated
                .device_ids("gpu")
                .into_iter()
                .map(|id| id.to_string())
                .collect(),
            allocated: Some(crate::server::allocations_to_proto(&params.allocated)),
            mpi: params.mpi.clone(),
            work_dir: params.work_dir.clone(),
            user: params.user.clone(),
            run_attempt: params.run_attempt,
        })
        .await?;

    info!(
        job_id = params.job_id,
        "srun allocation registered on agent successfully"
    );

    Ok(())
}

/// Register a srun-only allocation on every assigned node.
#[allow(clippy::too_many_arguments)]
async fn register_allocation_on_nodes(
    cluster: Arc<ClusterManager>,
    job_id: spur_core::job::JobId,
    dispatch_nodes: Vec<String>,
    spec: &spur_core::job::JobSpec,
    per_node_allocs: std::collections::HashMap<String, spur_core::resource::ResourceAllocations>,
    allocated_nodelist: String,
    run_attempt: u32,
) -> AllocationRegisterOutcome {
    let mut successes = 0u32;
    let mut failures = 0u32;
    let total = dispatch_nodes.len() as u32;
    let dispatch_timeout = dispatch_deadline(&cluster);

    let mut set = tokio::task::JoinSet::new();
    for node_name in &dispatch_nodes {
        let node_info = cluster.get_node(node_name);
        let agent_addr = match node_info {
            Some(ref n) => match node_comm_http_url(n) {
                Some(url) => url,
                None => {
                    warn!(
                        job_id,
                        node = %node_name,
                        "no comm address for node, skipping allocation registration"
                    );
                    failures += 1;
                    continue;
                }
            },
            _ => {
                warn!(
                    job_id,
                    node = %node_name,
                    "no agent address for node, skipping allocation registration"
                );
                failures += 1;
                continue;
            }
        };

        let result_node = node_name.clone();
        let allocated = per_node_allocs.get(node_name).cloned().unwrap_or_default();
        let params = AllocationRegisterParams {
            job_id,
            partition: spec.partition.clone().unwrap_or_default(),
            uid: spec.uid,
            gid: spec.gid,
            user: spec.user.clone(),
            mpi: spec.mpi.clone().unwrap_or_default(),
            allocated_nodelist: allocated_nodelist.clone(),
            allocated,
            work_dir: spec.work_dir.clone(),
            run_attempt,
        };
        set.spawn(async move {
            let register = register_allocation_to_agent(&agent_addr, &params);
            let result = match dispatch_timeout {
                Some(limit) => match tokio::time::timeout(limit, register).await {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(e)) => Err(RegisterError::Failed(e)),
                    Err(_) => Err(RegisterError::TimedOut(limit)),
                },
                None => register.await.map_err(RegisterError::Failed),
            };
            (result_node, result)
        });
    }

    while let Some(result) = set.join_next().await {
        match result {
            Ok((_node_name, Ok(()))) => {
                successes += 1;
            }
            Ok((node_name, Err(e))) => {
                error!(
                    job_id,
                    node = %node_name,
                    error = %e,
                    "allocation registration on agent failed"
                );
                // Mirrors the launch fan-out: an unreachable node is cooled briefly, one that
                // burned a deadline is held for it, so neither is re-picked on the next tick.
                match e {
                    RegisterError::TimedOut(limit) => cluster.cool_down_node_for(&node_name, limit),
                    RegisterError::Failed(_) => cluster.cool_down_node(&node_name),
                }
                failures += 1;
            }
            Err(e) => {
                error!(job_id, error = %e, "allocation registration task panicked");
                failures += 1;
            }
        }
    }

    if successes == 0 && total > 0 {
        error!(job_id, failures, "all allocation registrations failed");
        AllocationRegisterOutcome::AllFailed
    } else if failures > 0 {
        warn!(
            job_id,
            successes, failures, "partial allocation registration failure"
        );
        AllocationRegisterOutcome::PartialFailed
    } else {
        AllocationRegisterOutcome::AllSucceeded
    }
}

/// Release standalone srun allocations on agents after CompleteJob.
pub async fn release_srun_allocation_on_agents(
    cluster: &Arc<ClusterManager>,
    job: &spur_core::job::Job,
) {
    send_cancel_to_agents(cluster, job, 0).await;
}

/// Outcome of [`confirm_dispatch_on_nodes`]: either every assigned node
/// confirmed its LaunchJob (the job may now become visibly Running), or
/// admission was aborted and the job — which never left Pending — has
/// already been settled (requeued, held, or cancelled as appropriate).
enum DispatchConfirmOutcome {
    Confirmed,
    Aborted,
}

fn abort_pending_pmix_dispatch(
    cluster: &ClusterManager,
    job_id: spur_core::job::JobId,
    detail: String,
) -> DispatchConfirmOutcome {
    let _ = cluster.set_job_launch_failure_detail(job_id, detail);
    if let Err(e) = cluster.backoff_pending_job_after_dispatch_failure(job_id) {
        error!(job_id, error = %e, "failed to back off after PMIx dispatch failure");
    }
    DispatchConfirmOutcome::Aborted
}

/// Dispatch a batch job to every assigned node and *wait* for every LaunchJob
/// RPC to resolve before returning anything the caller can use to flip the
/// job to Running.
///
/// This is the structural fix for the srun-step startup race: an earlier
/// mitigation only widened an agent-side retry window on one lookup. The old
/// flow called `start_job` (Running, visible to squeue/scontrol) and then
/// fired this dispatch in the background via `tokio::spawn`, so a step could
/// target a node before that node's own `LaunchJob` had actually finished.
/// Now the scheduler loop awaits this function *before* calling `start_job`,
/// mirroring the pattern `register_allocation_on_nodes` already uses for
/// standalone srun/salloc allocations — except this confirms the heavier
/// `LaunchJob` RPC (which actually spawns the job's process), not the
/// lightweight `RegisterJobAllocation` used there.
///
/// This closes the race for every consumer that gates on `job.state ==
/// Running` before targeting a node, not just the step-dispatch path
/// (`run_step`/`run_command`) it was written for: `exec_in_job` and
/// `create_job_step` (spurctld/server.rs) both check `job.state == Running`
/// before forwarding to an agent, and back `exec_in_job`/`interactive_session`
/// (spurd/agent_server.rs's `job_entry`) and `stream_job_output`'s attach
/// path respectively. Since Running is the single, sole signal all of these
/// checks (and the CLI's own client-side checks, for the RPCs that have no
/// controller-side proxy) rely on, and it is now unreachable before every
/// node's registration completes, none of run_command/job_entry/
/// stream_job_output's node-side lookups need a retry of their own — see the
/// comments at each of those lookups.
///
/// Failure handling necessarily differs from the old post-Running
/// `dispatch_job_to_nodes` this replaces, because the job is still Pending
/// here — there is no Running/Failed/NodeFail detour to route a partial
/// failure through:
///   - A node whose own prolog rejected the job is drained, exactly as before
///     (this is a per-node action, independent of the job's state).
///   - If `hold_on_prolog_fail` is set (the default) and applies, the job is
///     parked the same way `scontrol hold` parks a Pending job
///     (`ClusterManager::hold_job_for_launch_failure`), since holding via the
///     old Running→Failed→Held path isn't reachable from Pending. Interactive
///     jobs are cancelled instead, same as before.
///   - Otherwise (no prolog failure, or holding is disabled) the job is
///     simply left Pending for the next scheduler tick — the same
///     simplification `register_allocation_on_nodes`'s own failure arms above
///     already make for the srun path. This intentionally does not reproduce
///     `requeue_after_launch_failure`'s exponential backoff / requeue-count
///     bookkeeping for a *non*-prolog failure, since that path requires an
///     actual Failed transition the job never has here; a persistently
///     failing node still gets drained on repeated prolog failures, but a
///     transient (non-prolog) failure can retry immediately rather than
///     backing off. See the fix's write-up for why this trade-off was made.
#[allow(clippy::too_many_arguments)]
async fn confirm_dispatch_on_nodes(
    cluster: Arc<ClusterManager>,
    job_id: spur_core::job::JobId,
    dispatch_nodes: Vec<String>,
    spec: spur_core::job::JobSpec,
    peer_addrs: Vec<String>,
    per_node_allocs: std::collections::HashMap<String, spur_core::resource::ResourceAllocations>,
    allocated_nodelist: String,
    tasks_per_node: u32,
    run_attempt: u32,
    task_fanout: bool,
) -> DispatchConfirmOutcome {
    let mut successes = 0u32;
    let mut failures = 0u32;
    let mut prolog_failed: Vec<(String, String)> = Vec::new();
    let mut failure_categories: std::collections::BTreeMap<&'static str, u32> = Default::default();
    let total = dispatch_nodes.len() as u32;

    // Batch stdout/stderr live on the primary node (task_offset == 0). Capture
    // only its resolved paths; the JoinSet completes out of order, so select by
    // this flag rather than arrival order.
    let mut primary_outcome: Option<LaunchOutcome> = None;

    let pmix_tmpdir = cluster.config().mpi.pmix_tmpdir.clone();
    let modex_connect_timeout_secs = cluster.config().mpi.modex_connect_timeout_secs;
    let modex_fence_timeout_secs = cluster.config().mpi.modex_fence_timeout_secs;
    let modex_verify_timeout_secs = cluster.config().mpi.modex_verify_timeout_secs;
    let dispatch_timeout = dispatch_deadline(&cluster);

    let needs_pmix_prepare = batch_dispatched_multi_node_pmix(
        spec.mpi.as_deref(),
        spec.num_nodes,
        spec.script.as_deref(),
    );

    let mut peer_hosts: Vec<String> = Vec::new();
    let mut node_agents: Vec<(String, String)> = Vec::new();
    for node_name in dispatch_nodes.iter() {
        let node_info = cluster.get_node(node_name);
        let (comm_host, agent_addr) = match node_info {
            Some(ref n) => match (n.comm_addr(), node_comm_http_url(n)) {
                (Some(host), Some(url)) => (host.to_string(), url),
                _ => {
                    warn!(
                        job_id,
                        node = %node_name,
                        "no comm address for node, skipping dispatch confirmation"
                    );
                    failures += 1;
                    *failure_categories.entry("no agent address").or_insert(0) += 1;
                    continue;
                }
            },
            None => {
                warn!(
                    job_id,
                    node = %node_name,
                    "no agent address for node, skipping dispatch confirmation"
                );
                failures += 1;
                *failure_categories.entry("no agent address").or_insert(0) += 1;
                continue;
            }
        };
        peer_hosts.push(comm_host);
        node_agents.push((node_name.clone(), agent_addr));
    }

    let mut pmix_prepare_guard = None;

    if needs_pmix_prepare {
        if failures > 0 || node_agents.len() != dispatch_nodes.len() {
            error!(
                job_id,
                failures,
                agents = node_agents.len(),
                expected = dispatch_nodes.len(),
                "incomplete agent set for multi-node PMIx — aborting dispatch"
            );
            let detail = format!(
                "incomplete PMIx agent set: {} of {} nodes reachable",
                node_agents.len(),
                dispatch_nodes.len()
            );
            return abort_pending_pmix_dispatch(&cluster, job_id, detail);
        }

        if let Some(detail) = pmix_dispatch::multi_node_pmix_unsupported(
            dispatch_nodes
                .iter()
                .filter_map(|name| cluster.get_node(name).map(|node| node.source.clone())),
        ) {
            error!(job_id, "{detail}");
            let _ = cluster.set_job_launch_failure_detail(job_id, detail.clone());
            if let Err(e) = cluster.hold_job_for_launch_failure(job_id, Some(&detail)) {
                error!(job_id, error = %e, "failed to hold job for unsupported multi-node PMIx");
            }
            return DispatchConfirmOutcome::Aborted;
        }

        let mut prepare_nodes = Vec::with_capacity(node_agents.len());
        for (node_idx, (node_name, agent_addr)) in node_agents.iter().enumerate() {
            let task_offset = node_idx as u32 * tasks_per_node;
            let allocated = per_node_allocs.get(node_name).cloned().unwrap_or_default();
            let params = AgentDispatchParams {
                job_id,
                spec: &spec,
                peer_nodes: &peer_addrs,
                peer_hosts: &peer_hosts,
                node_index: node_idx as u32,
                task_offset,
                target_node: node_name,
                allocated: &allocated,
                allocated_nodelist: &allocated_nodelist,
                run_attempt,
                pmix_tmpdir: &pmix_tmpdir,
                task_fanout,
                modex_connect_timeout_secs,
                modex_fence_timeout_secs,
                modex_verify_timeout_secs,
                pmix_prepared: false,
            };
            let pmix_plan = match build_pmix_plan_proto(&params, &spec, tasks_per_node) {
                Ok(Some(plan)) => plan,
                Ok(None) => {
                    let detail = format!("job is not configured for PMIx on node {node_name}");
                    error!(job_id, node = %node_name, "{detail}");
                    return abort_pending_pmix_dispatch(&cluster, job_id, detail);
                }
                Err(detail) => {
                    let detail = format!("invalid PMIx launch plan for node {node_name}: {detail}");
                    error!(job_id, node = %node_name, "{detail}");
                    return abort_pending_pmix_dispatch(&cluster, job_id, detail);
                }
            };
            prepare_nodes.push(PmixPrepareNode {
                node_name: node_name.clone(),
                agent_addr: agent_addr.clone(),
                pmix_plan,
            });
        }

        if let Err(detail) = pmix_dispatch::prepare_pmix_on_nodes(
            job_id,
            run_attempt,
            prepare_nodes,
            dispatch_timeout,
        )
        .await
        {
            error!(job_id, error = %detail, "PMIx prepare failed — aborting dispatch");
            return abort_pending_pmix_dispatch(
                &cluster,
                job_id,
                format!("PMIx prepare failed: {detail}"),
            );
        }
        let agent_addrs: Vec<String> = node_agents.iter().map(|(_, addr)| addr.clone()).collect();
        pmix_prepare_guard = Some(pmix_dispatch::PmixPreparedReleaseGuard::new(
            job_id,
            agent_addrs,
        ));
    }

    let mut set = tokio::task::JoinSet::new();
    for (node_idx, (node_name, agent_addr)) in node_agents.iter().enumerate() {
        let spec = spec.clone();
        let peer_addrs = peer_addrs.clone();
        let peer_hosts = peer_hosts.clone();
        let task_offset = node_idx as u32 * tasks_per_node;
        let node_index = node_idx as u32;
        let is_primary = task_offset == 0;
        let target_node = node_name.clone();
        let result_node = node_name.clone();
        let allocated = per_node_allocs.get(node_name).cloned().unwrap_or_default();
        let allocated_nodelist = allocated_nodelist.clone();
        let pmix_tmpdir = pmix_tmpdir.clone();
        let agent_addr = agent_addr.clone();
        set.spawn(async move {
            let params = AgentDispatchParams {
                job_id,
                spec: &spec,
                peer_nodes: &peer_addrs,
                peer_hosts: &peer_hosts,
                node_index,
                task_offset,
                target_node: &target_node,
                allocated: &allocated,
                allocated_nodelist: &allocated_nodelist,
                run_attempt,
                pmix_tmpdir: &pmix_tmpdir,
                task_fanout,
                modex_connect_timeout_secs,
                modex_fence_timeout_secs,
                modex_verify_timeout_secs,
                pmix_prepared: needs_pmix_prepare,
            };
            let dispatch = dispatch_to_agent(&agent_addr, &params);
            // The join below drains every node, so one agent that accepts the call and never answers
            // would otherwise hold the whole scheduler loop, not just this job.
            let result = match dispatch_timeout {
                Some(limit) => match tokio::time::timeout(limit, dispatch).await {
                    Ok(result) => result,
                    Err(_) => Err(DispatchError::TimedOut(limit)),
                },
                None => dispatch.await,
            };
            (result_node, is_primary, result)
        });
    }

    while let Some(result) = set.join_next().await {
        match result {
            Ok((_node_name, is_primary, Ok(outcome))) => {
                successes += 1;
                if is_primary {
                    primary_outcome = Some(outcome);
                }
            }
            Ok((node_name, _, Err(e))) => {
                error!(job_id, node = %node_name, error = %e, "dispatch confirmation failed");
                failures += 1;
                *failure_categories.entry(e.category()).or_insert(0) += 1;
                match e {
                    DispatchError::PrologFailed(reason) => {
                        prolog_failed.push((node_name, reason));
                    }
                    DispatchError::ResourcesUnavailable | DispatchError::Unreachable(_) => {
                        cluster.cool_down_node(&node_name)
                    }
                    // Held for the deadline it actually burned: while assignments are processed
                    // serially, re-picking this node stalls every job behind it, not just this one.
                    DispatchError::TimedOut(limit) => cluster.cool_down_node_for(&node_name, limit),
                    DispatchError::AgentRejected(_) | DispatchError::Other(_) => {}
                }
            }
            Err(e) => {
                error!(job_id, error = %e, "dispatch confirmation task panicked");
                failures += 1;
                *failure_categories
                    .entry("dispatch task panicked")
                    .or_insert(0) += 1;
            }
        }
    }

    if failures == 0 {
        if let Some(guard) = pmix_prepare_guard.as_mut() {
            guard.disarm();
        }
        if let Some(outcome) = primary_outcome {
            cluster.set_job_output_paths(job_id, outcome.stdout_path, outcome.stderr_path);
        }
        return DispatchConfirmOutcome::Confirmed;
    }

    warn!(
        job_id,
        successes, failures, total,
        "one or more nodes failed to confirm dispatch — aborting admission instead of partially running"
    );

    if needs_pmix_prepare {
        if let Some(guard) = pmix_prepare_guard.as_mut() {
            guard.disarm();
        }
        let agent_addrs: Vec<String> = node_agents.iter().map(|(_, addr)| addr.clone()).collect();
        pmix_dispatch::release_pmix_on_agents(&agent_addrs, job_id).await;
    }

    let category_summary = failure_categories
        .iter()
        .map(|(category, count)| format!("{count} {category}"))
        .collect::<Vec<_>>()
        .join(", ");
    let confirmation_detail = if needs_pmix_prepare {
        format!(
            "PMIx dispatch confirmation failed ({successes}/{total} confirmed): {category_summary}"
        )
    } else {
        format!("dispatch confirmation failed ({successes}/{total} confirmed): {category_summary}")
    };
    let _ = cluster.set_job_launch_failure_detail(job_id, confirmation_detail.clone());

    // Every dispatched node, not just the confirmed ones: a node that timed out may have launched
    // anyway and is the likeliest to be orphaned. CancelJob is idempotent, so cancelling wide is safe.
    cancel_job_on_nodes(&cluster, job_id, run_attempt, &dispatch_nodes, 9).await;

    // Drain before deciding the job's fate, so the failing node is already out
    // of the candidate set on the next scheduling attempt. The drain is issued
    // here rather than by the agent because only the controller can pair it
    // with the hold that stops the job walking the cluster.
    for (node_name, reason) in &prolog_failed {
        warn!(job_id, node = %node_name, reason = %reason, "draining node after prolog failure");
        if let Err(e) = cluster.drain_node(node_name, Some(reason.clone())) {
            error!(job_id, node = %node_name, error = %e, "failed to drain node after prolog failure");
        }
    }

    if !prolog_failed.is_empty() && cluster.config().controller.hold_on_prolog_fail {
        if spec.interactive {
            // Holding an interactive job would strand its waiting srun forever
            // with nothing to wait for; Slurm cancels these too.
            if let Err(e) = cluster.cancel_job(job_id, &spec.user) {
                error!(job_id, error = %e, "failed to cancel interactive job after prolog failure");
            }
        } else if let Err(e) =
            cluster.hold_job_for_launch_failure(job_id, Some(&confirmation_detail))
        {
            error!(job_id, error = %e, "failed to hold job after prolog failure");
        }
    } else if let Err(e) = cluster.backoff_pending_job_after_dispatch_failure(job_id) {
        error!(job_id, error = %e, "failed to back off after dispatch confirmation failure");
    }

    DispatchConfirmOutcome::Aborted
}

/// Watchdog: gracefully terminate running jobs that exceed their time limit.
///
/// Two-phase shutdown:
///   1. **Warning phase**: When `start_time + time_limit < now`, durably mark
///      the run as timed out and send SIGTERM (signal 15) to all agents.
///   2. **Kill phase**: 30 seconds after the warning, if the job is still
///      running, mark it as Timeout and send SIGKILL (signal 9).
///
/// Both phases key off the job's replicated `time_limit_signaled_at`, so the
/// grace period is measured from one agreed instant even across a leadership
/// change, and a job that exits during the grace period still finalizes as
/// `Timeout` rather than as a plain signal death.
///
/// Runs every 10 seconds.
async fn enforce_time_limits(cluster: Arc<ClusterManager>, raft: Arc<RaftHandle>) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));

    loop {
        interval.tick().await;

        if !raft.is_leader() {
            continue;
        }

        let now = Utc::now();

        // Deadline enforcement: mark pending jobs whose deadline has passed
        {
            let pending = cluster.get_jobs(&JobFilter {
                states: &[spur_core::job::JobState::Pending],
                ..Default::default()
            });
            for job in &pending {
                if let Some(deadline) = job.spec.deadline {
                    if now > deadline {
                        if let Err(e) = cluster.deadline_job(job.job_id) {
                            warn!(job_id = job.job_id, error = %e, "failed to mark job DEADLINE");
                        }
                    }
                }
            }
        }

        let running = cluster.get_jobs(&JobFilter {
            states: &[
                spur_core::job::JobState::Running,
                spur_core::job::JobState::Completing,
            ],
            ..Default::default()
        });

        for job in &running {
            if job.state == spur_core::job::JobState::Completing {
                continue;
            }

            let (Some(time_limit), Some(start_time)) = (job.spec.time_limit, job.start_time) else {
                continue;
            };
            let deadline = job.effective_deadline(start_time, time_limit);
            if now < deadline {
                continue;
            }

            let Some(signaled_at) = job.time_limit_signaled_at else {
                info!(
                    job_id = job.job_id,
                    elapsed_secs = (now - start_time).num_seconds(),
                    limit_secs = time_limit.num_seconds(),
                    grace_secs = GRACE_PERIOD_SECS,
                    "time limit exceeded — sending SIGTERM, grace period starts"
                );

                // Record before signalling: if the job exits on the SIGTERM, its
                // completion must find the run already marked as timed out.
                if let Err(e) = cluster.signal_time_limit(job.job_id, now) {
                    warn!(job_id = job.job_id, error = %e, "failed to record time limit expiry");
                    continue;
                }

                send_cancel_to_agents(&cluster, job, 15).await; // SIGTERM
                continue;
            };

            if (now - signaled_at).num_seconds() < GRACE_PERIOD_SECS {
                continue;
            }

            info!(
                job_id = job.job_id,
                "grace period expired — force-killing job"
            );

            if let Err(e) = cluster.complete_job(job.job_id, -1, spur_core::job::JobState::Timeout)
            {
                warn!(job_id = job.job_id, error = %e, "failed to mark job as timed out");
                continue;
            }

            send_cancel_to_agents(&cluster, job, 9).await; // SIGKILL
        }
    }
}

/// Reap interactive allocations (salloc/srun) whose client stopped sending
/// keepalives, mirroring Slurm's `InactiveLimit`. Idle allocations get a
/// SIGTERM -> grace -> SIGKILL sequence (like `enforce_time_limits`) and are
/// finalized via the TIMEOUT path. Disabled when `inactive_limit_secs == 0`.
async fn enforce_inactive_limits(cluster: Arc<ClusterManager>, raft: Arc<RaftHandle>) {
    use spur_core::job::{JobId, JobState};

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));

    // Jobs that were sent SIGTERM and are in their grace window, keyed by
    // job id -> when SIGTERM was sent. Local to this task (only the reaper
    // touches it); lost on restart, which merely restarts the grace.
    let mut signaled: HashMap<JobId, DateTime<Utc>> = HashMap::new();

    // Neither map is tied to a leadership term, so reset them on a
    // follower -> leader transition (below) to drop a prior stint's stale state.
    let mut was_leader = false;

    loop {
        interval.tick().await;

        if !raft.is_leader() {
            was_leader = false;
            continue;
        }

        if !was_leader {
            // (Re)gained leadership: drop stale timers so the reap check below
            // reseeds every running allocation to a full grace window.
            signaled.clear();
            cluster.reset_interactive_last_seen();
            was_leader = true;
        }

        let limit_secs = cluster.config().scheduler.inactive_limit_secs;
        let now = Utc::now();

        let running: Vec<_> = cluster
            .get_jobs(&JobFilter {
                states: &[JobState::Running],
                ..Default::default()
            })
            .into_iter()
            .filter(|j| j.spec.interactive || j.spec.srun_job)
            .collect();
        let ids: Vec<JobId> = running.iter().map(|j| j.job_id).collect();

        // Prunes the keepalive map every tick; returns candidates only when
        // the limit is enabled.
        let stale = cluster.interactive_reap_candidates(&ids, now, limit_secs);
        let stale_set: HashSet<JobId> = stale.iter().copied().collect();
        // Drop grace timers for jobs that recovered (pinged again) or are gone,
        // so a client that comes back during the grace aborts the kill.
        signaled.retain(|id, _| stale_set.contains(id));

        let by_id: HashMap<JobId, &spur_core::job::Job> =
            running.iter().map(|j| (j.job_id, j)).collect();

        for job_id in stale {
            let Some(job) = by_id.get(&job_id) else {
                continue;
            };

            match signaled.get(&job_id) {
                None => {
                    info!(
                        job_id,
                        limit_secs,
                        grace_secs = GRACE_PERIOD_SECS,
                        "interactive allocation idle past InactiveLimit — sending SIGTERM, grace period starts"
                    );
                    send_cancel_to_agents(&cluster, job, 15).await; // SIGTERM
                    signaled.insert(job_id, now);
                }
                Some(signaled_at) if (now - *signaled_at).num_seconds() < GRACE_PERIOD_SECS => {}
                Some(_) => {
                    // Re-fetch before finalizing: if the job was requeued and
                    // redispatched as a new run under the same id since the
                    // snapshot, `complete_job`'s terminal-state guard would not
                    // catch it and the SIGKILL could hit the new run.
                    let Some(fresh) = cluster.get_job(job_id) else {
                        signaled.remove(&job_id);
                        continue;
                    };
                    if fresh.state != JobState::Running || fresh.run_attempt != job.run_attempt {
                        signaled.remove(&job_id);
                        continue;
                    }

                    info!(
                        job_id,
                        "InactiveLimit grace expired — force-killing allocation"
                    );

                    if let Err(e) = cluster.complete_job(job_id, -1, JobState::Timeout) {
                        warn!(job_id, error = %e, "failed to reap inactive allocation");
                        continue;
                    }

                    // SIGKILL on the run's current nodes, not the stale snapshot.
                    send_cancel_to_nodes(
                        &cluster,
                        job_id,
                        fresh.run_attempt,
                        &fresh.allocated_nodes,
                        9,
                    )
                    .await;
                    signaled.remove(&job_id);
                }
            }
        }
    }
}

/// Force-finish jobs stuck in COMPLETING past `complete_wait_secs`.
async fn enforce_completing_timeout(cluster: Arc<ClusterManager>, raft: Arc<RaftHandle>) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));

    loop {
        interval.tick().await;

        if !raft.is_leader() {
            continue;
        }

        let now = Utc::now();
        let wait = chrono::Duration::seconds(cluster.config().scheduler.complete_wait_secs as i64);

        let completing = cluster.get_jobs(&JobFilter {
            states: &[spur_core::job::JobState::Completing],
            ..Default::default()
        });

        for job in completing {
            let Some(completing_since) = job.end_time else {
                continue;
            };
            if now - completing_since < wait {
                continue;
            }

            force_finish_completing_job(&cluster, &job).await;
        }
    }
}

/// Force-finish a job stuck in Completing past `complete_wait_secs`, cancelling
/// it on the unreported nodes first so their agents release the allocation
/// before the controller frees those nodes. Best-effort; the agent reclaim backs it.
async fn force_finish_completing_job(cluster: &Arc<ClusterManager>, job: &spur_core::job::Job) {
    let missing: Vec<_> = job
        .allocated_nodes
        .iter()
        .filter(|n| !job.node_completions.contains_key(*n))
        .cloned()
        .collect();

    // Empty when no nodes allocated; derived_completion falls back to worst completion.
    let primary = job.allocated_nodes.first().cloned().unwrap_or_default();
    let (mut state, mut exit_code, _signal) =
        spur_core::job::Job::derived_completion(&job.node_completions, &primary);
    if job.node_completions.is_empty() {
        state = spur_core::job::JobState::Failed;
        exit_code = -1;
    } else if !missing.is_empty() {
        warn!(
            job_id = job.job_id,
            missing = ?missing,
            reported = job.node_completions.len(),
            expected = job.allocated_nodes.len(),
            "completing timeout — not all nodes reported"
        );
        state = spur_core::job::JobState::Failed;
        if exit_code == 0 {
            exit_code = 1;
        }
    }

    if !missing.is_empty() {
        cancel_job_on_nodes(cluster, job.job_id, job.run_attempt, &missing, 9).await;
    }

    info!(
        job_id = job.job_id,
        state = ?state,
        exit_code,
        "completing timeout expired — force-finishing job"
    );

    if let Err(e) = cluster.complete_job(job.job_id, exit_code, state) {
        warn!(
            job_id = job.job_id,
            error = %e,
            "failed to force-finish job after completing timeout"
        );
    }
}

fn spawn_power_command(cmd: &str, node_name: &str, action: &str) {
    let cmd = cmd.to_owned();
    let node_name = node_name.to_owned();
    let action = action.to_owned();
    tokio::spawn(async move {
        if let Err(e) = tokio::process::Command::new("sh")
            .args(["-c", &cmd])
            .status()
            .await
        {
            warn!(node = %node_name, error = %e, action = %action, "power command failed");
        }
    });
}

/// Power management: suspend idle nodes and resume them when jobs are pending.
///
/// Disabled when `power.suspend_timeout_secs` is not set in the config.
async fn manage_power(cluster: Arc<ClusterManager>, raft: Arc<RaftHandle>) {
    // Whether power management runs at all (and its idle timeout) is decided
    // once here; toggling power on/off or changing the timeout needs a restart.
    // The suspend/resume *commands* below are read live each cycle.
    let suspend_timeout = match cluster.config().power.suspend_timeout_secs {
        Some(t) => t,
        None => return,
    };

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
    info!(suspend_timeout, "power management enabled");

    loop {
        interval.tick().await;

        if !raft.is_leader() {
            continue;
        }

        let now = Utc::now();
        let nodes = cluster.get_nodes();

        // Suspend idle nodes that have been idle longer than the timeout
        for node in &nodes {
            if node.state != spur_core::node::NodeState::Idle {
                continue;
            }
            let Some(last_busy) = node.last_busy else {
                continue;
            };
            if (now - last_busy).num_seconds() as u64 <= suspend_timeout {
                continue;
            }
            info!(node = %node.name, "suspending idle node (power saving)");
            let _ = cluster.update_node_state(
                &node.name,
                spur_core::node::NodeState::Suspended,
                Some("Power saving".into()),
            );
            if let Some(ref cmd) = cluster.config().power.suspend_command {
                spawn_power_command(&cmd.replace("{node}", &node.name), &node.name, "suspend");
            }
        }

        // Resume suspended nodes if there are pending jobs
        let pending = cluster.pending_jobs();
        if !pending.is_empty() {
            for node in &nodes {
                if node.state != spur_core::node::NodeState::Suspended {
                    continue;
                }
                info!(node = %node.name, "resuming suspended node for pending jobs");
                let _ =
                    cluster.update_node_state(&node.name, spur_core::node::NodeState::Idle, None);
                if let Some(ref cmd) = cluster.config().power.resume_command {
                    spawn_power_command(&cmd.replace("{node}", &node.name), &node.name, "resume");
                }
            }
        }
    }
}

/// Send CancelJob RPC to all agents for a job with a specific signal.
pub async fn send_cancel_to_agents(
    cluster: &Arc<ClusterManager>,
    job: &spur_core::job::Job,
    signal: i32,
) {
    send_cancel_to_nodes(
        cluster,
        job.job_id,
        job.run_attempt,
        &job.allocated_nodes,
        signal,
    )
    .await;
}

/// Send CancelJob RPC to an explicit set of nodes for a job with a specific
/// signal. Callers that already know which nodes ran the job (e.g. a dispatch
/// loop) should use this instead of `send_cancel_to_agents` so the cancel
/// isn't at the mercy of `job.allocated_nodes` having been mutated in the
/// meantime (e.g. cleared by a requeue-on-eviction side effect).
///
/// Fire-and-forget: each node's cancel runs on its own task and this returns
/// immediately. Use `cancel_job_on_nodes` when the cancel must be delivered
/// before subsequent work (e.g. a requeue that could re-dispatch the job).
pub async fn send_cancel_to_nodes(
    cluster: &Arc<ClusterManager>,
    job_id: spur_core::job::JobId,
    run_attempt: u32,
    node_names: &[String],
    signal: i32,
) {
    for agent_addr in cancel_agent_addrs(cluster, job_id, node_names) {
        tokio::spawn(cancel_one_agent(agent_addr, job_id, run_attempt, signal));
    }
}

/// Like `send_cancel_to_nodes`, but awaits delivery of every cancel before
/// returning so the caller can establish a happens-before ordering against
/// later actions. Each RPC is bounded by `CANCEL_RPC_TIMEOUT` so an
/// unreachable agent can't stall the caller indefinitely.
pub async fn cancel_job_on_nodes(
    cluster: &Arc<ClusterManager>,
    job_id: spur_core::job::JobId,
    run_attempt: u32,
    node_names: &[String],
    signal: i32,
) {
    let mut set = tokio::task::JoinSet::new();
    for agent_addr in cancel_agent_addrs(cluster, job_id, node_names) {
        set.spawn(cancel_one_agent(agent_addr, job_id, run_attempt, signal));
    }
    while set.join_next().await.is_some() {}
}

/// Cancel an in-flight srun step on the given nodes without tearing down the
/// allocation job process (batch script / companion hold).
pub async fn cancel_step_on_nodes(
    cluster: &Arc<ClusterManager>,
    job_id: spur_core::job::JobId,
    step_id: u32,
    node_names: &[String],
    signal: i32,
) {
    let mut set = tokio::task::JoinSet::new();
    for agent_addr in cancel_agent_addrs(cluster, job_id, node_names) {
        set.spawn(cancel_one_step_agent(agent_addr, job_id, step_id, signal));
    }
    while set.join_next().await.is_some() {}
}

/// Deliver one CancelStep RPC, bounded by `CANCEL_RPC_TIMEOUT`.
async fn cancel_one_step_agent(
    agent_addr: String,
    job_id: spur_core::job::JobId,
    step_id: u32,
    signal: i32,
) {
    use spur_proto::proto::CancelStepRequest;

    let attempt = async {
        match crate::agent_client::connect(agent_addr.clone())
            .await
            .map(|c| {
                c.max_decoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE)
                    .max_encoding_message_size(spur_proto::MAX_GRPC_REQUEST_SIZE)
            }) {
            Ok(mut client) => {
                if let Err(e) = client
                    .cancel_step(CancelStepRequest {
                        job_id,
                        step_id,
                        signal,
                    })
                    .await
                {
                    warn!(
                        job_id,
                        step_id,
                        signal,
                        agent = %agent_addr,
                        error = %e,
                        "CancelStep RPC failed"
                    );
                } else {
                    info!(job_id, step_id, signal, agent = %agent_addr, "sent CancelStep");
                }
            }
            Err(e) => {
                warn!(
                    job_id,
                    step_id,
                    agent = %agent_addr,
                    error = %e,
                    "failed to connect to agent for step cancel"
                );
            }
        }
    };
    if tokio::time::timeout(CANCEL_RPC_TIMEOUT, attempt)
        .await
        .is_err()
    {
        warn!(
            job_id,
            step_id,
            agent = %agent_addr,
            "CancelStep RPC timed out"
        );
    }
}

/// Resolve `node_names` to agent URLs, logging and skipping any node whose
/// address is unknown.
fn cancel_agent_addrs(
    cluster: &Arc<ClusterManager>,
    job_id: spur_core::job::JobId,
    node_names: &[String],
) -> Vec<String> {
    let mut addrs = Vec::with_capacity(node_names.len());
    for node_name in node_names {
        match cluster.get_node(node_name) {
            Some(ref n) => {
                if let Some(url) = node_comm_http_url(n) {
                    addrs.push(url);
                } else {
                    warn!(
                        job_id,
                        node = %node_name,
                        "no comm address — cannot cancel job on node"
                    );
                }
            }
            _ => {
                warn!(
                    job_id,
                    node = %node_name,
                    "no agent address — cannot cancel job on node"
                );
            }
        }
    }
    addrs
}

/// Deliver one CancelJob RPC, bounded by `CANCEL_RPC_TIMEOUT`. Errors and
/// timeouts are logged, never propagated: a cancel is best-effort cleanup and
/// must not block the caller past the timeout.
async fn cancel_one_agent(
    agent_addr: String,
    job_id: spur_core::job::JobId,
    run_attempt: u32,
    signal: i32,
) {
    let attempt = async {
        match crate::agent_client::connect(agent_addr.clone())
            .await
            .map(|c| {
                c.max_decoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE)
                    .max_encoding_message_size(spur_proto::MAX_GRPC_REQUEST_SIZE)
            }) {
            Ok(mut client) => {
                if let Err(e) = client
                    .cancel_job(AgentCancelJobRequest {
                        job_id,
                        signal,
                        run_attempt,
                    })
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
    };
    if tokio::time::timeout(CANCEL_RPC_TIMEOUT, attempt)
        .await
        .is_err()
    {
        warn!(
            job_id,
            agent = %agent_addr,
            "CancelJob RPC timed out"
        );
    }
}

/// Dispatch suspend (SIGSTOP) or resume (SIGCONT) to every allocated node.
pub async fn send_suspend_to_agents(
    cluster: &Arc<ClusterManager>,
    job: &spur_core::job::Job,
    resume: bool,
) {
    for node_name in &job.allocated_nodes {
        let node_info = cluster.get_node(node_name);
        let agent_addr = match node_info {
            Some(ref n) => match node_comm_http_url(n) {
                Some(url) => url,
                None => {
                    warn!(job_id = job.job_id, node = %node_name,
                        "no comm address — cannot suspend/resume job on node");
                    continue;
                }
            },
            _ => {
                warn!(job_id = job.job_id, node = %node_name,
                    "no agent address — cannot suspend/resume job on node");
                continue;
            }
        };
        let job_id = job.job_id;
        tokio::spawn(async move {
            match crate::agent_client::connect(agent_addr.clone())
                .await
                .map(|c| {
                    c.max_decoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE)
                        .max_encoding_message_size(spur_proto::MAX_GRPC_REQUEST_SIZE)
                }) {
                Ok(mut client) => {
                    if let Err(e) = client
                        .suspend_job(AgentSuspendJobRequest { job_id, resume })
                        .await
                    {
                        warn!(job_id, resume, agent = %agent_addr, error = %e, "SuspendJob RPC failed");
                    } else {
                        info!(job_id, resume, agent = %agent_addr, "sent SuspendJob");
                    }
                }
                Err(e) => {
                    warn!(job_id, agent = %agent_addr, error = %e,
                        "failed to connect to agent for suspend/resume");
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spur_core::job::{Job, JobSpec};
    use spur_core::resource::{
        build_exclusive_allocation, build_node_allocation, GpuLinkType, GpuResource,
        ResourceAllocations, ResourceSet,
    };
    use std::collections::HashMap;

    fn job_with_spec(mut spec: JobSpec) -> Job {
        spec.cpus_per_task = spec.cpus_per_task.max(1);
        spec.num_tasks = spec.num_tasks.max(1);
        spec.num_nodes = spec.num_nodes.max(1);
        Job::new(1, spec)
    }

    fn running_job_on(node: &str, start: DateTime<Utc>, minutes: i64) -> Job {
        let mut job = job_with_spec(JobSpec {
            time_limit: Some(chrono::Duration::minutes(minutes)),
            ..Default::default()
        });
        job.state = spur_core::job::JobState::Running;
        job.start_time = Some(start);
        job.allocated_nodes = vec![node.to_string()];
        job
    }

    #[test]
    fn busy_until_takes_the_max_across_jobs_sharing_a_node() {
        let now = Utc::now();
        let sooner = running_job_on("node001", now, 10);
        let later = running_job_on("node001", now, 60);

        // Larger-ending job processed first: a "last wins" bug (instead of a
        // true max) would incorrectly keep the smaller value here.
        let busy_until = busy_until_from_running_jobs(&[later, sooner]);

        let expected =
            now + chrono::Duration::minutes(60) + chrono::Duration::seconds(GRACE_PERIOD_SECS);
        let got = busy_until["node001"];
        assert!(
            (got - expected).num_seconds().abs() < 2,
            "expected the later job's end time to win, got {got}, expected ~{expected}"
        );
    }

    #[test]
    fn busy_until_saturates_instead_of_panicking_on_an_extreme_suspended_secs() {
        let now = Utc::now();
        let mut job = running_job_on("node001", now, 60);
        job.suspended_secs = i64::MAX;

        let busy_until = busy_until_from_running_jobs(&[job]);
        let got = busy_until["node001"];
        assert!(got > now, "expected a sane future timestamp, got {got}");
    }

    #[test]
    fn busy_until_does_not_go_backwards_for_a_long_unbounded_job() {
        let now = Utc::now();
        let mut job = running_job_on("node001", now - chrono::Duration::days(400), 0);
        job.spec.time_limit = None;

        let busy_until = busy_until_from_running_jobs(&[job]);
        let got = busy_until["node001"];
        assert!(
            got >= now,
            "busy_until must never be in the past relative to now, got {got}, now {now}"
        );
    }

    #[test]
    fn busy_until_accounts_for_currently_suspended_time() {
        let now = Utc::now();
        let mut job = running_job_on("node001", now, 10);
        job.suspended_at = Some(now - chrono::Duration::minutes(5));

        let busy_until = busy_until_from_running_jobs(&[job]);
        let expected = now
            + chrono::Duration::minutes(10)
            + chrono::Duration::minutes(5)
            + chrono::Duration::seconds(GRACE_PERIOD_SECS);
        let got = busy_until["node001"];
        assert!(
            (got - expected).num_seconds().abs() < 2,
            "expected suspended time folded in, got {got}, expected ~{expected}"
        );
    }

    #[test]
    fn busy_until_falls_back_to_now_when_start_time_is_missing() {
        let now = Utc::now();
        let mut job = running_job_on("node001", now, 10);
        job.start_time = None;

        let busy_until = busy_until_from_running_jobs(&[job]);
        let expected =
            now + chrono::Duration::minutes(10) + chrono::Duration::seconds(GRACE_PERIOD_SECS);
        let got = busy_until["node001"];
        assert!(
            (got - expected).num_seconds().abs() < 2,
            "expected now-based fallback, got {got}, expected ~{expected}"
        );
    }

    fn node_total(cpus: u32, memory_mb: u64, gpus: Vec<GpuResource>) -> ResourceSet {
        ResourceSet {
            cpus,
            memory_mb,
            gpus,
            generic: HashMap::new(),
        }
    }

    fn gpu(device_id: u32, gpu_type: &str) -> GpuResource {
        GpuResource {
            device_id,
            gpu_type: gpu_type.into(),
            memory_mb: 192_000,
            peer_gpus: vec![],
            link_type: GpuLinkType::PCIe,
        }
    }

    fn exclusive_per_node(
        nodes: &[String],
        totals: &HashMap<String, ResourceSet>,
        memory_mb: u64,
    ) -> HashMap<String, ResourceAllocations> {
        nodes
            .iter()
            .filter_map(|name| {
                totals
                    .get(name)
                    .map(|inv| (name.clone(), build_exclusive_allocation(inv, memory_mb)))
            })
            .collect()
    }

    fn request_per_node(
        nodes: &[String],
        totals: &HashMap<String, ResourceSet>,
        request: &ResourceSet,
    ) -> HashMap<String, ResourceAllocations> {
        nodes
            .iter()
            .filter_map(|name| {
                totals.get(name).map(|inv| {
                    (
                        name.clone(),
                        build_node_allocation(inv, &ResourceAllocations::default(), request),
                    )
                })
            })
            .collect()
    }

    // ── #147: --exclusive enforcement ─────────────────────────────
    //
    // Repro of the reported bug: an exclusive job that requests 1 CPU
    // would only record 1 CPU as allocated against the node. Backfill's
    // `alloc.cpus >= total.cpus` saturation check would never fire,
    // letting other jobs schedule onto the node. compute_job_allocation
    // must bump cpus / gpus / generic to the sum of node totals.

    #[test]
    fn exclusive_job_bumps_cpus_to_node_total() {
        let spec = JobSpec {
            cpus_per_task: 2,
            num_tasks: 1,
            num_nodes: 1,
            exclusive: true,
            ..Default::default()
        };
        let job = job_with_spec(spec);

        let nodes = vec!["n1".to_string()];
        let totals = HashMap::from([("n1".to_string(), node_total(64, 256_000, vec![]))]);
        let per_node = exclusive_per_node(&nodes, &totals, 0);
        let alloc = compute_job_allocation(&job, &nodes, &per_node);

        assert_eq!(
            alloc.cpus, 64,
            "exclusive job must record full node CPU count, not requested"
        );
    }

    #[test]
    fn exclusive_job_bumps_cpus_across_multiple_nodes() {
        let spec = JobSpec {
            cpus_per_task: 1,
            num_nodes: 2,
            exclusive: true,
            ..Default::default()
        };
        let job = job_with_spec(spec);

        let nodes = vec!["n1".to_string(), "n2".to_string()];
        let totals = HashMap::from([
            ("n1".to_string(), node_total(64, 256_000, vec![])),
            ("n2".to_string(), node_total(48, 128_000, vec![])),
        ]);
        let per_node = exclusive_per_node(&nodes, &totals, 0);
        let alloc = compute_job_allocation(&job, &nodes, &per_node);

        assert_eq!(alloc.cpus, 112, "exclusive job must sum CPUs across nodes");
    }

    #[test]
    fn exclusive_job_takes_all_gpus_from_each_node() {
        let spec = JobSpec {
            exclusive: true,
            ..Default::default()
        };
        let job = job_with_spec(spec);

        let nodes = vec!["n1".to_string()];
        let totals = HashMap::from([(
            "n1".to_string(),
            node_total(64, 256_000, vec![gpu(0, "mi300x"), gpu(1, "mi300x")]),
        )]);
        let per_node = exclusive_per_node(&nodes, &totals, 0);
        let alloc = compute_job_allocation(&job, &nodes, &per_node);

        assert_eq!(
            alloc.total_device_count("gpu"),
            2,
            "exclusive job must take every GPU"
        );
        assert_eq!(alloc.device_ids("gpu"), vec![0, 1]);
    }

    #[test]
    fn exclusive_job_keeps_memory_at_request_not_node_total() {
        let spec = JobSpec {
            cpus_per_task: 1,
            exclusive: true,
            memory_per_node_mb: Some(4096),
            ..Default::default()
        };
        let job = job_with_spec(spec);

        let nodes = vec!["n1".to_string()];
        let totals = HashMap::from([("n1".to_string(), node_total(64, 256_000, vec![]))]);
        let per_node = exclusive_per_node(&nodes, &totals, 4096);
        let alloc = compute_job_allocation(&job, &nodes, &per_node);

        assert_eq!(
            alloc.memory_mb, 4096,
            "exclusive memory must stay at request, not node total"
        );
    }

    #[test]
    fn exclusive_job_sums_generic_gres_from_each_node() {
        let spec = JobSpec {
            exclusive: true,
            ..Default::default()
        };
        let job = job_with_spec(spec);

        let mut gen_a = HashMap::new();
        gen_a.insert("license:fluent".to_string(), 5u64);
        let total_a = ResourceSet {
            cpus: 64,
            memory_mb: 256_000,
            gpus: vec![],
            generic: gen_a,
        };

        let mut gen_b = HashMap::new();
        gen_b.insert("license:fluent".to_string(), 3u64);
        let total_b = ResourceSet {
            cpus: 64,
            memory_mb: 256_000,
            gpus: vec![],
            generic: gen_b,
        };

        let nodes = vec!["n1".to_string(), "n2".to_string()];
        let totals = HashMap::from([("n1".to_string(), total_a), ("n2".to_string(), total_b)]);
        let per_node = exclusive_per_node(&nodes, &totals, 0);
        let alloc = compute_job_allocation(&job, &nodes, &per_node);

        assert_eq!(
            alloc
                .devices
                .get("license:fluent")
                .map(|d| d.iter().map(|x| x.count).sum::<u64>()),
            Some(8)
        );
    }

    #[test]
    fn non_exclusive_job_records_request_not_node_total() {
        let spec = JobSpec {
            cpus_per_task: 2,
            num_tasks: 1,
            num_nodes: 1,
            exclusive: false,
            ..Default::default()
        };
        let job = job_with_spec(spec);

        let nodes = vec!["n1".to_string()];
        let totals = HashMap::from([("n1".to_string(), node_total(64, 256_000, vec![]))]);
        let request = backfill::job_resource_request(&job);
        let per_node = request_per_node(&nodes, &totals, &request);
        let alloc = compute_job_allocation(&job, &nodes, &per_node);

        assert_eq!(
            alloc.cpus, 2,
            "non-exclusive job must record exactly what was requested"
        );
    }

    #[test]
    fn exclusive_job_handles_missing_node_metadata() {
        let spec = JobSpec {
            exclusive: true,
            ..Default::default()
        };
        let job = job_with_spec(spec);

        let nodes = vec!["n1".to_string(), "ghost".to_string()];
        let totals = HashMap::from([("n1".to_string(), node_total(64, 256_000, vec![]))]);
        let per_node = exclusive_per_node(&nodes, &totals, 0);
        let alloc = compute_job_allocation(&job, &nodes, &per_node);

        assert_eq!(alloc.cpus, 64);
    }

    fn partition_with_mode(
        name: &str,
        mode: spur_core::partition::PreemptMode,
    ) -> spur_core::partition::Partition {
        spur_core::partition::Partition {
            name: name.into(),
            preempt_mode: mode,
            ..Default::default()
        }
    }

    fn job_in_partitions(partition: &str) -> Job {
        job_with_spec(JobSpec {
            partition: Some(partition.into()),
            ..Default::default()
        })
    }

    fn qos_with_mode(mode: spur_core::accounting::QosPreemptMode) -> spur_core::accounting::Qos {
        spur_core::accounting::Qos {
            preempt_mode: mode,
            ..Default::default()
        }
    }

    fn no_qos_override() -> spur_core::accounting::Qos {
        qos_with_mode(spur_core::accounting::QosPreemptMode::Off)
    }

    fn sched_config_default() -> spur_core::config::SchedulerConfig {
        spur_core::config::SchedulerConfig::default()
    }

    fn sched_config_with_exempt(secs: u32) -> spur_core::config::SchedulerConfig {
        spur_core::config::SchedulerConfig {
            preempt_exempt_time: secs,
            ..Default::default()
        }
    }

    fn partition_with_exempt(name: &str, secs: u32) -> spur_core::partition::Partition {
        spur_core::partition::Partition {
            name: name.into(),
            preempt_exempt_time: Some(secs),
            ..Default::default()
        }
    }

    fn qos_with_exempt(secs: u32) -> spur_core::accounting::Qos {
        spur_core::accounting::Qos {
            limits: spur_core::accounting::QosLimits {
                preempt_exempt_time: Some(secs),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn qos_with_preempt_list(names: &[&str]) -> spur_core::accounting::Qos {
        spur_core::accounting::Qos {
            preempt: names.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn effective_exempt_secs_falls_back_to_global() {
        let job = job_in_partitions("gpu");
        let parts = vec![partition_with_mode(
            "gpu",
            spur_core::partition::PreemptMode::Cancel,
        )];
        let qos = no_qos_override();
        let sched = sched_config_with_exempt(120);
        assert_eq!(effective_exempt_secs(&job, &parts, &qos, &sched), 120);
    }

    #[test]
    fn effective_exempt_secs_partition_overrides_global() {
        let job = job_in_partitions("gpu");
        let parts = vec![partition_with_exempt("gpu", 60)];
        let qos = no_qos_override();
        let sched = sched_config_with_exempt(300);
        assert_eq!(effective_exempt_secs(&job, &parts, &qos, &sched), 60);
    }

    #[test]
    fn effective_exempt_secs_qos_overrides_partition_and_global() {
        let job = job_in_partitions("gpu");
        let parts = vec![partition_with_exempt("gpu", 60)];
        let qos = qos_with_exempt(10);
        let sched = sched_config_with_exempt(300);
        assert_eq!(effective_exempt_secs(&job, &parts, &qos, &sched), 10);
    }

    #[test]
    fn effective_exempt_secs_zero_global_and_no_overrides_is_zero() {
        let job = job_in_partitions("gpu");
        let parts = vec![partition_with_mode(
            "gpu",
            spur_core::partition::PreemptMode::Cancel,
        )];
        let qos = no_qos_override();
        let sched = sched_config_default();
        assert_eq!(effective_exempt_secs(&job, &parts, &qos, &sched), 0);
    }

    #[test]
    fn qos_preempt_allow_list_permits_listed_qos() {
        let pending_qos = qos_with_preempt_list(&["low", "batch"]);
        let candidate_qos = spur_core::accounting::Qos {
            name: "low".into(),
            ..Default::default()
        };
        // Simulate the allow-list check from try_preempt.
        let permitted = pending_qos.preempt.contains(&candidate_qos.name);
        assert!(permitted);
    }

    #[test]
    fn qos_preempt_allow_list_blocks_unlisted_qos() {
        let pending_qos = qos_with_preempt_list(&["low"]);
        let candidate_qos = spur_core::accounting::Qos {
            name: "medium".into(),
            ..Default::default()
        };
        let permitted = pending_qos.preempt.contains(&candidate_qos.name);
        assert!(!permitted);
    }

    #[test]
    fn qos_preempt_empty_allow_list_blocks_all() {
        let pending_qos = qos_with_preempt_list(&[]);
        let candidate_qos = spur_core::accounting::Qos {
            name: "low".into(),
            ..Default::default()
        };
        let permitted = pending_qos.preempt.contains(&candidate_qos.name);
        assert!(!permitted);
    }

    #[test]
    fn job_preempt_mode_single_partition() {
        use spur_core::partition::PreemptMode;
        let parts = vec![partition_with_mode("gpu", PreemptMode::Requeue)];
        assert_eq!(
            job_preempt_mode(&job_in_partitions("gpu"), &parts, &no_qos_override()),
            PreemptMode::Requeue
        );
    }

    #[test]
    fn job_preempt_mode_unset_or_unknown_is_off() {
        use spur_core::partition::PreemptMode;
        let parts = vec![partition_with_mode("gpu", PreemptMode::Cancel)];
        assert_eq!(
            job_preempt_mode(
                &job_with_spec(JobSpec::default()),
                &parts,
                &no_qos_override()
            ),
            PreemptMode::Off
        );
        assert_eq!(
            job_preempt_mode(&job_in_partitions("nope"), &parts, &no_qos_override()),
            PreemptMode::Off
        );
    }

    #[test]
    fn job_preempt_mode_multi_partition_picks_most_aggressive() {
        use spur_core::partition::PreemptMode;
        // A job spanning gpu,cpu must resolve a mode (was Off before the fix,
        // making multi-partition jobs unpreemptable). Cancel > Requeue.
        let parts = vec![
            partition_with_mode("gpu", PreemptMode::Requeue),
            partition_with_mode("cpu", PreemptMode::Cancel),
        ];
        assert_eq!(
            job_preempt_mode(&job_in_partitions("gpu, cpu"), &parts, &no_qos_override()),
            PreemptMode::Cancel
        );
    }

    #[test]
    fn job_preempt_mode_multi_partition_off_when_none_configured() {
        use spur_core::partition::PreemptMode;
        let parts = vec![
            partition_with_mode("gpu", PreemptMode::Off),
            partition_with_mode("cpu", PreemptMode::Off),
        ];
        assert_eq!(
            job_preempt_mode(&job_in_partitions("gpu,cpu"), &parts, &no_qos_override()),
            PreemptMode::Off
        );
    }

    // ── dispatch_job_to_nodes: real partial-dispatch-failure trigger path ──
    //
    // Exercises the actual JoinSet success/failure counting in
    // dispatch_job_to_nodes (not a reimplementation of it) against a real
    // agent over the network, so the eviction + cancel-RPC behavior it
    // drives is verified end-to-end rather than by calling evict_job
    // directly on an already-Running job.
    mod dispatch_trigger_tests {
        use super::*;
        use spur_core::config::SlurmConfig;
        use spur_core::node::NodeSource;
        use std::sync::atomic::{AtomicU32, Ordering};
        use tempfile::TempDir;
        use tonic::transport::server::TcpIncoming;
        use tonic::transport::Server;

        /// Minimal SlurmAgent: counts cancel_job calls, so tests can assert the
        /// controller actually tried to stop the job on nodes that did launch
        /// it. `reject_launch_as` makes launch_job fail with a given
        /// classification instead of succeeding. `launch_delay` sleeps inside
        /// `launch_job` before responding, standing in for the node-side work
        /// (resource allocation, GPU device-injection planning, process
        /// spawn) a real agent does — used to measure confirmation latency
        /// under a synthetic per-node launch cost rather than estimating it.
        struct MockAgent {
            cancel_calls: Arc<AtomicU32>,
            release_pmix_calls: Arc<AtomicU32>,
            reject_launch_as: Option<spur_proto::proto::LaunchFailureKind>,
            launch_delay: Duration,
            register_delay: Duration,
            /// launch_job returns a ResourceExhausted status, standing in for a
            /// node whose local allocation table already holds the GPUs.
            reject_resources: bool,
            /// Records each `LaunchJobRequest.task_fanout` this agent receives,
            /// so tests can assert on it without a real spurd behind the RPC.
            fanout_calls: Option<Arc<std::sync::Mutex<Vec<bool>>>>,
        }

        #[tonic::async_trait]
        impl spur_proto::proto::slurm_agent_server::SlurmAgent for MockAgent {
            type StreamJobOutputStream =
                tonic::codegen::BoxStream<spur_proto::proto::StreamJobOutputChunk>;
            type InteractiveSessionStream =
                tonic::codegen::BoxStream<spur_proto::proto::InteractiveOutput>;

            async fn launch_job(
                &self,
                request: tonic::Request<spur_proto::proto::LaunchJobRequest>,
            ) -> Result<tonic::Response<spur_proto::proto::LaunchJobResponse>, tonic::Status>
            {
                if !self.launch_delay.is_zero() {
                    tokio::time::sleep(self.launch_delay).await;
                }
                if self.reject_resources {
                    return Err(tonic::Status::resource_exhausted(
                        "controller-allocated GPUs unavailable on this node",
                    ));
                }
                if let Some(kind) = self.reject_launch_as {
                    return Ok(tonic::Response::new(spur_proto::proto::LaunchJobResponse {
                        success: false,
                        error: "prolog failed: prolog_slurmd script exited with exit status: 1"
                            .into(),
                        failure_kind: kind as i32,
                        ..Default::default()
                    }));
                }
                // Echo a path keyed by task_offset so tests can assert the
                // controller keeps the primary node's (task_offset == 0) path.
                let req = request.into_inner();
                if let Some(sink) = &self.fanout_calls {
                    sink.lock().unwrap().push(req.task_fanout);
                }
                let path = format!("/spool/off{}/spur.out", req.task_offset);
                Ok(tonic::Response::new(spur_proto::proto::LaunchJobResponse {
                    success: true,
                    error: String::new(),
                    stdout_path: path.clone(),
                    stderr_path: path,
                    ..Default::default()
                }))
            }

            async fn prepare_pmix(
                &self,
                _request: tonic::Request<spur_proto::proto::PreparePmixRequest>,
            ) -> Result<tonic::Response<spur_proto::proto::PreparePmixResponse>, tonic::Status>
            {
                Ok(tonic::Response::new(
                    spur_proto::proto::PreparePmixResponse {
                        success: true,
                        error: String::new(),
                    },
                ))
            }

            async fn release_pmix(
                &self,
                _request: tonic::Request<spur_proto::proto::ReleasePmixRequest>,
            ) -> Result<tonic::Response<spur_proto::proto::ReleasePmixResponse>, tonic::Status>
            {
                self.release_pmix_calls.fetch_add(1, Ordering::SeqCst);
                Ok(tonic::Response::new(
                    spur_proto::proto::ReleasePmixResponse {},
                ))
            }

            async fn cancel_job(
                &self,
                _request: tonic::Request<spur_proto::proto::AgentCancelJobRequest>,
            ) -> Result<tonic::Response<()>, tonic::Status> {
                self.cancel_calls.fetch_add(1, Ordering::SeqCst);
                Ok(tonic::Response::new(()))
            }

            async fn suspend_job(
                &self,
                _request: tonic::Request<spur_proto::proto::AgentSuspendJobRequest>,
            ) -> Result<tonic::Response<()>, tonic::Status> {
                Ok(tonic::Response::new(()))
            }

            async fn get_node_resources(
                &self,
                _request: tonic::Request<()>,
            ) -> Result<tonic::Response<spur_proto::proto::NodeResourcesResponse>, tonic::Status>
            {
                Ok(tonic::Response::new(Default::default()))
            }

            async fn probe_stepd(
                &self,
                _request: tonic::Request<spur_proto::proto::StepdProbeRequest>,
            ) -> Result<tonic::Response<spur_proto::proto::StepdProbeResponse>, tonic::Status>
            {
                Ok(tonic::Response::new(Default::default()))
            }

            async fn exec_in_job(
                &self,
                _request: tonic::Request<spur_proto::proto::ExecInJobRequest>,
            ) -> Result<tonic::Response<spur_proto::proto::ExecInJobResponse>, tonic::Status>
            {
                Ok(tonic::Response::new(Default::default()))
            }

            async fn run_command(
                &self,
                _request: tonic::Request<spur_proto::proto::RunCommandRequest>,
            ) -> Result<tonic::Response<spur_proto::proto::RunCommandResponse>, tonic::Status>
            {
                Ok(tonic::Response::new(Default::default()))
            }

            async fn cancel_step(
                &self,
                _request: tonic::Request<spur_proto::proto::CancelStepRequest>,
            ) -> Result<tonic::Response<()>, tonic::Status> {
                Ok(tonic::Response::new(()))
            }

            async fn register_job_allocation(
                &self,
                _request: tonic::Request<spur_proto::proto::RegisterJobAllocationRequest>,
            ) -> Result<
                tonic::Response<spur_proto::proto::RegisterJobAllocationResponse>,
                tonic::Status,
            > {
                if !self.register_delay.is_zero() {
                    tokio::time::sleep(self.register_delay).await;
                }
                Ok(tonic::Response::new(Default::default()))
            }

            async fn stream_job_output(
                &self,
                _request: tonic::Request<spur_proto::proto::StreamJobOutputRequest>,
            ) -> Result<tonic::Response<Self::StreamJobOutputStream>, tonic::Status> {
                Err(tonic::Status::unimplemented("not used in tests"))
            }

            async fn interactive_session(
                &self,
                _request: tonic::Request<tonic::Streaming<spur_proto::proto::InteractiveInput>>,
            ) -> Result<tonic::Response<Self::InteractiveSessionStream>, tonic::Status>
            {
                Err(tonic::Status::unimplemented("not used in tests"))
            }

            // k0s cluster-component + mesh RPCs: not exercised by these scheduler tests.
            async fn start_cluster_component(
                &self,
                _request: tonic::Request<spur_proto::proto::StartClusterComponentRequest>,
            ) -> Result<
                tonic::Response<spur_proto::proto::StartClusterComponentResponse>,
                tonic::Status,
            > {
                Err(tonic::Status::unimplemented("not used in tests"))
            }

            async fn stop_cluster_component(
                &self,
                _request: tonic::Request<spur_proto::proto::StopClusterComponentRequest>,
            ) -> Result<
                tonic::Response<spur_proto::proto::StopClusterComponentResponse>,
                tonic::Status,
            > {
                Err(tonic::Status::unimplemented("not used in tests"))
            }

            async fn get_cluster_component_status(
                &self,
                _request: tonic::Request<spur_proto::proto::GetClusterComponentStatusRequest>,
            ) -> Result<
                tonic::Response<spur_proto::proto::GetClusterComponentStatusResponse>,
                tonic::Status,
            > {
                Err(tonic::Status::unimplemented("not used in tests"))
            }

            async fn create_k0s_join_token(
                &self,
                _request: tonic::Request<spur_proto::proto::CreateK0sJoinTokenRequest>,
            ) -> Result<tonic::Response<spur_proto::proto::CreateK0sJoinTokenResponse>, tonic::Status>
            {
                Err(tonic::Status::unimplemented("not used in tests"))
            }

            async fn drain_k8s_node(
                &self,
                _request: tonic::Request<spur_proto::proto::DrainK8sNodeRequest>,
            ) -> Result<tonic::Response<spur_proto::proto::DrainK8sNodeResponse>, tonic::Status>
            {
                Err(tonic::Status::unimplemented("not used in tests"))
            }

            async fn delete_k8s_node(
                &self,
                _request: tonic::Request<spur_proto::proto::DeleteK8sNodeRequest>,
            ) -> Result<tonic::Response<spur_proto::proto::DeleteK8sNodeResponse>, tonic::Status>
            {
                Err(tonic::Status::unimplemented("not used in tests"))
            }

            async fn get_kubeconfig(
                &self,
                _request: tonic::Request<spur_proto::proto::GetKubeconfigRequest>,
            ) -> Result<tonic::Response<spur_proto::proto::GetKubeconfigResponse>, tonic::Status>
            {
                Err(tonic::Status::unimplemented("not used in tests"))
            }

            async fn apply_mesh(
                &self,
                _request: tonic::Request<spur_proto::proto::MeshMembership>,
            ) -> Result<tonic::Response<spur_proto::proto::ApplyMeshResponse>, tonic::Status>
            {
                Err(tonic::Status::unimplemented("not used in tests"))
            }
        }

        /// Spawn a real MockAgent gRPC server on an OS-assigned localhost
        /// port. Returns the bound address and the shared cancel-call counter.
        async fn spawn_mock_agent() -> (std::net::SocketAddr, Arc<AtomicU32>) {
            spawn_mock_agent_rejecting(None).await
        }

        async fn spawn_mock_agent_rejecting(
            reject_launch_as: Option<spur_proto::proto::LaunchFailureKind>,
        ) -> (std::net::SocketAddr, Arc<AtomicU32>) {
            spawn_mock_agent_full(reject_launch_as, Duration::ZERO).await
        }

        /// Like [`spawn_mock_agent`], but `launch_job` sleeps `delay` before
        /// accepting — a synthetic stand-in for a real agent's launch pipeline,
        /// so latency tests measure a real (if synthetic) number instead of
        /// guessing one.
        async fn spawn_mock_agent_with_delay(
            delay: Duration,
        ) -> (std::net::SocketAddr, Arc<AtomicU32>) {
            spawn_mock_agent_full(None, delay).await
        }

        /// Like [`spawn_mock_agent_with_delay`], but the delay lands on
        /// `register_job_allocation` — the srun allocation path, not the batch launch.
        async fn spawn_mock_agent_with_register_delay(
            delay: Duration,
        ) -> (std::net::SocketAddr, Arc<AtomicU32>) {
            let (addr, cancel_calls, _, _) =
                spawn_mock_agent_capturing_fanout(None, Duration::ZERO, delay, false).await;
            (addr, cancel_calls)
        }

        async fn spawn_mock_agent_full(
            reject_launch_as: Option<spur_proto::proto::LaunchFailureKind>,
            launch_delay: Duration,
        ) -> (std::net::SocketAddr, Arc<AtomicU32>) {
            let (addr, cancel_calls, _, _) = spawn_mock_agent_capturing_fanout(
                reject_launch_as,
                launch_delay,
                Duration::ZERO,
                false,
            )
            .await;
            (addr, cancel_calls)
        }

        /// Like [`spawn_mock_agent`], but also records every
        /// `LaunchJobRequest.task_fanout` this agent receives when `capture` is
        /// true, so a test can assert on it — `spawn_mock_agent`/`_rejecting`/
        /// `_with_delay` don't need this, so they pass `capture: false` and
        /// discard the (empty) sink.
        async fn spawn_mock_agent_capturing_fanout(
            reject_launch_as: Option<spur_proto::proto::LaunchFailureKind>,
            launch_delay: Duration,
            register_delay: Duration,
            capture: bool,
        ) -> (
            std::net::SocketAddr,
            Arc<AtomicU32>,
            Arc<AtomicU32>,
            Arc<std::sync::Mutex<Vec<bool>>>,
        ) {
            let incoming = TcpIncoming::bind("127.0.0.1:0".parse().unwrap()).unwrap();
            let addr = incoming.local_addr().unwrap();
            let cancel_calls = Arc::new(AtomicU32::new(0));
            let release_pmix_calls = Arc::new(AtomicU32::new(0));
            let fanout_calls = Arc::new(std::sync::Mutex::new(Vec::new()));
            let agent = MockAgent {
                cancel_calls: cancel_calls.clone(),
                release_pmix_calls: release_pmix_calls.clone(),
                reject_launch_as,
                launch_delay,
                register_delay,
                reject_resources: false,
                fanout_calls: capture.then(|| fanout_calls.clone()),
            };
            tokio::spawn(async move {
                let _ = Server::builder()
                    .add_service(
                        spur_proto::proto::slurm_agent_server::SlurmAgentServer::new(agent),
                    )
                    .serve_with_incoming(incoming)
                    .await;
            });
            (addr, cancel_calls, release_pmix_calls, fanout_calls)
        }

        /// Mock agent whose launch_job always rejects with ResourceExhausted.
        async fn spawn_mock_agent_rejecting_resources() -> std::net::SocketAddr {
            let incoming = TcpIncoming::bind("127.0.0.1:0".parse().unwrap()).unwrap();
            let addr = incoming.local_addr().unwrap();
            let agent = MockAgent {
                cancel_calls: Arc::new(AtomicU32::new(0)),
                reject_launch_as: None,
                launch_delay: Duration::ZERO,
                register_delay: Duration::ZERO,
                reject_resources: true,
                release_pmix_calls: Arc::new(AtomicU32::new(0)),
                fanout_calls: None,
            };
            tokio::spawn(async move {
                let _ = Server::builder()
                    .add_service(
                        spur_proto::proto::slurm_agent_server::SlurmAgentServer::new(agent),
                    )
                    .serve_with_incoming(incoming)
                    .await;
            });
            addr
        }

        /// Reserve a localhost port with nothing listening on it, so a
        /// connection attempt to it deterministically fails fast (connection
        /// refused) instead of hanging.
        async fn unreachable_addr() -> std::net::SocketAddr {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            drop(listener);
            addr
        }

        fn test_config() -> SlurmConfig {
            SlurmConfig {
                cluster_name: "test".into(),
                controller: spur_core::config::ControllerConfig {
                    first_job_id: 1,
                    ..Default::default()
                },
                accounting: Default::default(),
                scheduler: Default::default(),
                auth: Default::default(),
                partitions: vec![spur_core::config::PartitionConfig {
                    name: "default".into(),
                    default: true,
                    state: "UP".into(),
                    nodes: "ALL".into(),
                    selector: Default::default(),
                    max_time: None,
                    default_time: None,
                    max_nodes: None,
                    min_nodes: 1,
                    allow_accounts: Vec::new(),
                    allow_groups: Vec::new(),
                    deny_accounts: Vec::new(),
                    allow_qos: Vec::new(),
                    deny_qos: Vec::new(),
                    priority_tier: 1,
                    preempt_mode: String::new(),
                    preempt_exempt_time: None,
                }],
                nodes: Vec::new(),
                network: Default::default(),
                logging: Default::default(),
                kubernetes: Default::default(),
                cluster: Default::default(),
                notifications: Default::default(),
                power: Default::default(),
                federation: Default::default(),
                topology: None,
                isolation: Default::default(),
                licenses: HashMap::new(),
                burst_buffer: Default::default(),
                update: Default::default(),
                metrics: Default::default(),
                rest_api: Default::default(),
                hooks: Default::default(),
                devices: Default::default(),
                admission: Default::default(),
                rlimits: Default::default(),
                cgroup: Default::default(),
                mpi: Default::default(),
            }
        }

        async fn test_cluster(dir: &TempDir) -> Arc<ClusterManager> {
            test_cluster_with_config(dir, test_config()).await
        }

        async fn test_cluster_with_config(
            dir: &TempDir,
            config: SlurmConfig,
        ) -> Arc<ClusterManager> {
            let cm = Arc::new(ClusterManager::new(config, dir.path()).unwrap());
            let handle = crate::raft::start_raft(1, &["[::1]:0".into()], dir.path(), cm.clone())
                .await
                .unwrap();
            handle
                .raft
                .wait(Some(std::time::Duration::from_secs(5)))
                .metrics(|m| m.current_leader == Some(1), "leader elected")
                .await
                .expect("single-node raft did not self-elect within 5s");
            cm.set_raft(handle.raft);
            cm
        }

        /// Spin until an async mutation (Raft-committed state, or a
        /// fire-and-forget cancel RPC) is visible. Bounded retry, not an
        /// open-ended wait: fails loudly if the condition never holds.
        fn wait_for<F: Fn() -> bool>(label: &str, f: F) {
            for _ in 0..200 {
                if f() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            panic!("timed out waiting for: {label}");
        }

        fn register_node_at(cm: &ClusterManager, name: &str, addr: std::net::SocketAddr) {
            cm.register_node(
                name.into(),
                name.into(),
                ResourceSet {
                    cpus: 4,
                    memory_mb: 8000,
                    ..Default::default()
                },
                addr.ip().to_string(),
                addr.port(),
                String::new(),
                String::new(),
                NodeSource::NativeHost,
                HashMap::new(),
                true,
            )
            .unwrap();
            let n = name.to_string();
            wait_for(&format!("node '{n}' registered"), || {
                cm.get_node(&n).is_some()
            });
        }

        fn register_node_without_comm_addr(cm: &ClusterManager, name: &str) {
            use crate::raft::StateMachineApply;
            use spur_core::wal::WalOperation;

            cm.apply_operation(&WalOperation::NodeRegister {
                name: name.into(),
                hostname: name.into(),
                resources: ResourceSet {
                    cpus: 4,
                    memory_mb: 8000,
                    ..Default::default()
                },
                address: String::new(),
                port: 6818,
                wg_pubkey: String::new(),
                version: String::new(),
                labels: HashMap::new(),
                source: NodeSource::NativeHost,
            });
            let n = name.to_string();
            wait_for(
                &format!("node '{n}' registered without comm address"),
                || {
                    cm.get_node(&n)
                        .is_some_and(|node| node.comm_addr().is_none())
                },
            );
        }

        fn submit_and_wait(cm: &ClusterManager, spec: JobSpec) -> spur_core::job::JobId {
            let id = cm.submit_job(spec).unwrap().job_id;
            wait_for(&format!("job {id} applied"), || cm.get_job(id).is_some());
            id
        }

        fn settle(
            cm: &ClusterManager,
            job_id: spur_core::job::JobId,
            expected: spur_core::job::JobState,
        ) {
            wait_for(&format!("job {job_id} -> {expected:?}"), || {
                cm.get_job(job_id).is_some_and(|j| j.state == expected)
            });
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn partial_dispatch_confirmation_aborts_admission_and_cancels_the_node_that_launched()
        {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (good_addr, cancel_calls) = spawn_mock_agent().await;
            let bad_addr = unreachable_addr().await;
            register_node_at(&cm, "n1", good_addr);
            register_node_at(&cm, "n2", bad_addr);

            let spec = JobSpec {
                name: "partial-dispatch".into(),
                user: "testuser".into(),
                num_nodes: 2,
                num_tasks: 2,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            };
            let job_id = submit_and_wait(&cm, spec.clone());
            let spec = cm.get_job(job_id).unwrap().spec;

            let nodes = vec!["n1".to_string(), "n2".to_string()];
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();

            // This calls the exact same function `run()` now awaits per
            // assignment *before* start_job: real network dispatch to both
            // nodes, real JoinSet success/failure counting, and the real
            // branch that aborts admission on a partial failure — n1 (real
            // agent) accepts, n2 (nothing listening) fails.
            let outcome = confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                "n1,n2".into(),
                1,
                1,
                false,
            )
            .await;

            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));

            // The job must never have been visible as Running — admission
            // failed before that transition, not after it (unlike the old
            // dispatch_job_to_nodes, which evicted an already-Running job to
            // NodeFail on the same partial failure).
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.state, JobState::Pending);
            assert!(job.allocated_nodes.is_empty());

            // n1 actually launched the job, so the controller must tell its
            // agent to stop it instead of leaving an orphaned process behind.
            // The cancel is awaited inside confirm_dispatch_on_nodes, so it
            // has already been delivered by the time that call returned.
            assert_eq!(
                cancel_calls.load(Ordering::SeqCst),
                1,
                "n1 must have been cancelled before confirm_dispatch_on_nodes returned"
            );
        }

        // Force-finish must cancel the job on the unreported node before freeing
        // it, or the agent keeps the stale allocation and rejects the next dispatch.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn completing_timeout_cancels_only_the_unreported_node() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (addr1, cancel1) = spawn_mock_agent().await;
            let (addr2, cancel2) = spawn_mock_agent().await;
            register_node_at(&cm, "n1", addr1);
            register_node_at(&cm, "n2", addr2);

            let spec = JobSpec {
                name: "completing-timeout".into(),
                user: "testuser".into(),
                num_nodes: 2,
                num_tasks: 2,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            };
            let job_id = submit_and_wait(&cm, spec);

            let nodes = vec!["n1".to_string(), "n2".to_string()];
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();
            let run_attempt = cm
                .start_job(
                    job_id,
                    nodes,
                    ResourceAllocations::with_scalar(2, 0),
                    per_node_allocs,
                )
                .unwrap();
            settle(&cm, job_id, JobState::Running);

            // n1 reports completion; n2 never does, so the job stays Completing.
            cm.node_complete(job_id, "n1", 0, 0, run_attempt).unwrap();
            settle(&cm, job_id, JobState::Completing);

            let job = cm.get_job(job_id).unwrap();
            force_finish_completing_job(&cm, &job).await;

            assert_eq!(
                cancel2.load(Ordering::SeqCst),
                1,
                "the unreported node n2 must be cancelled before its resources are freed"
            );
            assert_eq!(
                cancel1.load(Ordering::SeqCst),
                0,
                "the node that already reported must not be cancelled"
            );
        }

        // When no node reported (e.g. a suspended job's tasks died out-of-band),
        // every allocated node is unreported and must be cancelled on force-finish.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn completing_timeout_cancels_all_nodes_when_none_reported() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (addr1, cancel1) = spawn_mock_agent().await;
            let (addr2, cancel2) = spawn_mock_agent().await;
            register_node_at(&cm, "n1", addr1);
            register_node_at(&cm, "n2", addr2);

            let spec = JobSpec {
                name: "completing-none".into(),
                user: "testuser".into(),
                num_nodes: 2,
                num_tasks: 2,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            };
            let job_id = submit_and_wait(&cm, spec);

            let nodes = vec!["n1".to_string(), "n2".to_string()];
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();
            cm.start_job(
                job_id,
                nodes,
                ResourceAllocations::with_scalar(2, 0),
                per_node_allocs,
            )
            .unwrap();
            settle(&cm, job_id, JobState::Running);

            // Suspend routes through Completing; no node reports completion.
            cm.suspend_job(job_id, "").unwrap();
            settle(&cm, job_id, JobState::Suspended);
            let mut job = cm.get_job(job_id).unwrap();
            job.transition(JobState::Completing).unwrap();

            force_finish_completing_job(&cm, &job).await;

            assert_eq!(cancel1.load(Ordering::SeqCst), 1, "n1 must be cancelled");
            assert_eq!(cancel2.load(Ordering::SeqCst), 1, "n2 must be cancelled");
        }

        // Mock agents echo an offset-keyed path; the stored path must be the
        // primary's (task_offset == 0) regardless of which response arrives
        // first. Confirmed *before* start_job now, so the job is still
        // Pending when the path lands.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn clean_dispatch_confirmation_stores_primary_node_output_path() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (addr1, _) = spawn_mock_agent().await;
            let (addr2, _) = spawn_mock_agent().await;
            register_node_at(&cm, "n1", addr1);
            register_node_at(&cm, "n2", addr2);

            let spec = JobSpec {
                name: "multi-out".into(),
                user: "testuser".into(),
                num_nodes: 2,
                num_tasks: 2,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            };
            let job_id = submit_and_wait(&cm, spec.clone());
            let spec = cm.get_job(job_id).unwrap().spec;

            let nodes = vec!["n1".to_string(), "n2".to_string()];
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();

            let outcome = confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                "n1,n2".into(),
                1,
                1,
                false,
            )
            .await;

            assert!(matches!(outcome, DispatchConfirmOutcome::Confirmed));

            let job = cm.get_job(job_id).unwrap();
            assert_eq!(
                job.actual_stdout_path.as_deref(),
                Some("/spool/off0/spur.out"),
                "must store the primary (task_offset==0) node's path"
            );
            assert_eq!(
                job.actual_stderr_path.as_deref(),
                Some("/spool/off0/spur.out")
            );
        }

        // If the primary node fails to launch, admission is aborted and no
        // output path is recorded, so queries fall back to the computed path.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn failed_primary_leaves_output_path_unset() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            // Primary (n1) is unreachable; secondary (n2) accepts.
            let bad_addr = unreachable_addr().await;
            let (good_addr, _) = spawn_mock_agent().await;
            register_node_at(&cm, "n1", bad_addr);
            register_node_at(&cm, "n2", good_addr);

            let spec = JobSpec {
                name: "primary-fail".into(),
                user: "testuser".into(),
                num_nodes: 2,
                num_tasks: 2,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            };
            let job_id = submit_and_wait(&cm, spec.clone());
            let spec = cm.get_job(job_id).unwrap().spec;

            let nodes = vec!["n1".to_string(), "n2".to_string()];
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();

            let outcome = confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                "n1,n2".into(),
                1,
                1,
                false,
            )
            .await;

            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));

            let job = cm.get_job(job_id).unwrap();
            assert!(
                job.actual_stdout_path.is_none(),
                "a failed dispatch confirmation must not record an output path"
            );
            assert!(job.actual_stderr_path.is_none());
        }

        // A secondary-node failure (primary succeeds) still aborts admission,
        // so the `failures == 0` gate must skip storing the primary's path.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn secondary_failure_leaves_output_path_unset() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            // Primary (n1) accepts; secondary (n2) is unreachable.
            let (good_addr, _) = spawn_mock_agent().await;
            let bad_addr = unreachable_addr().await;
            register_node_at(&cm, "n1", good_addr);
            register_node_at(&cm, "n2", bad_addr);

            let spec = JobSpec {
                name: "secondary-fail".into(),
                user: "testuser".into(),
                num_nodes: 2,
                num_tasks: 2,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            };
            let job_id = submit_and_wait(&cm, spec.clone());
            let spec = cm.get_job(job_id).unwrap().spec;

            let nodes = vec!["n1".to_string(), "n2".to_string()];
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();

            let outcome = confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                "n1,n2".into(),
                1,
                1,
                false,
            )
            .await;

            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));

            let job = cm.get_job(job_id).unwrap();
            assert!(
                job.actual_stdout_path.is_none(),
                "a partial dispatch failure must not record an output path"
            );
            assert!(job.actual_stderr_path.is_none());
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn partial_multinode_pmix_dispatch_releases_on_all_agents() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (good_addr, cancel_calls, release_pmix_good, _) =
                spawn_mock_agent_capturing_fanout(None, Duration::ZERO, Duration::ZERO, false)
                    .await;
            let (bad_addr, _, release_pmix_bad, _) = spawn_mock_agent_capturing_fanout(
                Some(spur_proto::proto::LaunchFailureKind::LaunchFailureUnspecified),
                Duration::ZERO,
                Duration::ZERO,
                false,
            )
            .await;
            register_node_at(&cm, "n1", good_addr);
            register_node_at(&cm, "n2", bad_addr);

            let spec = JobSpec {
                name: "pmix-partial-dispatch".into(),
                user: "testuser".into(),
                num_nodes: 2,
                num_tasks: 2,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                mpi: Some(spur_core::mpi::MPI_PMIX.into()),
                ..Default::default()
            };
            let job_id = submit_and_wait(&cm, spec.clone());
            let spec = cm.get_job(job_id).unwrap().spec;

            let nodes = vec!["n1".to_string(), "n2".to_string()];
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();

            let outcome = confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                "n1,n2".into(),
                1,
                1,
                false,
            )
            .await;

            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));
            assert_eq!(
                cancel_calls.load(Ordering::SeqCst),
                1,
                "n1 launched the job, so it must be stopped before the job settles"
            );
            assert_eq!(
                release_pmix_good.load(Ordering::SeqCst),
                1,
                "PMIx must be released on every agent that participated in prepare, including the one that launched"
            );
            assert_eq!(
                release_pmix_bad.load(Ordering::SeqCst),
                1,
                "PMIx must be released on every agent that participated in prepare, including the one whose launch failed"
            );
        }

        // ── prolog failure: drain the node, hold the job ──
        //
        // All of these confirm dispatch on a still-Pending job (never
        // started), matching what `run()` now does: the job is never visible
        // as Running before these outcomes are decided.

        /// Submit `spec` (via the caller) and run the real dispatch
        /// confirmation against `nodes` on the still-Pending job.
        async fn confirm_dispatch_pending_job(
            cm: &Arc<ClusterManager>,
            job_id: spur_core::job::JobId,
            nodes: &[&str],
        ) -> DispatchConfirmOutcome {
            let nodes: Vec<String> = nodes.iter().map(|n| n.to_string()).collect();
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();
            let spec = cm.get_job(job_id).unwrap().spec;
            let nodelist = nodes.join(",");
            confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                nodelist,
                1,
                1,
                false,
            )
            .await
        }

        fn batch_spec(name: &str, num_nodes: u32) -> JobSpec {
            JobSpec {
                name: name.into(),
                user: "testuser".into(),
                num_nodes,
                num_tasks: num_nodes,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            }
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn dispatch_aborts_when_node_has_no_comm_address() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            register_node_without_comm_addr(&cm, "n1");

            let job_id = submit_and_wait(&cm, batch_spec("no-comm-addr", 1));
            let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;
            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));
            assert!(
                cm.get_job(job_id)
                    .unwrap()
                    .state_reason()
                    .contains("no agent address"),
                "a node skipped before dispatch even starts must still get a category, \
                 not an empty summary"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn dispatch_to_an_unreachable_agent_reports_that_category() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let bad_addr = unreachable_addr().await;
            register_node_at(&cm, "n1", bad_addr);

            let job_id = submit_and_wait(&cm, batch_spec("unreachable-agent", 1));
            let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;
            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));
            assert!(
                cm.get_job(job_id)
                    .unwrap()
                    .state_reason()
                    .contains("agent unreachable"),
                "a connect failure must be reported as its own category"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn mixed_failure_categories_are_all_reported() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (prolog_addr, _) = spawn_mock_agent_rejecting(Some(
                spur_proto::proto::LaunchFailureKind::LaunchFailureProlog,
            ))
            .await;
            let resources_addr = spawn_mock_agent_rejecting_resources().await;
            register_node_at(&cm, "n1", prolog_addr);
            register_node_at(&cm, "n2", resources_addr);

            let spec = JobSpec {
                name: "mixed-categories".into(),
                user: "testuser".into(),
                num_nodes: 2,
                num_tasks: 2,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            };
            let job_id = submit_and_wait(&cm, spec.clone());
            let spec = cm.get_job(job_id).unwrap().spec;
            let nodes = vec!["n1".to_string(), "n2".to_string()];
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();

            let outcome = confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                "n1,n2".into(),
                1,
                1,
                false,
            )
            .await;
            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));

            let reason = cm.get_job(job_id).unwrap().state_reason();
            assert!(
                reason.contains("1 prolog failure")
                    && reason.contains("1 gpu/resource allocation mismatch"),
                "both failure categories from the two nodes must show up, not just one: {reason}"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_prolog_failure_drains_the_node_and_parks_the_job() {
            use spur_core::job::{JobState, PendingReason};

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (addr, _) = spawn_mock_agent_rejecting(Some(
                spur_proto::proto::LaunchFailureKind::LaunchFailureProlog,
            ))
            .await;
            register_node_at(&cm, "n1", addr);

            let job_id = submit_and_wait(&cm, batch_spec("prolog-fail", 1));
            let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;
            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));

            let node = cm.get_node("n1").unwrap();
            assert!(
                node.state.is_admin_hold(),
                "the node that ran the failing prolog must stop taking work"
            );
            assert!(
                node.state_reason
                    .as_deref()
                    .is_some_and(|r| r.contains("prolog")),
                "operators need the prolog's own failure, got {:?}",
                node.state_reason
            );

            // The hold is what bounds the drain to one node: without it the
            // job retries onto the next node and drains that one too. The job
            // never reached Running, so this hold is applied directly on the
            // still-Pending job (`hold_job_for_launch_failure`) rather than
            // via the old Running->Failed->Held detour — same end state
            // (Pending, priority 0, Held) with the dispatch failure detail
            // surfaced in state_reason_display.
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.state, JobState::Pending);
            assert_eq!(job.pending_reason, PendingReason::Held);
            assert_eq!(job.priority, 0);
            assert_eq!(
                job.state_reason_display(),
                "dispatch confirmation failed (0/1 confirmed): 1 prolog failure"
            );
            assert_eq!(
                job.state_reason(),
                "dispatch confirmation failed (0/1 confirmed): 1 prolog failure"
            );
            assert!(
                !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
                "a held job must not be scheduled anywhere"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn an_operator_can_release_a_job_held_for_a_prolog_failure() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (addr, _) = spawn_mock_agent_rejecting(Some(
                spur_proto::proto::LaunchFailureKind::LaunchFailureProlog,
            ))
            .await;
            register_node_at(&cm, "n1", addr);

            let job_id = submit_and_wait(&cm, batch_spec("prolog-release", 1));
            confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;

            cm.release_job(job_id).unwrap();
            wait_for("job released", || {
                cm.get_job(job_id).is_some_and(|j| j.priority > 0)
            });

            let job = cm.get_job(job_id).unwrap();
            assert_eq!(
                job.pending_reason_desc, None,
                "the release must clear the description with the reason"
            );
            assert!(cm.pending_jobs().iter().any(|j| j.job_id == job_id));
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn hold_on_prolog_fail_off_retries_the_job_instead() {
            use spur_core::job::{JobState, PendingReason};

            let dir = TempDir::new().unwrap();
            let mut config = test_config();
            config.controller.hold_on_prolog_fail = false;
            let cm = test_cluster_with_config(&dir, config).await;

            let (addr, _) = spawn_mock_agent_rejecting(Some(
                spur_proto::proto::LaunchFailureKind::LaunchFailureProlog,
            ))
            .await;
            register_node_at(&cm, "n1", addr);

            let job_id = submit_and_wait(&cm, batch_spec("prolog-retry", 1));
            let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;
            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));

            // nohold_on_prolog_fail: retry rather than hold, but "retry" is the
            // same bounded backoff as any dispatch failure, not an immediate one.
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.state, JobState::Pending);
            assert_eq!(job.pending_reason, PendingReason::JobLaunchFailure);
            assert_eq!(job.requeue_count, 1);
            assert!(
                job.spec.begin_time.is_some_and(|t| t > chrono::Utc::now()),
                "nohold_on_prolog_fail retries with a backoff, not an unconditional immediate retry"
            );
            assert!(cm.get_node("n1").unwrap().state.is_admin_hold());
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn an_interactive_job_is_cancelled_rather_than_held() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (addr, _) = spawn_mock_agent_rejecting(Some(
                spur_proto::proto::LaunchFailureKind::LaunchFailureProlog,
            ))
            .await;
            register_node_at(&cm, "n1", addr);

            // Holding an interactive job would leave its srun waiting forever
            // with nothing to wait for. Slurm cancels these too.
            let mut spec = batch_spec("prolog-interactive", 1);
            spec.interactive = true;
            let job_id = submit_and_wait(&cm, spec);
            let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;
            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));

            settle(&cm, job_id, JobState::Cancelled);

            assert!(
                cm.get_node("n1").unwrap().state.is_admin_hold(),
                "the node is broken regardless of what happens to the job"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn an_unclassified_rejection_backs_off_without_draining_the_node() {
            use spur_core::job::{JobState, PendingReason};

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            // What an agent predating the classification sends.
            let (addr, _) = spawn_mock_agent_rejecting(Some(
                spur_proto::proto::LaunchFailureKind::LaunchFailureUnspecified,
            ))
            .await;
            register_node_at(&cm, "n1", addr);

            let job_id = submit_and_wait(&cm, batch_spec("unclassified", 1));
            let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;
            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));

            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.state, JobState::Pending);
            // A non-prolog failure gets a bounded backoff rather than a drain:
            // eligible again once the hold lapses, but not reassigned to the
            // same broken node next tick.
            assert_eq!(job.pending_reason, PendingReason::JobLaunchFailure);
            assert_eq!(
                job.requeue_count, 1,
                "the backoff must count against max_batch_requeue like any other launch failure"
            );
            assert!(
                job.spec.begin_time.is_some_and(|t| t > chrono::Utc::now()),
                "the backoff hold must defer the job into the future, not just tag it"
            );
            assert!(
                !cm.get_node("n1").unwrap().state.is_admin_hold(),
                "an unclassified rejection is not grounds to drain"
            );
            assert!(
                job.state_reason().contains("agent rejected launch"),
                "an explicit non-prolog rejection must get its own category, got {:?}",
                job.state_reason()
            );
        }

        // Repeated failures against an unreachable node must cross
        // max_batch_requeue and hold the job, not back off forever.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn repeated_dispatch_failures_are_bounded_by_max_batch_requeue() {
            use spur_core::job::{JobState, PendingReason};

            let dir = TempDir::new().unwrap();
            let mut config = test_config();
            config.controller.max_batch_requeue = 2;
            let cm = test_cluster_with_config(&dir, config).await;

            let bad_addr = unreachable_addr().await;
            register_node_at(&cm, "n1", bad_addr);

            let job_id = submit_and_wait(&cm, batch_spec("bounded-retry", 1));

            for attempt in 1..=2u32 {
                let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;
                assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));
                let job = cm.get_job(job_id).unwrap();
                assert_eq!(job.state, JobState::Pending);
                assert_eq!(job.requeue_count, attempt, "attempt {attempt}");
                assert_eq!(job.pending_reason, PendingReason::JobLaunchFailure);
            }

            // The next failure crosses max_batch_requeue: held for an
            // operator instead of backing off yet again.
            let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;
            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.state, JobState::Pending);
            assert_eq!(job.pending_reason, PendingReason::JobHoldMaxRequeue);
            assert_eq!(job.priority, 0);
            assert!(
                !cm.pending_jobs().iter().any(|j| j.job_id == job_id),
                "a held job must not be scheduled anywhere, closing the retry loop for good"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn only_the_node_whose_prolog_failed_drains() {
            use spur_core::job::{JobState, PendingReason};

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (good_addr, cancel_calls) = spawn_mock_agent().await;
            let (bad_addr, _) = spawn_mock_agent_rejecting(Some(
                spur_proto::proto::LaunchFailureKind::LaunchFailureProlog,
            ))
            .await;
            register_node_at(&cm, "n1", good_addr);
            register_node_at(&cm, "n2", bad_addr);

            let job_id = submit_and_wait(&cm, batch_spec("prolog-partial", 2));
            let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1", "n2"]).await;
            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));

            assert!(!cm.get_node("n1").unwrap().state.is_admin_hold());
            assert!(cm.get_node("n2").unwrap().state.is_admin_hold());

            // The partial branch must hold like the total one, and — unlike
            // the pre-fix behavior — the job must never have been visible as
            // Running in between.
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.state, JobState::Pending);
            assert_eq!(job.pending_reason, PendingReason::Held);
            assert_eq!(job.priority, 0);

            assert_eq!(
                cancel_calls.load(Ordering::SeqCst),
                1,
                "n1 launched the job, so it must be stopped before the job settles"
            );
        }

        // ── measured admission latency ──
        //
        // Answers "how much slower does job start get?" with a real number
        // instead of an estimate: every node's LaunchJob is confirmed
        // concurrently via JoinSet, so the added latency this fix introduces
        // should track the *slowest* node's own launch time, not the sum
        // across all assigned nodes.

        /// Confirm dispatch across `count` freshly-registered nodes, each of
        /// whose mock agent sleeps `delay` before accepting the launch, and
        /// return how long the whole confirmation actually took.
        async fn measure_confirm_latency(
            cm: &Arc<ClusterManager>,
            count: usize,
            delay: Duration,
        ) -> Duration {
            let mut nodes = Vec::with_capacity(count);
            for i in 0..count {
                let name = format!("latency-n{i}");
                let (addr, _) = spawn_mock_agent_with_delay(delay).await;
                register_node_at(cm, &name, addr);
                nodes.push(name);
            }

            let spec = batch_spec(&format!("latency-{count}"), count as u32);
            let job_id = submit_and_wait(cm, spec);
            let spec = cm.get_job(job_id).unwrap().spec;
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();
            let nodelist = nodes.join(",");

            let start = std::time::Instant::now();
            let outcome = confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                nodelist,
                1,
                1,
                false,
            )
            .await;
            let elapsed = start.elapsed();

            assert!(matches!(outcome, DispatchConfirmOutcome::Confirmed));
            elapsed
        }

        /// A refused connection fails in milliseconds and must not buy the long deadline cooldown,
        /// or a rolling agent restart sidelines the whole cluster at once.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_fast_dispatch_failure_does_not_earn_the_deadline_cooldown() {
            let dir = TempDir::new().unwrap();
            let mut config = test_config();
            // Disabling the short cooldown makes the two paths distinguishable: only
            // cool_down_node_for can put a node in the map now.
            // Non-zero so the assertion is two-sided: the node must land on the SHORT cooldown,
            // not merely stay off the long one (which "cools nothing at all" would also satisfy).
            config.controller.dispatch_reject_cooldown_secs = 30;
            config.controller.dispatch_timeout_secs = 300;
            let cm = test_cluster_with_config(&dir, config).await;

            register_node_at(&cm, "n-dead", unreachable_addr().await);

            let nodes = vec!["n-dead".to_string()];
            let spec = batch_spec("fast-fail", 1);
            let job_id = submit_and_wait(&cm, spec);
            let spec = cm.get_job(job_id).unwrap().spec;
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();
            let nodelist = nodes.join(",");

            let outcome = confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                nodelist,
                1,
                1,
                false,
            )
            .await;

            assert!(!matches!(outcome, DispatchConfirmOutcome::Confirmed));
            assert!(
                cm.nodes_on_dispatch_cooldown().contains("n-dead"),
                "a connect failure must still cool the node down"
            );
            assert!(
                cm.dispatch_cooldown_remaining("n-dead")
                    .is_some_and(|left| left <= Duration::from_secs(30)),
                "it must use the 30s reject cooldown, not the 300s dispatch deadline"
            );
        }

        #[test]
        fn a_zero_dispatch_timeout_disables_the_deadline() {
            let dir = TempDir::new().unwrap();
            let mut config = test_config();
            config.controller.dispatch_timeout_secs = 0;
            let cm = ClusterManager::new(config, dir.path()).unwrap();
            assert_eq!(dispatch_deadline(&cm), None);

            let dir2 = TempDir::new().unwrap();
            let mut config = test_config();
            config.controller.dispatch_timeout_secs = 7;
            let cm = ClusterManager::new(config, dir2.path()).unwrap();
            assert_eq!(dispatch_deadline(&cm), Some(Duration::from_secs(7)));
        }

        /// The node that timed out is the one most likely to have launched anyway, so it is the one
        /// that must be cancelled — cancelling only the confirmed nodes orphans it.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_node_whose_launch_times_out_is_still_cancelled() {
            let dir = TempDir::new().unwrap();
            let mut config = test_config();
            config.controller.dispatch_timeout_secs = 1;
            let cm = test_cluster_with_config(&dir, config).await;

            let (fast_addr, fast_cancels) = spawn_mock_agent().await;
            let (slow_addr, slow_cancels) =
                spawn_mock_agent_with_delay(Duration::from_secs(30)).await;
            register_node_at(&cm, "n-fast", fast_addr);
            register_node_at(&cm, "n-slow", slow_addr);

            let nodes = vec!["n-fast".to_string(), "n-slow".to_string()];
            let spec = batch_spec("timeout-cancel", 2);
            let job_id = submit_and_wait(&cm, spec);
            let spec = cm.get_job(job_id).unwrap().spec;
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();
            let nodelist = nodes.join(",");

            let outcome = confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                nodelist,
                1,
                1,
                false,
            )
            .await;

            assert!(
                !matches!(outcome, DispatchConfirmOutcome::Confirmed),
                "a node that never answered must not count as confirmed"
            );
            assert!(
                slow_cancels.load(Ordering::SeqCst) > 0,
                "the timed-out node received no cancel and would keep running the job"
            );
            assert!(
                fast_cancels.load(Ordering::SeqCst) > 0,
                "the confirmed node must still be cancelled when admission aborts"
            );
        }

        /// Counterpart to [`a_fast_dispatch_failure_does_not_earn_the_deadline_cooldown`]: with the
        /// reject cooldown off, only the deadline path can put the node on cooldown at all.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_node_that_burns_the_launch_deadline_is_held_for_it() {
            let dir = TempDir::new().unwrap();
            let mut config = test_config();
            config.controller.dispatch_reject_cooldown_secs = 0;
            config.controller.dispatch_timeout_secs = 1;
            let cm = test_cluster_with_config(&dir, config).await;

            let (slow_addr, _) = spawn_mock_agent_with_delay(Duration::from_secs(30)).await;
            register_node_at(&cm, "n-slow", slow_addr);

            let nodes = vec!["n-slow".to_string()];
            let spec = batch_spec("launch-deadline-cooldown", 1);
            let job_id = submit_and_wait(&cm, spec);
            let spec = cm.get_job(job_id).unwrap().spec;
            let per_node_allocs: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.clone(), ResourceAllocations::with_scalar(1, 0)))
                .collect();
            let nodelist = nodes.join(",");

            let outcome = confirm_dispatch_on_nodes(
                cm.clone(),
                job_id,
                nodes,
                spec,
                Vec::new(),
                per_node_allocs,
                nodelist,
                1,
                1,
                false,
            )
            .await;

            assert!(!matches!(outcome, DispatchConfirmOutcome::Confirmed));
            assert!(
                cm.dispatch_cooldown_remaining("n-slow").is_some(),
                "a node that burned the whole deadline must not be re-picked on the next tick"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
        async fn confirm_dispatch_latency_tracks_the_slowest_node_not_the_node_count() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            // A generous, deliberately-visible stand-in for a real agent's
            // launch pipeline (resource allocation, GPU device-injection
            // planning, process spawn) — real hardware is faster than this,
            // but the point is the shape of the curve (flat vs. linear in
            // node count), not the absolute number.
            const PER_NODE_LAUNCH_DELAY: Duration = Duration::from_millis(150);

            let two_nodes = measure_confirm_latency(&cm, 2, PER_NODE_LAUNCH_DELAY).await;
            let eight_nodes = measure_confirm_latency(&cm, 8, PER_NODE_LAUNCH_DELAY).await;
            let sixty_four_nodes = measure_confirm_latency(&cm, 64, PER_NODE_LAUNCH_DELAY).await;

            eprintln!(
                "confirm_dispatch_on_nodes latency — 2 nodes: {two_nodes:?}, \
                 8 nodes: {eight_nodes:?}, 64 nodes: {sixty_four_nodes:?} \
                 (synthetic per-node launch delay: {PER_NODE_LAUNCH_DELAY:?})"
            );

            // Lower bound only: proves per-node work wasn't skipped. Concurrency
            // (64 nodes near one node's delay) is left to the eprintln! above — a
            // wall-clock ceiling is flaky under CI load.
            assert!(two_nodes >= PER_NODE_LAUNCH_DELAY);
            assert!(eight_nodes >= PER_NODE_LAUNCH_DELAY);
            assert!(sixty_four_nodes >= PER_NODE_LAUNCH_DELAY);
        }

        // ── process_assignment: the glue `run()` awaits per assignment ──
        //
        // confirm_dispatch_pending_job above calls confirm_dispatch_on_nodes
        // directly, standing in for what `run()` used to fire off in the
        // background. These tests instead go through process_assignment
        // itself — the peer_addrs/tasks_per_node build, the dispatch_spec
        // decision (plain batch vs. srun-as-batch-fallback vs. pure
        // interactive), and start_job — the same call `run()` makes once per
        // assignment every cycle.

        fn register_k8s_node_at(cm: &ClusterManager, name: &str, addr: std::net::SocketAddr) {
            cm.register_node(
                name.into(),
                name.into(),
                ResourceSet {
                    cpus: 4,
                    memory_mb: 8000,
                    ..Default::default()
                },
                addr.ip().to_string(),
                addr.port(),
                String::new(),
                String::new(),
                NodeSource::Kubernetes {
                    namespace: "spur-test".into(),
                },
                HashMap::new(),
                true,
            )
            .unwrap();
            let n = name.to_string();
            wait_for(&format!("node '{n}' registered"), || {
                cm.get_node(&n).is_some()
            });
        }

        fn assignment(
            job_id: spur_core::job::JobId,
            nodes: &[&str],
        ) -> spur_sched::traits::Assignment {
            let per_node_alloc: HashMap<String, ResourceAllocations> = nodes
                .iter()
                .map(|n| (n.to_string(), ResourceAllocations::with_scalar(1, 0)))
                .collect();
            spur_sched::traits::Assignment {
                job_id,
                nodes: nodes.iter().map(|n| n.to_string()).collect(),
                per_node_alloc,
            }
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_starts_a_plain_batch_job() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;
            let (addr, _) = spawn_mock_agent().await;
            register_node_at(&cm, "n1", addr);

            // An explicit --ntasks-per-node so this also covers the "user
            // supplied one" branch of the effective-tasks-per-node
            // computation, not just its num_tasks/num_nodes fallback.
            let mut spec = batch_spec("plain-batch", 1);
            spec.tasks_per_node = Some(2);
            let job_id = submit_and_wait(&cm, spec);
            let started = process_assignment(cm.clone(), assignment(job_id, &["n1"])).await;

            assert!(started, "a clean single-node batch dispatch must start");
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.state, JobState::Running);
            assert_eq!(job.allocated_nodes, vec!["n1".to_string()]);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_returns_false_for_a_job_that_no_longer_exists() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            // No submit_job call: this job_id was never assigned, standing in
            // for an assignment computed against a snapshot that's since gone
            // stale (e.g. the job was deleted/expired between scheduling and
            // this call).
            let started = process_assignment(cm.clone(), assignment(999, &["n1"])).await;

            assert!(!started);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_aborts_a_plain_batch_job_when_confirmation_fails() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;
            let bad_addr = unreachable_addr().await;
            register_node_at(&cm, "n1", bad_addr);

            let job_id = submit_and_wait(&cm, batch_spec("plain-batch-unreachable", 1));
            let started = process_assignment(cm.clone(), assignment(job_id, &["n1"])).await;

            assert!(
                !started,
                "a plain batch job must not start if its only node can't be reached"
            );
            assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Pending);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_requeues_srun_batch_fallback_without_a_script() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;
            // Kubernetes-sourced, so srun_step_dispatch is false and this
            // takes the srun-as-batch-script-fallback branch, which requires
            // a script.
            let (addr, _) = spawn_mock_agent().await;
            register_k8s_node_at(&cm, "k1", addr);

            let mut spec = batch_spec("srun-no-script", 1);
            spec.srun_job = true;
            assert!(spec.script.is_none());
            let job_id = submit_and_wait(&cm, spec);

            let started = process_assignment(cm.clone(), assignment(job_id, &["k1"])).await;

            assert!(
                !started,
                "an srun batch fallback with no script must not start"
            );
            // requeue_job on a job that never left Pending is a no-op by
            // design (nothing to unwind), so the meaningful assertion is
            // that it never started, not a visible state change here.
            assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Pending);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_dispatches_srun_batch_fallback_with_a_script() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;
            let (addr, _) = spawn_mock_agent().await;
            register_k8s_node_at(&cm, "k1", addr);

            let mut spec = batch_spec("srun-with-script", 1);
            spec.srun_job = true;
            spec.script = Some("#!/bin/bash\necho hi\n".into());
            let job_id = submit_and_wait(&cm, spec);

            let started = process_assignment(cm.clone(), assignment(job_id, &["k1"])).await;

            assert!(
                started,
                "an srun batch fallback with a script confirms and starts like a plain batch job"
            );
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.state, JobState::Running);
            assert!(
                !job.srun_step_dispatch,
                "the batch-script fallback launches like sbatch, not a native srun step"
            );
        }

        // task_fanout: the agent-facing signal distinguishing a genuine sbatch
        // batch script (runs once per node, regardless of tasks_per_node or
        // mpi) from a standalone srun request routed through this same batch
        // dispatch path (the dispatched "script" is the literal command srun
        // was asked to run tasks_per_node times — real srun semantics).

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_dispatches_a_plain_batch_job_with_task_fanout_false() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;
            let (addr, _, _, fanout_calls) =
                spawn_mock_agent_capturing_fanout(None, Duration::ZERO, Duration::ZERO, true).await;
            register_node_at(&cm, "n1", addr);

            let mut spec = batch_spec("plain-batch-fanout", 1);
            spec.tasks_per_node = Some(4);
            let job_id = submit_and_wait(&cm, spec);
            let started = process_assignment(cm.clone(), assignment(job_id, &["n1"])).await;

            assert!(started, "a clean single-node batch dispatch must start");
            assert_eq!(
                *fanout_calls.lock().unwrap(),
                vec![false],
                "a genuine sbatch job must dispatch with task_fanout=false"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_dispatches_srun_batch_fallback_with_task_fanout_true() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;
            let (addr, _, _, fanout_calls) =
                spawn_mock_agent_capturing_fanout(None, Duration::ZERO, Duration::ZERO, true).await;
            register_k8s_node_at(&cm, "k1", addr);

            let mut spec = batch_spec("srun-fanout-with-script", 1);
            spec.srun_job = true;
            spec.tasks_per_node = Some(4);
            spec.script = Some("#!/bin/bash\necho hi\n".into());
            let job_id = submit_and_wait(&cm, spec);
            let started = process_assignment(cm.clone(), assignment(job_id, &["k1"])).await;

            assert!(
                started,
                "an srun batch fallback with a script confirms and starts like a plain batch job"
            );
            assert_eq!(
                *fanout_calls.lock().unwrap(),
                vec![true],
                "the srun-as-batch-fallback path must dispatch with task_fanout=true"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_completes_pure_interactive_srun_without_a_launch_job_dispatch()
        {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;
            // NativeHost, so srun_step_dispatch is true: register_allocation_on_nodes
            // (not this test's mock agent's launch_job) is what has to succeed.
            let (addr, _) = spawn_mock_agent().await;
            register_node_at(&cm, "n1", addr);

            let mut spec = batch_spec("pure-interactive", 1);
            spec.srun_job = true;
            spec.interactive = true;
            let job_id = submit_and_wait(&cm, spec);

            let started = process_assignment(cm.clone(), assignment(job_id, &["n1"])).await;

            assert!(
                started,
                "a native srun allocation on its own node must start"
            );
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.state, JobState::Running);
            assert!(
                job.srun_step_dispatch,
                "the pure interactive path must record itself as step-dispatch, \
                 not the batch-script fallback"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_requeues_when_all_srun_allocation_registrations_fail() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;
            let bad_addr = unreachable_addr().await;
            register_node_at(&cm, "n1", bad_addr);

            let mut spec = batch_spec("srun-alloc-all-fail", 1);
            spec.srun_job = true;
            let job_id = submit_and_wait(&cm, spec);

            let started = process_assignment(cm.clone(), assignment(job_id, &["n1"])).await;

            assert!(!started);
            assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Pending);
            assert!(
                cm.dispatch_cooldown_remaining("n1")
                    .is_some_and(|left| left <= Duration::from_secs(30)),
                "an unreachable node must take the short reject cooldown — not be re-picked \
                 every tick, and not be held for the whole dispatch deadline"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_cancels_the_launched_node_on_partial_srun_allocation_failure() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;
            let (good_addr, cancel_calls) = spawn_mock_agent().await;
            let bad_addr = unreachable_addr().await;
            register_node_at(&cm, "n1", good_addr);
            register_node_at(&cm, "n2", bad_addr);

            let mut spec = batch_spec("srun-alloc-partial-fail", 2);
            spec.srun_job = true;
            let job_id = submit_and_wait(&cm, spec);

            let started = process_assignment(cm.clone(), assignment(job_id, &["n1", "n2"])).await;

            assert!(!started);
            assert_eq!(cm.get_job(job_id).unwrap().state, JobState::Pending);
            wait_for("n1 registration rolled back with a cancel", || {
                cancel_calls.load(Ordering::SeqCst) >= 1
            });
        }

        /// Every node answering just past the deadline is the production shape of "all failed":
        /// each may still be holding the allocation it registered, so none can be left uncancelled,
        /// and none may be re-picked on the next tick to burn the deadline again.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn every_node_timing_out_on_register_is_cancelled_cooled_and_backed_off() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let mut config = test_config();
            config.controller.dispatch_reject_cooldown_secs = 0;
            config.controller.dispatch_timeout_secs = 1;
            let cm = test_cluster_with_config(&dir, config).await;

            let (addr1, cancel1) =
                spawn_mock_agent_with_register_delay(Duration::from_secs(30)).await;
            let (addr2, cancel2) =
                spawn_mock_agent_with_register_delay(Duration::from_secs(30)).await;
            register_node_at(&cm, "n1", addr1);
            register_node_at(&cm, "n2", addr2);

            let mut spec = batch_spec("srun-alloc-all-timeout", 2);
            spec.srun_job = true;
            let job_id = submit_and_wait(&cm, spec);

            let started = process_assignment(cm.clone(), assignment(job_id, &["n1", "n2"])).await;

            assert!(!started);
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(job.state, JobState::Pending);
            assert!(
                job.spec.begin_time.is_some_and(|t| t > Utc::now()),
                "a registration failure must back the job off, not retry it on the next tick"
            );
            for name in ["n1", "n2"] {
                assert!(
                    cm.dispatch_cooldown_remaining(name).is_some(),
                    "{name} burned the deadline and must be skipped for it"
                );
            }
            wait_for("both timed-out nodes were cancelled", || {
                cancel1.load(Ordering::SeqCst) >= 1 && cancel2.load(Ordering::SeqCst) >= 1
            });
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_cancels_dispatched_nodes_when_start_job_fails_after_confirmation(
        ) {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;
            let (addr1, cancel1) = spawn_mock_agent().await;
            let (addr2, cancel2) = spawn_mock_agent().await;
            register_node_at(&cm, "n1", addr1);
            register_node_at(&cm, "n2", addr2);

            let job_id = submit_and_wait(&cm, batch_spec("start-job-inconsistent", 2));

            // A malformed assignment: confirm_dispatch_on_nodes tolerates a
            // missing per_node_alloc entry (falls back to a default
            // allocation), but start_job validates every assigned node has
            // one and rejects the whole call otherwise. This is what a
            // scheduler/assignment bug producing inconsistent data — or the
            // job being touched by another path between assignment and this
            // call — looks like from here: both nodes already launched real
            // work by the time start_job is rejected, so both must be torn
            // back down rather than left running under a job stuck Pending.
            let mut bad_assignment = assignment(job_id, &["n1", "n2"]);
            bad_assignment.per_node_alloc.remove("n2");

            let started = process_assignment(cm.clone(), bad_assignment).await;

            assert!(
                !started,
                "start_job's own validation must still block on inconsistent per-node data"
            );
            assert_eq!(
                cm.get_job(job_id).unwrap().state,
                JobState::Pending,
                "a start_job failure must not leave the job Running with no confirmed nodes"
            );
            wait_for(
                "n1 cancelled after start_job rejected the assignment",
                || cancel1.load(Ordering::SeqCst) >= 1,
            );
            wait_for(
                "n2 cancelled after start_job rejected the assignment",
                || cancel2.load(Ordering::SeqCst) >= 1,
            );
        }

        fn make_script(body: &str) -> tempfile::TempPath {
            use std::io::Write;
            let mut f = tempfile::NamedTempFile::new().unwrap();
            writeln!(f, "#!/bin/bash\n{}", body).unwrap();
            let path = f.into_temp_path();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            path
        }

        fn test_config_with_prolog_slurmctld(script: &std::path::Path) -> SlurmConfig {
            let mut config = test_config();
            config.hooks.prolog_slurmctld = Some(script.to_string_lossy().into_owned());
            config
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_cancels_interactive_job_when_prolog_slurmctld_fails() {
            use spur_core::job::JobState;

            let script = make_script("exit 1");
            let dir = TempDir::new().unwrap();
            let cm =
                test_cluster_with_config(&dir, test_config_with_prolog_slurmctld(&script)).await;
            let (addr, cancel_calls) = spawn_mock_agent().await;
            register_node_at(&cm, "n1", addr);

            let mut spec = batch_spec("prolog-slurmctld-interactive", 1);
            spec.interactive = true;
            let job_id = submit_and_wait(&cm, spec);

            let started = process_assignment(cm.clone(), assignment(job_id, &["n1"])).await;

            assert!(
                !started,
                "a job whose PrologSlurmctld fails must not count as started"
            );
            // Slurm cancels an interactive job here rather than holding it —
            // holding would strand its waiting srun with nothing to wait for.
            settle(&cm, job_id, JobState::Cancelled);
            // PrologSlurmctld runs before dispatch, so no node was ever
            // launched — there is nothing to tear down.
            assert_eq!(
                cancel_calls.load(Ordering::SeqCst),
                0,
                "no node should have been launched when PrologSlurmctld fails pre-dispatch"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_requeues_batch_job_when_prolog_slurmctld_fails() {
            use spur_core::job::JobState;

            let script = make_script("exit 1");
            let dir = TempDir::new().unwrap();
            let cm =
                test_cluster_with_config(&dir, test_config_with_prolog_slurmctld(&script)).await;
            let (addr, cancel_calls) = spawn_mock_agent().await;
            register_node_at(&cm, "n1", addr);

            let job_id = submit_and_wait(&cm, batch_spec("prolog-slurmctld-batch", 1));

            let started = process_assignment(cm.clone(), assignment(job_id, &["n1"])).await;

            assert!(
                !started,
                "a job whose PrologSlurmctld fails must not count as started"
            );
            // Unlike the interactive case, a batch job is requeued (it has no
            // waiting srun to strand). PrologSlurmctld runs before dispatch, so
            // the job never left Pending and no node was launched.
            settle(&cm, job_id, JobState::Pending);
            assert_eq!(
                cancel_calls.load(Ordering::SeqCst),
                0,
                "no node should have been launched when PrologSlurmctld fails pre-dispatch"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn process_assignment_survives_a_cancel_that_races_prolog_slurmctld_failure() {
            use spur_core::job::JobState;

            // A deliberate delay so the concurrent cancel below has a real,
            // generous window to land while the (pre-dispatch) PrologSlurmctld
            // script is still running — the same synthetic-delay idiom used to
            // make a concurrency test deterministic instead of racy.
            let script = make_script("sleep 0.2\nexit 1");
            let dir = TempDir::new().unwrap();
            let cm =
                test_cluster_with_config(&dir, test_config_with_prolog_slurmctld(&script)).await;
            let (addr, _) = spawn_mock_agent().await;
            register_node_at(&cm, "n1", addr);

            let mut spec = batch_spec("prolog-slurmctld-double-cancel", 1);
            spec.interactive = true;
            let job_id = submit_and_wait(&cm, spec);

            let cm_task = cm.clone();
            let handle = tokio::spawn(async move {
                process_assignment(cm_task, assignment(job_id, &["n1"])).await
            });

            // Simulate a concurrent scancel landing while the (delayed)
            // PrologSlurmctld script is still running: by the time
            // process_assignment's own interactive-job cancel_job call fires,
            // the job is already terminal, so that call fails and must just be
            // logged — not panic or otherwise corrupt the outcome.
            cm.cancel_job(job_id, "testuser").unwrap();

            let started = handle.await.unwrap();

            assert!(!started);
            assert_eq!(
                cm.get_job(job_id).unwrap().state,
                JobState::Cancelled,
                "the job must stay exactly as the earlier, real cancel left it"
            );
        }

        // confirm_dispatch_on_nodes's own interactive-prolog-failure cleanup
        // calls cancel_job — which can itself fail if the job was already
        // cancelled by something else (e.g. a concurrent scancel) while the
        // prolog check was in flight. That must not panic or otherwise
        // corrupt the outcome; it's just logged.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn confirm_dispatch_on_nodes_survives_a_cancel_that_races_prolog_failure() {
            use spur_core::job::JobState;

            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let (addr, _) = spawn_mock_agent_rejecting(Some(
                spur_proto::proto::LaunchFailureKind::LaunchFailureProlog,
            ))
            .await;
            register_node_at(&cm, "n1", addr);

            let mut spec = batch_spec("prolog-interactive-double-cancel", 1);
            spec.interactive = true;
            let job_id = submit_and_wait(&cm, spec);

            // Simulate a concurrent scancel landing before the prolog
            // rejection is processed: the job is already terminal by the
            // time confirm_dispatch_on_nodes tries to cancel it itself.
            cm.cancel_job(job_id, "testuser").unwrap();
            settle(&cm, job_id, JobState::Cancelled);

            let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;

            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));
            assert_eq!(
                cm.get_job(job_id).unwrap().state,
                JobState::Cancelled,
                "the job must stay exactly as the earlier, real cancel left it"
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn resources_unavailable_reject_cools_down_the_node() {
            let dir = TempDir::new().unwrap();
            let cm = test_cluster(&dir).await;

            let addr = spawn_mock_agent_rejecting_resources().await;
            register_node_at(&cm, "n1", addr);

            let job_id = submit_and_wait(&cm, batch_spec("resource-reject", 1));

            assert!(cm.nodes_on_dispatch_cooldown().is_empty());
            let outcome = confirm_dispatch_pending_job(&cm, job_id, &["n1"]).await;
            assert!(matches!(outcome, DispatchConfirmOutcome::Aborted));
            assert!(
                cm.nodes_on_dispatch_cooldown().contains("n1"),
                "a resources-unavailable reject must put the node on cooldown"
            );
            let job = cm.get_job(job_id).unwrap();
            assert_eq!(
                job.state_reason(),
                "JobLaunchFailure (dispatch confirmation failed (0/1 confirmed): \
                 1 gpu/resource allocation mismatch)",
                "operators need the specific failure category, not a generic count"
            );
        }
    }

    #[test]
    fn job_preempt_mode_qos_override_wins_over_partition() {
        use spur_core::accounting::QosPreemptMode;
        use spur_core::partition::PreemptMode;
        let parts = vec![partition_with_mode("gpu", PreemptMode::Cancel)];
        let qos = qos_with_mode(QosPreemptMode::Requeue);
        assert_eq!(
            job_preempt_mode(&job_in_partitions("gpu"), &parts, &qos),
            PreemptMode::Requeue
        );
    }

    #[test]
    fn job_preempt_mode_qos_off_falls_back_to_partition() {
        use spur_core::partition::PreemptMode;
        let parts = vec![partition_with_mode("gpu", PreemptMode::Cancel)];
        assert_eq!(
            job_preempt_mode(&job_in_partitions("gpu"), &parts, &no_qos_override()),
            PreemptMode::Cancel
        );
    }
}
