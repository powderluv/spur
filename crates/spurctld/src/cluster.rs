use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

use chrono::Utc;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use spur_core::accounting::{Qos, TresRecord, TresType};
use spur_core::config::SlurmConfig;
use spur_core::job::{Job, JobId, JobSpec, JobState, PendingReason};
use spur_core::node::{Node, NodeSource, NodeState};
use spur_core::partition::Partition;
use spur_core::qos::{check_qos_limits, QosCheckResult};
use spur_core::resource::ResourceSet;
use spur_core::wal::{WalEntry, WalOperation, WalStore};
use spur_state::snapshot::SnapshotStore;
use spur_state::wal_store::FileWalStore;

/// Central cluster state manager.
///
/// Thread-safe via RwLock. The scheduler and gRPC server both access this.
pub struct ClusterManager {
    pub config: SlurmConfig,
    jobs: RwLock<HashMap<JobId, Job>>,
    nodes: RwLock<HashMap<String, Node>>,
    partitions: RwLock<Vec<Partition>>,
    next_job_id: AtomicU32,
    wal: RwLock<FileWalStore>,
    snapshot: RwLock<Option<SnapshotStore>>,
    wal_since_snapshot: AtomicU32,
}

impl ClusterManager {
    pub fn new(config: SlurmConfig, state_dir: &Path) -> anyhow::Result<Self> {
        let wal = FileWalStore::new(&state_dir.join("wal"))?;
        let snapshot = if state_dir.join("snapshot.redb").exists() {
            Some(SnapshotStore::open(&state_dir.join("snapshot.redb"))?)
        } else {
            std::fs::create_dir_all(state_dir)?;
            Some(SnapshotStore::open(&state_dir.join("snapshot.redb"))?)
        };

        let partitions = config.build_partitions();

        // Recover state from snapshot + WAL replay
        let mut jobs = HashMap::new();
        let mut nodes = HashMap::new();
        let mut next_id = config.controller.first_job_id;

        if let Some(ref snap) = snapshot {
            let snap_seq = snap.wal_sequence().unwrap_or(0);
            for job in snap.load_jobs().unwrap_or_default() {
                next_id = next_id.max(job.job_id + 1);
                jobs.insert(job.job_id, job);
            }
            for node in snap.load_nodes().unwrap_or_default() {
                nodes.insert(node.name.clone(), node);
            }

            // Replay WAL entries after snapshot
            let wal_entries = wal.read_from(snap_seq + 1).unwrap_or_default();
            if !wal_entries.is_empty() {
                info!(
                    count = wal_entries.len(),
                    from_seq = snap_seq + 1,
                    "replaying WAL entries"
                );
                for entry in &wal_entries {
                    replay_entry(entry, &mut jobs, &mut nodes, &mut next_id);
                }
            }
        }

        info!(
            jobs = jobs.len(),
            nodes = nodes.len(),
            next_job_id = next_id,
            "cluster state recovered"
        );

        Ok(Self {
            config,
            jobs: RwLock::new(jobs),
            nodes: RwLock::new(nodes),
            partitions: RwLock::new(partitions),
            next_job_id: AtomicU32::new(next_id),
            wal: RwLock::new(wal),
            snapshot: RwLock::new(snapshot),
            wal_since_snapshot: AtomicU32::new(0),
        })
    }

    /// Submit a new job. If it has an array spec, expand into individual tasks.
    pub fn submit_job(&self, spec: JobSpec) -> anyhow::Result<JobId> {
        // Validate partition constraints before accepting the job
        self.validate_partition(&spec)?;

        // Check for array job
        if let Some(ref array_spec) = spec.array_spec {
            return self.submit_array_job(spec.clone(), array_spec);
        }

        let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);
        let job = Job::new(job_id, spec.clone());

        // WAL
        self.append_wal(WalOperation::JobSubmit {
            job_id,
            spec: spec.clone(),
        });

        self.jobs.write().insert(job_id, job);

        info!(job_id, name = %spec.name, user = %spec.user, "job submitted");
        self.maybe_snapshot();
        Ok(job_id)
    }

    /// Submit an array job — expands the spec into individual task jobs.
    fn submit_array_job(&self, spec: JobSpec, array_spec_str: &str) -> anyhow::Result<JobId> {
        use spur_core::array::parse_array_spec;

        let array = parse_array_spec(array_spec_str)
            .map_err(|e| anyhow::anyhow!("invalid array spec: {}", e))?;

        let array_job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);
        let mut jobs = self.jobs.write();

        info!(
            array_job_id,
            tasks = array.task_ids.len(),
            max_concurrent = array.max_concurrent,
            name = %spec.name,
            "array job submitted"
        );

        for &task_id in &array.task_ids {
            let task_job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);
            let mut task_spec = spec.clone();
            task_spec.array_spec = None; // Don't recurse

            self.append_wal(WalOperation::JobSubmit {
                job_id: task_job_id,
                spec: task_spec.clone(),
            });

            let mut job = Job::new(task_job_id, task_spec);
            job.array_job_id = Some(array_job_id);
            job.array_task_id = Some(task_id);
            if array.max_concurrent > 0 {
                job.array_max_concurrent = Some(array.max_concurrent);
            }

            jobs.insert(task_job_id, job);
        }

        // Return the array job ID (first task IDs are array_job_id + 1, +2, ...)
        Ok(array_job_id)
    }

    /// Validate partition constraints: access control and node limits.
    fn validate_partition(&self, spec: &JobSpec) -> anyhow::Result<()> {
        let partition_name = match spec.partition.as_ref() {
            Some(p) if !p.is_empty() => p,
            _ => return Ok(()), // No partition specified — default, no restrictions
        };

        let partitions = self.partitions.read();
        let part = match partitions.iter().find(|p| p.name == *partition_name) {
            Some(p) => p,
            None => anyhow::bail!("partition '{}' not found", partition_name),
        };

        // Check partition state
        if part.state != spur_core::partition::PartitionState::Up {
            anyhow::bail!("partition '{}' is {}", partition_name, part.state.display());
        }

        // Check allow_accounts (if non-empty, user's account must be in the list)
        if !part.allow_accounts.is_empty() {
            let account = spec.account.as_deref().unwrap_or("");
            if !part.allow_accounts.iter().any(|a| a == account) {
                anyhow::bail!(
                    "account '{}' not allowed on partition '{}'",
                    account,
                    partition_name
                );
            }
        }

        // Check deny_accounts
        if let Some(ref account) = spec.account {
            if part.deny_accounts.iter().any(|a| a == account) {
                anyhow::bail!(
                    "account '{}' denied on partition '{}'",
                    account,
                    partition_name
                );
            }
        }

        // Check max_nodes
        if let Some(max) = part.max_nodes {
            if spec.num_nodes > max {
                anyhow::bail!(
                    "requested {} nodes exceeds partition '{}' max of {}",
                    spec.num_nodes,
                    partition_name,
                    max
                );
            }
        }

        // Check max_time
        if let (Some(max_mins), Some(ref tl)) = (part.max_time_minutes, &spec.time_limit) {
            let requested_mins = tl.num_minutes() as u32;
            if requested_mins > max_mins {
                anyhow::bail!(
                    "requested time {} min exceeds partition '{}' max of {} min",
                    requested_mins,
                    partition_name,
                    max_mins
                );
            }
        }

        Ok(())
    }

    /// Get a job by ID.
    pub fn get_job(&self, job_id: JobId) -> Option<Job> {
        self.jobs.read().get(&job_id).cloned()
    }

    /// Get jobs matching filters.
    pub fn get_jobs(
        &self,
        states: &[JobState],
        user: Option<&str>,
        partition: Option<&str>,
        account: Option<&str>,
        job_ids: &[JobId],
    ) -> Vec<Job> {
        let jobs = self.jobs.read();
        jobs.values()
            .filter(|j| {
                if !states.is_empty() && !states.contains(&j.state) {
                    return false;
                }
                if let Some(u) = user {
                    if !u.is_empty() && j.spec.user != u {
                        return false;
                    }
                }
                if let Some(p) = partition {
                    if !p.is_empty() && j.spec.partition.as_deref() != Some(p) {
                        return false;
                    }
                }
                if let Some(a) = account {
                    if !a.is_empty() && j.spec.account.as_deref() != Some(a) {
                        return false;
                    }
                }
                if !job_ids.is_empty() && !job_ids.contains(&j.job_id) {
                    return false;
                }
                true
            })
            .cloned()
            .collect()
    }

    /// Cancel a job.
    pub fn cancel_job(&self, job_id: JobId, _user: &str) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write();
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;

        let old_state = job.state;
        job.transition(JobState::Cancelled)?;

        self.append_wal(WalOperation::JobStateChange {
            job_id,
            old_state,
            new_state: JobState::Cancelled,
        });

        info!(job_id, "job cancelled");
        Ok(())
    }

    /// Start a job on specific nodes.
    pub fn start_job(
        &self,
        job_id: JobId,
        node_names: Vec<String>,
        resources: ResourceSet,
    ) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write();
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;

        let old_state = job.state;
        let exclusive = job.spec.exclusive;
        job.transition(JobState::Running)?;
        job.start_time = Some(Utc::now());
        job.allocated_nodes = node_names.clone();
        job.allocated_resources = Some(resources.clone());
        job.pending_reason = PendingReason::None;

        // Compute per-node resource share
        let node_count = node_names.len().max(1) as u32;
        let per_node = ResourceSet {
            cpus: resources.cpus / node_count,
            memory_mb: resources.memory_mb / node_count as u64,
            gpus: resources.gpus.clone(), // GPUs are already per-node in the request
            generic: resources
                .generic
                .iter()
                .map(|(k, v)| (k.clone(), v / node_count as u64))
                .collect(),
        };

        // Drop jobs lock before acquiring nodes lock (lock ordering: jobs → nodes)
        drop(jobs);

        self.append_wal(WalOperation::JobStateChange {
            job_id,
            old_state,
            new_state: JobState::Running,
        });
        self.append_wal(WalOperation::JobStart {
            job_id,
            nodes: node_names.clone(),
            resources,
        });

        // Update node alloc_resources
        let mut nodes = self.nodes.write();
        for name in &node_names {
            if let Some(node) = nodes.get_mut(name) {
                if exclusive {
                    // Exclusive: claim entire node so no other jobs can co-schedule
                    node.alloc_resources = node.total_resources.clone();
                } else {
                    node.alloc_resources = node.alloc_resources.add(&per_node);
                }
                node.update_state_from_alloc();
            }
        }

        debug!(job_id, "job started");
        Ok(())
    }

    /// Complete a job.
    pub fn complete_job(
        &self,
        job_id: JobId,
        exit_code: i32,
        state: JobState,
    ) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write();
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;

        job.transition(state)?;
        job.exit_code = Some(exit_code);

        // Capture allocation info before dropping jobs lock
        let freed_nodes = job.allocated_nodes.clone();
        let allocated_resources = job.allocated_resources.clone();
        drop(jobs);

        self.append_wal(WalOperation::JobComplete {
            job_id,
            exit_code,
            state,
        });

        // Subtract per-node resources instead of blanket-zeroing
        let mut nodes = self.nodes.write();
        if let Some(ref total_resources) = allocated_resources {
            let node_count = freed_nodes.len().max(1) as u32;
            let per_node = ResourceSet {
                cpus: total_resources.cpus / node_count,
                memory_mb: total_resources.memory_mb / node_count as u64,
                gpus: total_resources.gpus.clone(),
                generic: total_resources
                    .generic
                    .iter()
                    .map(|(k, v)| (k.clone(), v / node_count as u64))
                    .collect(),
            };
            for name in &freed_nodes {
                if let Some(node) = nodes.get_mut(name) {
                    node.alloc_resources = node.alloc_resources.subtract(&per_node);
                    node.update_state_from_alloc();

                    // Draining → Drain: if node was draining and all resources are now free,
                    // complete the drain transition.
                    if node.state == NodeState::Draining
                        && node.alloc_resources.cpus == 0
                        && node.alloc_resources.gpus.is_empty()
                    {
                        node.state = NodeState::Drain;
                        info!(node = %node.name, "draining node fully drained");
                    }
                }
            }
        } else {
            // Fallback: zero out (legacy jobs without tracked resources)
            for name in &freed_nodes {
                if let Some(node) = nodes.get_mut(name) {
                    node.alloc_resources = ResourceSet::default();
                    node.update_state_from_alloc();

                    // Draining → Drain on fallback path too
                    if node.state == NodeState::Draining {
                        node.state = NodeState::Drain;
                        info!(node = %node.name, "draining node fully drained");
                    }
                }
            }
        }

        debug!(job_id, exit_code, "job completed");

        // Check for requeue: if the job has --requeue and hit a retriable state,
        // reset it to Pending so the scheduler picks it up again.
        let should_requeue = matches!(
            state,
            JobState::Timeout | JobState::Preempted | JobState::NodeFail
        );
        if should_requeue {
            self.maybe_requeue(job_id);
        }

        self.maybe_snapshot();
        Ok(())
    }

    /// Requeue a job if spec.requeue is set and attempt limit not exceeded.
    fn maybe_requeue(&self, job_id: JobId) {
        const MAX_REQUEUE: u32 = 3;
        let mut jobs = self.jobs.write();
        let Some(job) = jobs.get_mut(&job_id) else {
            return;
        };
        if !job.spec.requeue || job.requeue_count >= MAX_REQUEUE {
            return;
        }

        let old_state = job.state;
        if job.transition(JobState::Pending).is_err() {
            return;
        }
        job.requeue_count += 1;
        job.start_time = None;
        job.exit_code = None;
        job.allocated_nodes.clear();
        job.allocated_resources = None;
        job.pending_reason = PendingReason::Priority;

        self.append_wal(WalOperation::JobStateChange {
            job_id,
            old_state,
            new_state: JobState::Pending,
        });

        info!(
            job_id,
            requeue_count = job.requeue_count,
            from = %old_state,
            "job requeued"
        );
    }

    /// Register a node agent.
    pub fn register_node(
        &self,
        name: String,
        resources: ResourceSet,
        address: String,
        port: u16,
        wg_pubkey: String,
        version: String,
        source: NodeSource,
    ) {
        let mut node = Node::new(name.clone(), resources.clone());
        node.state = NodeState::Idle;
        node.source = source;
        node.address = Some(address.clone());
        node.port = port;
        if !wg_pubkey.is_empty() {
            node.wg_pubkey = Some(wg_pubkey);
        }
        node.version = Some(version);
        node.agent_start_time = Some(Utc::now());
        node.last_heartbeat = Some(Utc::now());

        // Assign to partitions based on config
        let partitions = self.partitions.read();
        for part in partitions.iter() {
            if let Ok(hosts) = spur_core::hostlist::expand(&part.nodes) {
                if hosts.contains(&name) {
                    node.partitions.push(part.name.clone());
                }
            }
        }

        // Copy features from node config
        for nc in &self.config.nodes {
            if let Ok(hosts) = spur_core::hostlist::expand(&nc.names) {
                if hosts.contains(&name) {
                    node.features = nc.features.clone();
                    break;
                }
            }
        }

        self.append_wal(WalOperation::NodeRegister {
            name: name.clone(),
            resources,
            address,
        });

        info!(node = %name, partitions = ?node.partitions, "node registered");
        self.nodes.write().insert(name, node);
    }

    /// Update node heartbeat data.
    pub fn update_heartbeat(&self, name: &str, cpu_load: u32, free_memory_mb: u64) {
        let mut nodes = self.nodes.write();
        if let Some(node) = nodes.get_mut(name) {
            node.cpu_load = cpu_load;
            node.free_memory_mb = free_memory_mb;
            node.last_heartbeat = Some(Utc::now());
        }
    }

    /// Get all nodes.
    pub fn get_nodes(&self) -> Vec<Node> {
        self.nodes.read().values().cloned().collect()
    }

    /// Get a node by name.
    pub fn get_node(&self, name: &str) -> Option<Node> {
        self.nodes.read().get(name).cloned()
    }

    /// Get all partitions.
    pub fn get_partitions(&self) -> Vec<Partition> {
        self.partitions.read().clone()
    }

    /// Hold a job (prevent scheduling).
    pub fn hold_job(&self, job_id: JobId) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write();
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
        if job.state != JobState::Pending {
            anyhow::bail!(
                "can only hold pending jobs (job {} is {:?})",
                job_id,
                job.state
            );
        }
        job.pending_reason = PendingReason::Held;
        job.priority = 0;
        self.append_wal(WalOperation::JobPriorityChange {
            job_id,
            old_priority: job.priority,
            new_priority: 0,
        });
        info!(job_id, "job held");
        Ok(())
    }

    /// Release a held job.
    pub fn release_job(&self, job_id: JobId) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write();
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;
        if job.pending_reason != PendingReason::Held {
            anyhow::bail!("job {} is not held", job_id);
        }
        job.pending_reason = PendingReason::Priority;
        job.priority = 1000; // Restore default priority
        self.append_wal(WalOperation::JobPriorityChange {
            job_id,
            old_priority: 0,
            new_priority: 1000,
        });
        info!(job_id, "job released");
        Ok(())
    }

    /// Update job properties.
    pub fn update_job(
        &self,
        job_id: JobId,
        time_limit: Option<chrono::Duration>,
        priority: Option<u32>,
        partition: Option<String>,
        comment: Option<String>,
        account: Option<String>,
        qos: Option<String>,
    ) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write();
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {} not found", job_id))?;

        if let Some(tl) = time_limit {
            job.spec.time_limit = Some(tl);
        }
        if let Some(p) = priority {
            let old = job.priority;
            job.priority = p;
            self.append_wal(WalOperation::JobPriorityChange {
                job_id,
                old_priority: old,
                new_priority: p,
            });
        }
        if let Some(part) = partition {
            job.spec.partition = Some(part);
        }
        if let Some(c) = comment {
            job.spec.comment = Some(c);
        }
        if let Some(a) = account {
            job.spec.account = Some(a);
        }
        if let Some(q) = qos {
            job.spec.qos = Some(q);
        }
        info!(job_id, "job updated");
        Ok(())
    }

    /// Update node state (admin: drain, resume, etc.)
    ///
    /// When draining a node that still has running jobs, the state is set to
    /// `Draining` instead of `Drain`. Once all jobs complete (tracked in
    /// `complete_job`), the node transitions to `Drain`.
    pub fn update_node_state(
        &self,
        name: &str,
        state: NodeState,
        reason: Option<String>,
    ) -> anyhow::Result<()> {
        let mut nodes = self.nodes.write();
        let node = nodes
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("node {} not found", name))?;
        let old_state = node.state;

        // If admin requests Drain but node has running jobs, set Draining instead
        let effective_state = if state == NodeState::Drain
            && (node.alloc_resources.cpus > 0 || !node.alloc_resources.gpus.is_empty())
        {
            NodeState::Draining
        } else {
            state
        };

        node.state = effective_state;
        node.state_reason = reason.clone();
        self.append_wal(WalOperation::NodeStateChange {
            name: name.to_string(),
            old_state,
            new_state: effective_state,
            reason,
        });
        info!(node = %name, old = ?old_state, new = ?effective_state, "node state updated");
        Ok(())
    }

    /// Check for stale nodes (no heartbeat within timeout) and mark them DOWN.
    pub fn check_node_health(&self, timeout_secs: u64) {
        let now = Utc::now();
        let threshold = chrono::Duration::seconds(timeout_secs as i64);
        let mut nodes = self.nodes.write();

        for node in nodes.values_mut() {
            if node.state == NodeState::Down || node.state == NodeState::Drain {
                continue;
            }
            if let Some(last_hb) = node.last_heartbeat {
                if now - last_hb > threshold {
                    let old_state = node.state;
                    node.state = NodeState::Down;
                    node.state_reason = Some("Not responding".into());
                    warn!(
                        node = %node.name,
                        last_heartbeat = %last_hb,
                        "node marked DOWN (heartbeat timeout)"
                    );
                    self.append_wal(WalOperation::NodeStateChange {
                        name: node.name.clone(),
                        old_state,
                        new_state: NodeState::Down,
                        reason: Some("Not responding".into()),
                    });
                }
            }
        }
    }

    /// Get pending jobs sorted by priority, filtering out held and dependency-blocked jobs.
    /// Recomputes effective priority using age and partition tier before sorting.
    pub fn pending_jobs(&self) -> Vec<Job> {
        let jobs = self.jobs.read();
        let mut pending: Vec<Job> = jobs
            .values()
            .filter(|j| j.state == JobState::Pending && j.pending_reason != PendingReason::Held)
            .cloned()
            .collect();

        // Check dependencies
        let get_job = |id: JobId| -> Option<Job> { jobs.get(&id).cloned() };
        let get_jobs_by_name_user = |name: &str, user: &str| -> Vec<Job> {
            jobs.values()
                .filter(|j| j.spec.name == name && j.spec.user == user)
                .cloned()
                .collect()
        };

        pending.retain(|job| {
            if job.spec.dependency.is_empty() {
                return true;
            }
            use spur_core::dependency::{check_dependencies, DependencyResult};
            match check_dependencies(job, &get_job, &get_jobs_by_name_user) {
                DependencyResult::Satisfied => true,
                DependencyResult::Waiting => false,
                DependencyResult::Failed => {
                    // Dependency can never be satisfied — cancel the job
                    // (can't mutate here since we hold a read lock; mark for later)
                    false
                }
            }
        });

        // Enforce array max_concurrent: suppress tasks if too many siblings already running
        let running_array_counts: std::collections::HashMap<JobId, u32> = {
            let mut counts = std::collections::HashMap::new();
            for j in jobs.values() {
                if j.state == JobState::Running {
                    if let Some(aid) = j.array_job_id {
                        *counts.entry(aid).or_insert(0) += 1;
                    }
                }
            }
            counts
        };
        pending.retain(|job| {
            if let (Some(aid), Some(max)) = (job.array_job_id, job.array_max_concurrent) {
                let running = running_array_counts.get(&aid).copied().unwrap_or(0);
                if running >= max {
                    return false; // Throttled — too many siblings running
                }
            }
            true
        });

        // QoS enforcement: check per-user limits for jobs with a QoS
        pending.retain(|job| {
            if job.spec.qos.is_none() {
                return true; // No QoS — skip check
            }

            let user = &job.spec.user;

            let running_count = jobs
                .values()
                .filter(|j| j.state == JobState::Running && j.spec.user == *user)
                .count() as u32;

            let submitted_count = jobs
                .values()
                .filter(|j| {
                    (j.state == JobState::Pending || j.state == JobState::Running)
                        && j.spec.user == *user
                })
                .count() as u32;

            // Compute running TRES for this user (total CPUs from running jobs)
            let mut running_tres = TresRecord::new();
            let running_cpus: u64 = jobs
                .values()
                .filter(|j| j.state == JobState::Running && j.spec.user == *user)
                .map(|j| (j.spec.num_tasks * j.spec.cpus_per_task) as u64)
                .sum();
            running_tres.set(TresType::Cpu, running_cpus);

            // Use a default QoS (no limits) — real QoS definitions would come
            // from the accounting database; for now this wires the enforcement
            // path so it's ready when QoS configs are populated.
            let qos = Qos::default();

            match check_qos_limits(job, &qos, running_count, submitted_count, &running_tres) {
                QosCheckResult::Allowed => true,
                QosCheckResult::Blocked(_reason) => false,
            }
        });

        // Recompute effective priority with age + partition tier
        let now = Utc::now();
        let partitions = self.partitions.read();
        for job in &mut pending {
            let age_minutes = (now - job.submit_time).num_minutes().max(0);
            let partition_tier = job
                .spec
                .partition
                .as_ref()
                .and_then(|pname| partitions.iter().find(|p| p.name == *pname))
                .map(|p| p.priority_tier)
                .unwrap_or(1);
            // fair_share = 1.0 (neutral) until spurdbd integration
            job.priority = spur_sched::priority::effective_priority(
                job.priority,
                1.0,
                age_minutes,
                partition_tier,
            );
        }

        pending.sort_by(|a, b| b.priority.cmp(&a.priority));
        pending
    }

    /// Find jobs by name and user (for singleton dependency).
    pub fn get_jobs_by_name_user(&self, name: &str, user: &str) -> Vec<Job> {
        self.jobs
            .read()
            .values()
            .filter(|j| j.spec.name == name && j.spec.user == user)
            .cloned()
            .collect()
    }

    fn append_wal(&self, op: WalOperation) {
        let mut wal = self.wal.write();
        let seq = wal.latest_sequence() + 1;
        let entry = WalEntry::new(seq, op);
        if let Err(e) = wal.append(&entry) {
            warn!(error = %e, "failed to append WAL entry");
        }
        self.wal_since_snapshot.fetch_add(1, Ordering::Relaxed);
    }

    fn maybe_snapshot(&self) {
        let count = self.wal_since_snapshot.load(Ordering::Relaxed);
        if count >= 10_000 {
            self.take_snapshot();
        }
    }

    pub fn take_snapshot(&self) {
        let snap = self.snapshot.read();
        if let Some(ref store) = *snap {
            let jobs: Vec<Job> = self.jobs.read().values().cloned().collect();
            let nodes: Vec<Node> = self.nodes.read().values().cloned().collect();
            let seq = self.wal.read().latest_sequence();

            match store.save(&jobs, &nodes, seq) {
                Ok(()) => {
                    self.wal_since_snapshot.store(0, Ordering::Relaxed);
                    debug!(wal_sequence = seq, "snapshot taken");
                }
                Err(e) => warn!(error = %e, "failed to take snapshot"),
            }
        }
    }
}

/// Replay a single WAL entry into state.
fn replay_entry(
    entry: &WalEntry,
    jobs: &mut HashMap<JobId, Job>,
    nodes: &mut HashMap<String, Node>,
    next_id: &mut u32,
) {
    match &entry.operation {
        WalOperation::JobSubmit { job_id, spec } => {
            jobs.insert(*job_id, Job::new(*job_id, spec.clone()));
            *next_id = (*next_id).max(job_id + 1);
        }
        WalOperation::JobStateChange {
            job_id, new_state, ..
        } => {
            if let Some(job) = jobs.get_mut(job_id) {
                let _ = job.transition(*new_state);
            }
        }
        WalOperation::JobStart {
            job_id,
            nodes: node_names,
            resources,
        } => {
            if let Some(job) = jobs.get_mut(job_id) {
                job.start_time = Some(entry.timestamp);
                job.allocated_nodes = node_names.clone();
                job.allocated_resources = Some(resources.clone());
            }
        }
        WalOperation::JobComplete {
            job_id, exit_code, ..
        } => {
            if let Some(job) = jobs.get_mut(job_id) {
                job.exit_code = Some(*exit_code);
                job.end_time = Some(entry.timestamp);
            }
        }
        WalOperation::JobPriorityChange {
            job_id,
            new_priority,
            ..
        } => {
            if let Some(job) = jobs.get_mut(job_id) {
                job.priority = *new_priority;
            }
        }
        WalOperation::NodeRegister {
            name,
            resources,
            address,
        } => {
            let mut node = Node::new(name.clone(), resources.clone());
            node.address = Some(address.clone());
            node.state = NodeState::Idle;
            nodes.insert(name.clone(), node);
        }
        WalOperation::NodeStateChange {
            name,
            new_state,
            reason,
            ..
        } => {
            if let Some(node) = nodes.get_mut(name) {
                node.state = *new_state;
                node.state_reason = reason.clone();
            }
        }
        WalOperation::NodeHeartbeat {
            name,
            cpu_load,
            free_memory_mb,
        } => {
            if let Some(node) = nodes.get_mut(name) {
                node.cpu_load = *cpu_load;
                node.free_memory_mb = *free_memory_mb;
            }
        }
    }
}
