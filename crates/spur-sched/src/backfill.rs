use chrono::{Duration, Utc};
use tracing::debug;

use spur_core::job::Job;
use spur_core::node::Node;
use spur_core::resource::ResourceSet;

use crate::timeline::NodeTimeline;
use crate::traits::{Assignment, ClusterState, Scheduler};

/// Backfill scheduler.
///
/// Algorithm:
/// 1. Sort pending jobs by effective priority (highest first).
/// 2. For each top-priority job, compute its earliest start time across
///    all suitable nodes (creating "shadow reservations").
/// 3. For lower-priority jobs, check if they can fit in the gaps
///    without delaying any shadow reservation.
pub struct BackfillScheduler {
    /// Per-node timelines for resource tracking.
    timelines: Vec<NodeTimeline>,
    /// Max jobs to consider per cycle.
    max_jobs: usize,
}

impl BackfillScheduler {
    pub fn new(max_jobs: usize) -> Self {
        Self {
            timelines: Vec::new(),
            max_jobs,
        }
    }

    /// Initialize or reset timelines from current cluster state.
    fn init_timelines(&mut self, nodes: &[Node]) {
        self.timelines = nodes
            .iter()
            .map(|n| NodeTimeline::new(n.name.clone(), n.total_resources.clone()))
            .collect();
    }

    /// Find nodes that satisfy a job's resource requirements.
    fn find_suitable_nodes(
        &self,
        job: &Job,
        nodes: &[Node],
        partitions: &[spur_core::partition::Partition],
    ) -> Vec<usize> {
        let partition_name = job.spec.partition.as_deref();
        let required = job_resource_request(job);

        // Parse nodelist / exclude constraints once, outside the per-node loop.
        let nodelist: Option<Vec<&str>> = job
            .spec
            .nodelist
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(|s| s.split(',').map(str::trim).collect());

        let exclude: Vec<&str> = job
            .spec
            .exclude
            .as_deref()
            .map(|s| s.split(',').map(str::trim).collect())
            .unwrap_or_default();

        nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                // Honour --nodelist: node must be in the explicit allow-list.
                if let Some(ref allowed) = nodelist {
                    if !allowed.contains(&node.name.as_str()) {
                        return false;
                    }
                }
                // Honour --exclude: node must not be in the deny-list.
                if exclude.contains(&node.name.as_str()) {
                    return false;
                }
                // Check partition membership
                if let Some(pname) = partition_name {
                    if !node.partitions.contains(&pname.to_string()) {
                        return false;
                    }
                }
                // Check node is schedulable
                if !node.is_schedulable() {
                    return false;
                }
                // Exclusive job needs an idle node (no current allocations)
                if job.spec.exclusive
                    && (node.alloc_resources.cpus > 0 || !node.alloc_resources.gpus.is_empty())
                {
                    return false;
                }
                // Skip nodes fully consumed by an exclusive job
                if node.alloc_resources.cpus >= node.total_resources.cpus
                    && node.total_resources.cpus > 0
                {
                    return false;
                }
                // Check resource capacity (total, not current available)
                node.total_resources.can_satisfy(&required)
            })
            .map(|(i, _)| i)
            .collect()
    }
}

impl Scheduler for BackfillScheduler {
    fn schedule(&mut self, pending: &[Job], cluster: &ClusterState) -> Vec<Assignment> {
        let now = Utc::now();
        self.init_timelines(cluster.nodes);

        // Add current allocations to timelines
        for (i, node) in cluster.nodes.iter().enumerate() {
            if node.alloc_resources.cpus > 0 || !node.alloc_resources.gpus.is_empty() {
                // Existing allocations — we don't know their end time,
                // so we use a conservative estimate. In practice, running jobs
                // are tracked with actual time limits.
                self.timelines[i].reserve(
                    now,
                    now + Duration::hours(24), // Conservative
                    node.alloc_resources.clone(),
                );
            }
        }

        let mut assignments = Vec::new();
        let limit = pending.len().min(self.max_jobs);

        // Shadow reservations for top-priority jobs
        let mut shadows: Vec<(usize, chrono::DateTime<Utc>)> = Vec::new(); // (job_idx, earliest_start)

        for (job_idx, job) in pending.iter().enumerate().take(limit) {
            let suitable = self.find_suitable_nodes(job, cluster.nodes, cluster.partitions);
            if suitable.is_empty() {
                continue;
            }

            let required = job_resource_request(job);
            let duration = job.spec.time_limit.unwrap_or(Duration::hours(1));
            let needed_nodes = job.spec.num_nodes as usize;

            // Find earliest start across needed_nodes
            let mut node_starts: Vec<(usize, chrono::DateTime<Utc>)> = suitable
                .iter()
                .map(|&ni| {
                    let start = self.timelines[ni].earliest_start(&required, duration, now);
                    (ni, start)
                })
                .collect();

            node_starts.sort_by_key(|(_, t)| *t);

            if node_starts.len() < needed_nodes {
                continue;
            }

            let assigned_nodes: Vec<(usize, chrono::DateTime<Utc>)> =
                node_starts.into_iter().take(needed_nodes).collect();

            let earliest = assigned_nodes.iter().map(|(_, t)| *t).max().unwrap();

            if earliest <= now {
                // Can start immediately
                let node_names: Vec<String> = assigned_nodes
                    .iter()
                    .map(|(ni, _)| cluster.nodes[*ni].name.clone())
                    .collect();

                // Reserve on timelines
                for (ni, _) in &assigned_nodes {
                    self.timelines[*ni].reserve(now, now + duration, required.clone());
                }

                debug!(
                    job_id = job.job_id,
                    nodes = ?node_names,
                    "scheduling job"
                );

                assignments.push(Assignment {
                    job_id: job.job_id,
                    nodes: node_names,
                });
            } else {
                // Create shadow reservation (blocks lower-priority backfill)
                for (ni, _) in &assigned_nodes {
                    self.timelines[*ni].reserve(earliest, earliest + duration, required.clone());
                }
                shadows.push((job_idx, earliest));
            }
        }

        assignments
    }

    fn name(&self) -> &str {
        "backfill"
    }
}

/// Extract the per-node resource request from a job spec.
pub fn job_resource_request(job: &Job) -> ResourceSet {
    let cpus = if job.spec.tasks_per_node.is_some() {
        job.spec.tasks_per_node.unwrap_or(1) * job.spec.cpus_per_task
    } else {
        (job.spec.num_tasks / job.spec.num_nodes.max(1)) * job.spec.cpus_per_task
    };

    let memory = job
        .spec
        .memory_per_node_mb
        .or_else(|| job.spec.memory_per_cpu_mb.map(|m| m * cpus as u64))
        .unwrap_or(0);

    // Parse GRES into GPU count for scheduling
    let mut gpu_count = 0u32;
    let mut gpu_type = String::new();
    for gres in &job.spec.gres {
        if let Some((name, gtype, count)) = spur_core::resource::parse_gres(gres) {
            if name == "gpu" {
                gpu_count += count;
                if let Some(t) = gtype {
                    gpu_type = t;
                }
            }
        }
    }

    // Create placeholder GPU resources for matching
    let gpus: Vec<spur_core::resource::GpuResource> = (0..gpu_count)
        .map(|i| spur_core::resource::GpuResource {
            device_id: i,
            gpu_type: if gpu_type.is_empty() {
                "any".into()
            } else {
                gpu_type.clone()
            },
            memory_mb: 0,
            peer_gpus: Vec::new(),
            link_type: spur_core::resource::GpuLinkType::PCIe,
        })
        .collect();

    ResourceSet {
        cpus,
        memory_mb: memory,
        gpus,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spur_core::job::JobSpec;
    use spur_core::node::NodeState;
    use spur_core::partition::Partition;

    fn make_nodes(count: usize) -> Vec<Node> {
        (0..count)
            .map(|i| {
                let mut node = Node::new(
                    format!("node{:03}", i + 1),
                    ResourceSet {
                        cpus: 64,
                        memory_mb: 256_000,
                        ..Default::default()
                    },
                );
                node.state = NodeState::Idle;
                node.partitions = vec!["default".into()];
                node
            })
            .collect()
    }

    fn make_job(id: u32, nodes: u32, cpus: u32) -> Job {
        Job::new(
            id,
            JobSpec {
                name: format!("job{}", id),
                partition: Some("default".into()),
                user: "test".into(),
                num_nodes: nodes,
                num_tasks: nodes * cpus,
                cpus_per_task: 1,
                time_limit: Some(Duration::hours(1)),
                ..Default::default()
            },
        )
    }

    #[test]
    fn test_schedule_single_job() {
        let mut sched = BackfillScheduler::new(100);
        let nodes = make_nodes(4);
        let partitions = vec![Partition {
            name: "default".into(),
            ..Default::default()
        }];

        let pending = vec![make_job(1, 2, 32)];
        let cluster = ClusterState {
            nodes: &nodes,
            partitions: &partitions,
        };

        let assignments = sched.schedule(&pending, &cluster);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].job_id, 1);
        assert_eq!(assignments[0].nodes.len(), 2);
    }

    #[test]
    fn test_schedule_multiple_jobs() {
        let mut sched = BackfillScheduler::new(100);
        let nodes = make_nodes(4);
        let partitions = vec![Partition {
            name: "default".into(),
            ..Default::default()
        }];

        let pending = vec![make_job(1, 2, 32), make_job(2, 2, 32)];
        let cluster = ClusterState {
            nodes: &nodes,
            partitions: &partitions,
        };

        let assignments = sched.schedule(&pending, &cluster);
        assert_eq!(assignments.len(), 2);
    }

    #[test]
    fn test_insufficient_resources() {
        let mut sched = BackfillScheduler::new(100);
        let nodes = make_nodes(2);
        let partitions = vec![Partition {
            name: "default".into(),
            ..Default::default()
        }];

        // Request 4 nodes but only 2 available
        let pending = vec![make_job(1, 4, 32)];
        let cluster = ClusterState {
            nodes: &nodes,
            partitions: &partitions,
        };

        let assignments = sched.schedule(&pending, &cluster);
        assert_eq!(assignments.len(), 0);
    }

    fn make_job_with_nodelist(
        id: u32,
        nodes: u32,
        nodelist: Option<&str>,
        exclude: Option<&str>,
    ) -> Job {
        let mut spec = JobSpec {
            name: format!("job{}", id),
            partition: Some("default".into()),
            user: "test".into(),
            num_nodes: nodes,
            num_tasks: nodes,
            cpus_per_task: 1,
            time_limit: Some(Duration::hours(1)),
            ..Default::default()
        };
        if let Some(nl) = nodelist {
            spec.nodelist = Some(nl.into());
        }
        if let Some(ex) = exclude {
            spec.exclude = Some(ex.into());
        }
        Job::new(id, spec)
    }

    #[test]
    fn test_nodelist_pins_to_allowed_nodes() {
        let mut sched = BackfillScheduler::new(100);
        let nodes = make_nodes(4); // node001, node002, node003, node004
        let partitions = vec![Partition {
            name: "default".into(),
            ..Default::default()
        }];

        let pending = vec![make_job_with_nodelist(1, 1, Some("node001"), None)];
        let cluster = ClusterState {
            nodes: &nodes,
            partitions: &partitions,
        };

        let assignments = sched.schedule(&pending, &cluster);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].nodes, vec!["node001"]);
    }

    #[test]
    fn test_nodelist_multi_node() {
        let mut sched = BackfillScheduler::new(100);
        let nodes = make_nodes(4);
        let partitions = vec![Partition {
            name: "default".into(),
            ..Default::default()
        }];

        let pending = vec![make_job_with_nodelist(1, 2, Some("node001,node002"), None)];
        let cluster = ClusterState {
            nodes: &nodes,
            partitions: &partitions,
        };

        let assignments = sched.schedule(&pending, &cluster);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].nodes.len(), 2);
        assert!(assignments[0].nodes.contains(&"node001".to_string()));
        assert!(assignments[0].nodes.contains(&"node002".to_string()));
    }

    #[test]
    fn test_nodelist_no_match_returns_empty() {
        let mut sched = BackfillScheduler::new(100);
        let nodes = make_nodes(4);
        let partitions = vec![Partition {
            name: "default".into(),
            ..Default::default()
        }];

        // "nodeXXX" does not exist in the cluster
        let pending = vec![make_job_with_nodelist(1, 1, Some("nodeXXX"), None)];
        let cluster = ClusterState {
            nodes: &nodes,
            partitions: &partitions,
        };

        let assignments = sched.schedule(&pending, &cluster);
        assert_eq!(assignments.len(), 0);
    }

    #[test]
    fn test_exclude_removes_nodes() {
        let mut sched = BackfillScheduler::new(100);
        let nodes = make_nodes(4);
        let partitions = vec![Partition {
            name: "default".into(),
            ..Default::default()
        }];

        // Exclude node001 and node002 → only node003 and node004 remain
        let pending = vec![make_job_with_nodelist(1, 2, None, Some("node001,node002"))];
        let cluster = ClusterState {
            nodes: &nodes,
            partitions: &partitions,
        };

        let assignments = sched.schedule(&pending, &cluster);
        assert_eq!(assignments.len(), 1);
        assert!(!assignments[0].nodes.contains(&"node001".to_string()));
        assert!(!assignments[0].nodes.contains(&"node002".to_string()));
    }

    #[test]
    fn test_exclude_too_many_leaves_unschedulable() {
        let mut sched = BackfillScheduler::new(100);
        let nodes = make_nodes(2); // node001, node002
        let partitions = vec![Partition {
            name: "default".into(),
            ..Default::default()
        }];

        // Exclude both → 2-node job can't schedule
        let pending = vec![make_job_with_nodelist(1, 2, None, Some("node001,node002"))];
        let cluster = ClusterState {
            nodes: &nodes,
            partitions: &partitions,
        };

        let assignments = sched.schedule(&pending, &cluster);
        assert_eq!(assignments.len(), 0);
    }
}
