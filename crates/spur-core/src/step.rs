// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Job steps — sub-executions within a running job.
//!
//! When `srun` is called inside a batch script, it creates a job step.
//! Steps track their own resource allocation, task distribution, and status.

use std::convert::Infallible;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::job::JobId;
use crate::resource::ResourceAllocations;

/// Job step identifier.
pub type StepId = u32;

/// Special step IDs (matching Slurm conventions).
pub const STEP_BATCH: StepId = 0xFFFF_FFFE;
pub const STEP_EXTERN: StepId = 0xFFFF_FFFD;
pub const STEP_INTERACTIVE: StepId = 0xFFFF_FFFC;

/// Step IDs at or above this are reserved (batch/extern/interactive); real
/// user `srun` steps are numbered below it (0, 1, 2, ...).
pub const STEP_RESERVED_MIN: StepId = 0xFFFF_FFF0;

/// Default for a step id deserialized from a payload written before this
/// field existed — the batch step, since that was the only kind in flight.
pub fn default_step_id() -> StepId {
    STEP_BATCH
}

/// A step within a job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStep {
    pub job_id: JobId,
    pub step_id: StepId,
    pub name: String,
    pub state: StepState,

    pub num_tasks: u32,
    pub cpus_per_task: u32,
    pub resources: ResourceAllocations,
    pub nodes: Vec<String>,

    /// Task distribution across nodes.
    pub distribution: TaskDistribution,

    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl StepState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    pub fn display(&self) -> &'static str {
        match self {
            Self::Pending => "PENDING",
            Self::Running => "RUNNING",
            Self::Completed => "COMPLETED",
            Self::Failed => "FAILED",
            Self::Cancelled => "CANCELLED",
        }
    }
}

/// How tasks are distributed across nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TaskDistribution {
    /// Round-robin across nodes (default for most workloads).
    #[default]
    Block,
    /// Cyclic distribution (task 0→node0, task 1→node1, task 2→node0, ...).
    Cyclic,
    /// Fill each node before moving to next.
    Plane,
    /// Arbitrary (explicit mapping).
    Arbitrary,
}

impl FromStr for TaskDistribution {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Infallible> {
        Ok(match s.to_lowercase().as_str() {
            "cyclic" => Self::Cyclic,
            "plane" => Self::Plane,
            "arbitrary" => Self::Arbitrary,
            _ => Self::Block,
        })
    }
}

/// Compute task-to-node mapping.
///
/// Returns a Vec where index = task_id, value = node index.
pub fn distribute_tasks(
    num_tasks: u32,
    num_nodes: u32,
    distribution: TaskDistribution,
) -> Vec<u32> {
    let num_nodes = num_nodes.max(1);
    let mut mapping = Vec::with_capacity(num_tasks as usize);

    match distribution {
        TaskDistribution::Block => {
            // Fill nodes evenly: [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]
            let per_node = num_tasks / num_nodes;
            let remainder = num_tasks % num_nodes;
            for node in 0..num_nodes {
                let count = per_node + if node < remainder { 1 } else { 0 };
                for _ in 0..count {
                    mapping.push(node);
                }
            }
        }
        TaskDistribution::Cyclic => {
            // Round-robin: [0, 1, 2, 0, 1, 2, ...]
            for task in 0..num_tasks {
                mapping.push(task % num_nodes);
            }
        }
        TaskDistribution::Plane => {
            // Same as block for now
            let per_node = num_tasks / num_nodes;
            let remainder = num_tasks % num_nodes;
            for node in 0..num_nodes {
                let count = per_node + if node < remainder { 1 } else { 0 };
                for _ in 0..count {
                    mapping.push(node);
                }
            }
        }
        TaskDistribution::Arbitrary => {
            // Default to block
            for task in 0..num_tasks {
                mapping.push(task % num_nodes);
            }
        }
    }

    mapping
}

/// CPU binding modes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CpuBind {
    #[default]
    None,
    /// Bind to cores.
    Cores,
    /// Bind to threads (hyperthreads).
    Threads,
    /// Bind to sockets.
    Sockets,
    /// Bind to NUMA domains.
    Ldoms,
    /// Bind by rank (task N → core N).
    Rank,
    /// Explicit CPU map (comma-separated core IDs).
    Map(String),
    /// Explicit CPU mask (hex mask).
    Mask(String),
}

impl FromStr for CpuBind {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Infallible> {
        let lower = s.to_lowercase();
        Ok(if lower.starts_with("map_cpu:") {
            Self::Map(s[8..].to_string())
        } else if lower.starts_with("mask_cpu:") {
            Self::Mask(s[9..].to_string())
        } else {
            match lower.as_str() {
                "cores" => Self::Cores,
                "threads" => Self::Threads,
                "sockets" => Self::Sockets,
                "ldoms" => Self::Ldoms,
                "rank" => Self::Rank,
                _ => Self::None,
            }
        })
    }
}

/// GPU binding modes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GpuBind {
    #[default]
    None,
    /// Bind to closest GPU(s) by PCIe topology.
    Closest,
    /// Explicit GPU map.
    Map(String),
    /// Explicit GPU mask.
    Mask(String),
}

impl FromStr for GpuBind {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Infallible> {
        let lower = s.to_lowercase();
        Ok(if lower.starts_with("map_gpu:") {
            Self::Map(s[8..].to_string())
        } else if lower.starts_with("mask_gpu:") {
            Self::Mask(s[9..].to_string())
        } else {
            match lower.as_str() {
                "closest" => Self::Closest,
                _ => Self::None,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribute_block() {
        let m = distribute_tasks(8, 4, TaskDistribution::Block);
        assert_eq!(m, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn test_distribute_block_uneven() {
        let m = distribute_tasks(10, 3, TaskDistribution::Block);
        // 10/3 = 3 rem 1 → 4,3,3
        assert_eq!(m, vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn test_distribute_cyclic() {
        let m = distribute_tasks(8, 4, TaskDistribution::Cyclic);
        assert_eq!(m, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn test_distribute_single_node() {
        let m = distribute_tasks(4, 1, TaskDistribution::Block);
        assert_eq!(m, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_step_state() {
        assert!(!StepState::Running.is_terminal());
        assert!(StepState::Completed.is_terminal());
        assert!(StepState::Failed.is_terminal());
    }

    #[test]
    fn test_cpu_bind_parse() {
        assert_eq!("cores".parse::<CpuBind>().unwrap(), CpuBind::Cores);
        assert_eq!("threads".parse::<CpuBind>().unwrap(), CpuBind::Threads);
        assert_eq!("none".parse::<CpuBind>().unwrap(), CpuBind::None);
        assert!(matches!(
            "map_cpu:0,1,2,3".parse::<CpuBind>().unwrap(),
            CpuBind::Map(_)
        ));
    }

    #[test]
    fn test_gpu_bind_parse() {
        assert_eq!("closest".parse::<GpuBind>().unwrap(), GpuBind::Closest);
        assert_eq!("none".parse::<GpuBind>().unwrap(), GpuBind::None);
        assert!(matches!(
            "map_gpu:0,1".parse::<GpuBind>().unwrap(),
            GpuBind::Map(_)
        ));
    }

    #[test]
    fn test_task_distribution_from_str() {
        assert_eq!(
            "cyclic".parse::<TaskDistribution>().unwrap(),
            TaskDistribution::Cyclic
        );
        assert_eq!(
            "block".parse::<TaskDistribution>().unwrap(),
            TaskDistribution::Block
        );
        assert_eq!(
            "plane".parse::<TaskDistribution>().unwrap(),
            TaskDistribution::Plane
        );
    }

    #[test]
    fn test_block_distribution_local_ranks() {
        // 8 tasks across 2 nodes, block distribution
        // Block: tasks 0-3 on node 0, tasks 4-7 on node 1
        let tasks = distribute_tasks(8, 2, TaskDistribution::Block);
        assert_eq!(tasks[0], 0); // task 0 -> node 0
        assert_eq!(tasks[3], 0); // task 3 -> node 0
        assert_eq!(tasks[4], 1); // task 4 -> node 1
        assert_eq!(tasks[7], 1); // task 7 -> node 1

        // Per-node local ranks: node 0 gets tasks 0..3 (local 0..3)
        let node0_tasks: Vec<usize> = tasks
            .iter()
            .enumerate()
            .filter(|(_, &n)| n == 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(node0_tasks.len(), 4);
        // local_rank for each task on node 0
        for (local_rank, &global_task) in node0_tasks.iter().enumerate() {
            assert_eq!(local_rank, global_task);
        }
    }

    #[test]
    fn test_cyclic_distribution_local_ranks() {
        // 8 tasks across 2 nodes, cyclic distribution
        // Cyclic: tasks 0,2,4,6 on node 0; tasks 1,3,5,7 on node 1
        let tasks = distribute_tasks(8, 2, TaskDistribution::Cyclic);
        assert_eq!(tasks[0], 0); // task 0 -> node 0
        assert_eq!(tasks[1], 1); // task 1 -> node 1
        assert_eq!(tasks[2], 0); // task 2 -> node 0
        assert_eq!(tasks[3], 1); // task 3 -> node 1

        let node0_count = tasks.iter().filter(|&&n| n == 0).count();
        let node1_count = tasks.iter().filter(|&&n| n == 1).count();
        assert_eq!(node0_count, 4);
        assert_eq!(node1_count, 4);
    }

    #[test]
    fn test_step_special_ids() {
        assert_eq!(STEP_BATCH, 0xFFFF_FFFE);
        assert_eq!(STEP_EXTERN, 0xFFFF_FFFD);
        assert_eq!(STEP_INTERACTIVE, 0xFFFF_FFFC);
        // All special IDs should be distinct
        assert_ne!(STEP_BATCH, STEP_EXTERN);
        assert_ne!(STEP_BATCH, STEP_INTERACTIVE);
        assert_ne!(STEP_EXTERN, STEP_INTERACTIVE);
    }

    #[test]
    fn test_step_state_transitions() {
        use crate::resource::ResourceAllocations;

        let mut step = JobStep {
            job_id: 1,
            step_id: 0,
            name: "test".into(),
            state: StepState::Pending,
            num_tasks: 4,
            cpus_per_task: 2,
            resources: ResourceAllocations::default(),
            nodes: vec!["node001".into()],
            distribution: TaskDistribution::Block,
            start_time: None,
            end_time: None,
            exit_code: None,
        };

        assert!(!step.state.is_terminal());
        assert_eq!(step.state.display(), "PENDING");

        // Pending -> Running
        step.state = StepState::Running;
        step.start_time = Some(chrono::Utc::now());
        assert!(!step.state.is_terminal());
        assert_eq!(step.state.display(), "RUNNING");

        // Running -> Completed
        step.state = StepState::Completed;
        step.exit_code = Some(0);
        step.end_time = Some(chrono::Utc::now());
        assert!(step.state.is_terminal());
        assert_eq!(step.state.display(), "COMPLETED");
    }

    #[test]
    fn test_step_state_failed() {
        let state = StepState::Failed;
        assert!(state.is_terminal());
        assert_eq!(state.display(), "FAILED");
    }

    #[test]
    fn test_step_state_cancelled() {
        let state = StepState::Cancelled;
        assert!(state.is_terminal());
        assert_eq!(state.display(), "CANCELLED");
    }

    #[test]
    fn test_distribute_zero_tasks() {
        let m = distribute_tasks(0, 4, TaskDistribution::Block);
        assert!(m.is_empty());
    }

    #[test]
    fn test_distribute_zero_nodes_uses_one() {
        // 0 nodes should be treated as 1 (max(1))
        let m = distribute_tasks(4, 0, TaskDistribution::Block);
        assert_eq!(m.len(), 4);
        assert!(m.iter().all(|&n| n == 0));
    }
}
