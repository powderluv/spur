use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use crate::resource::ResourceSet;

/// Unique job identifier assigned by the controller.
pub type JobId = u32;

/// Job states matching Slurm's state model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobState {
    Pending,
    Running,
    Completing,
    Completed,
    Failed,
    Cancelled,
    Timeout,
    NodeFail,
    Preempted,
    Suspended,
}

impl JobState {
    /// Short code used in squeue output (matches Slurm).
    pub fn code(&self) -> &'static str {
        match self {
            Self::Pending => "PD",
            Self::Running => "R",
            Self::Completing => "CG",
            Self::Completed => "CD",
            Self::Failed => "F",
            Self::Cancelled => "CA",
            Self::Timeout => "TO",
            Self::NodeFail => "NF",
            Self::Preempted => "PR",
            Self::Suspended => "S",
        }
    }

    /// Full display name (matches Slurm).
    pub fn display(&self) -> &'static str {
        match self {
            Self::Pending => "PENDING",
            Self::Running => "RUNNING",
            Self::Completing => "COMPLETING",
            Self::Completed => "COMPLETED",
            Self::Failed => "FAILED",
            Self::Cancelled => "CANCELLED",
            Self::Timeout => "TIMEOUT",
            Self::NodeFail => "NODE_FAIL",
            Self::Preempted => "PREEMPTED",
            Self::Suspended => "SUSPENDED",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Timeout | Self::NodeFail
        )
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Running | Self::Completing | Self::Suspended)
    }
}

impl std::fmt::Display for JobState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display())
    }
}

/// Reason a job is pending.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PendingReason {
    #[default]
    None,
    Priority,
    Resources,
    PartitionDown,
    PartitionNodeLimit,
    PartitionTimeLimit,
    Dependency,
    NodeDown,
    Held,
    QoSMaxJobsPerUser,
    ReqNodeNotAvail,
}

impl PendingReason {
    pub fn display(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Priority => "Priority",
            Self::Resources => "Resources",
            Self::PartitionDown => "PartitionDown",
            Self::PartitionNodeLimit => "PartNodeLimit",
            Self::PartitionTimeLimit => "PartTimeLimit",
            Self::Dependency => "Dependency",
            Self::NodeDown => "NodeDown",
            Self::Held => "JobHeldUser",
            Self::QoSMaxJobsPerUser => "QOSMaxJobsPerUserLimit",
            Self::ReqNodeNotAvail => "ReqNodeNotAvail",
        }
    }
}

impl std::fmt::Display for PendingReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display())
    }
}

/// Job specification submitted by the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSpec {
    pub name: String,
    pub partition: Option<String>,
    pub account: Option<String>,
    pub user: String,
    pub uid: u32,
    pub gid: u32,

    // Resources
    pub num_nodes: u32,
    pub num_tasks: u32,
    pub tasks_per_node: Option<u32>,
    pub cpus_per_task: u32,
    pub memory_per_node_mb: Option<u64>,
    pub memory_per_cpu_mb: Option<u64>,
    pub gres: Vec<String>,

    // Execution
    pub script: Option<String>,
    pub argv: Vec<String>,
    pub work_dir: String,
    pub stdout_path: Option<String>,
    pub stderr_path: Option<String>,
    pub environment: HashMap<String, String>,

    // Time
    pub time_limit: Option<chrono::Duration>,
    pub time_min: Option<chrono::Duration>,

    // Scheduling
    pub qos: Option<String>,
    pub priority: Option<u32>,
    pub reservation: Option<String>,
    pub dependency: Vec<String>,
    pub nodelist: Option<String>,
    pub exclude: Option<String>,

    // Array
    pub array_spec: Option<String>,

    // Flags
    pub requeue: bool,
    pub exclusive: bool,
    pub hold: bool,
    pub comment: Option<String>,
    pub wckey: Option<String>,

    // Container
    pub container_image: Option<String>,
    pub container_mounts: Vec<String>,
    pub container_workdir: Option<String>,
    pub container_name: Option<String>,
    pub container_readonly: bool,
    pub container_mount_home: bool,
    pub container_env: HashMap<String, String>,
    pub container_entrypoint: Option<String>,
    pub container_remap_root: bool,
}

impl Default for JobSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            partition: None,
            account: None,
            user: String::new(),
            uid: 0,
            gid: 0,
            num_nodes: 1,
            num_tasks: 1,
            tasks_per_node: None,
            cpus_per_task: 1,
            memory_per_node_mb: None,
            memory_per_cpu_mb: None,
            gres: Vec::new(),
            script: None,
            argv: Vec::new(),
            work_dir: String::from("/tmp"),
            stdout_path: None,
            stderr_path: None,
            environment: HashMap::new(),
            time_limit: None,
            time_min: None,
            qos: None,
            priority: None,
            reservation: None,
            dependency: Vec::new(),
            nodelist: None,
            exclude: None,
            array_spec: None,
            requeue: false,
            exclusive: false,
            hold: false,
            comment: None,
            wckey: None,
            container_image: None,
            container_mounts: Vec::new(),
            container_workdir: None,
            container_name: None,
            container_readonly: false,
            container_mount_home: false,
            container_env: HashMap::new(),
            container_entrypoint: None,
            container_remap_root: false,
        }
    }
}

/// Internal job record held by the controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub job_id: JobId,
    pub spec: JobSpec,
    pub state: JobState,
    pub pending_reason: PendingReason,
    pub priority: u32,

    pub submit_time: DateTime<Utc>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,

    pub allocated_nodes: Vec<String>,
    pub allocated_resources: Option<ResourceSet>,

    pub exit_code: Option<i32>,

    /// Number of times this job has been requeued.
    #[serde(default)]
    pub requeue_count: u32,

    // Array support
    pub array_job_id: Option<JobId>,
    pub array_task_id: Option<u32>,
    /// Max concurrent tasks for this array (0 = unlimited).
    #[serde(default)]
    pub array_max_concurrent: Option<u32>,
}

impl Job {
    pub fn new(job_id: JobId, spec: JobSpec) -> Self {
        let priority = spec.priority.unwrap_or(1000);
        let state = if spec.hold {
            JobState::Pending
        } else {
            JobState::Pending
        };
        let pending_reason = if spec.hold {
            PendingReason::Held
        } else {
            PendingReason::Priority
        };
        Self {
            job_id,
            spec,
            state,
            pending_reason,
            priority,
            submit_time: Utc::now(),
            start_time: None,
            end_time: None,
            allocated_nodes: Vec::new(),
            allocated_resources: None,
            exit_code: None,
            requeue_count: 0,
            array_job_id: None,
            array_task_id: None,
            array_max_concurrent: None,
        }
    }

    /// Compute the run time.
    pub fn run_time(&self) -> Option<chrono::Duration> {
        let start = self.start_time?;
        let end = self.end_time.unwrap_or_else(Utc::now);
        Some(end - start)
    }

    /// Resolve stdout path, substituting %j/%N patterns.
    pub fn resolved_stdout(&self) -> String {
        self.resolve_path(self.spec.stdout_path.as_deref().unwrap_or("spur-%j.out"))
    }

    /// Resolve stderr path.
    pub fn resolved_stderr(&self) -> String {
        self.resolve_path(self.spec.stderr_path.as_deref().unwrap_or("spur-%j.out"))
    }

    fn resolve_path(&self, pattern: &str) -> String {
        let mut result = pattern.to_string();
        result = result.replace("%j", &self.job_id.to_string());
        result = result.replace("%J", &self.job_id.to_string());
        result = result.replace("%x", &self.spec.name);
        if let Some(tid) = self.array_task_id {
            result = result.replace("%a", &tid.to_string());
            result = result.replace("%A", &self.array_job_id.unwrap_or(self.job_id).to_string());
        }
        if let Some(node) = self.allocated_nodes.first() {
            result = result.replace("%N", node);
        }
        result = result.replace("%u", &self.spec.user);
        result
    }
}

/// State transitions.
#[derive(Debug, Error)]
pub enum JobTransitionError {
    #[error("invalid transition from {from} to {to}")]
    Invalid { from: JobState, to: JobState },
}

impl Job {
    /// Attempt a state transition, enforcing the state machine.
    pub fn transition(&mut self, to: JobState) -> Result<(), JobTransitionError> {
        let valid = match (self.state, to) {
            (JobState::Pending, JobState::Running) => true,
            (JobState::Pending, JobState::Cancelled) => true,
            (JobState::Running, JobState::Completing) => true,
            (JobState::Running, JobState::Completed) => true,
            (JobState::Running, JobState::Failed) => true,
            (JobState::Running, JobState::Cancelled) => true,
            (JobState::Running, JobState::Timeout) => true,
            (JobState::Running, JobState::NodeFail) => true,
            (JobState::Running, JobState::Preempted) => true,
            (JobState::Running, JobState::Suspended) => true,
            (JobState::Completing, JobState::Completed) => true,
            (JobState::Completing, JobState::Failed) => true,
            (JobState::Suspended, JobState::Running) => true,
            (JobState::Suspended, JobState::Cancelled) => true,
            // Requeue transitions: terminal → Pending (for --requeue jobs)
            (JobState::Timeout, JobState::Pending) => true,
            (JobState::Preempted, JobState::Pending) => true,
            (JobState::NodeFail, JobState::Pending) => true,
            (JobState::Failed, JobState::Pending) => true,
            _ => false,
        };

        if valid {
            self.state = to;
            if to.is_terminal() && self.end_time.is_none() {
                self.end_time = Some(Utc::now());
            }
            // Requeue: clear end_time when going back to Pending
            if to == JobState::Pending {
                self.end_time = None;
            }
            Ok(())
        } else {
            Err(JobTransitionError::Invalid {
                from: self.state,
                to,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job() -> Job {
        Job::new(
            1,
            JobSpec {
                name: "test".into(),
                user: "alice".into(),
                ..Default::default()
            },
        )
    }

    #[test]
    fn test_state_transitions() {
        let mut job = make_job();
        assert_eq!(job.state, JobState::Pending);

        job.transition(JobState::Running).unwrap();
        assert_eq!(job.state, JobState::Running);
        assert!(job.start_time.is_none()); // start_time set externally

        job.transition(JobState::Completed).unwrap();
        assert_eq!(job.state, JobState::Completed);
        assert!(job.end_time.is_some());
    }

    #[test]
    fn test_invalid_transition() {
        let mut job = make_job();
        assert!(job.transition(JobState::Completed).is_err());
    }

    #[test]
    fn test_path_resolution() {
        let mut job = make_job();
        job.job_id = 42;
        job.spec.name = "train".into();
        job.spec.user = "bob".into();

        assert_eq!(job.resolve_path("spur-%j.out"), "spur-42.out");
        assert_eq!(job.resolve_path("output-%x-%u.log"), "output-train-bob.log");
    }

    #[test]
    fn test_state_codes() {
        assert_eq!(JobState::Pending.code(), "PD");
        assert_eq!(JobState::Running.code(), "R");
        assert_eq!(JobState::Completed.code(), "CD");
    }
}
