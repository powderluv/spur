//! QOS enforcement logic.
//!
//! Checks per-QOS limits before allowing a job to be scheduled.

use crate::accounting::{Qos, QosLimits, TresRecord, TresType};
use crate::job::{Job, JobState, PendingReason};

/// Result of QOS limit check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QosCheckResult {
    /// Job passes all QOS checks.
    Allowed,
    /// Job blocked by a QOS limit.
    Blocked(PendingReason),
}

/// Check if a job would violate QOS limits.
///
/// `running_jobs` = currently running jobs from the same user+QOS.
/// `pending_jobs` = currently pending jobs from the same user+QOS.
pub fn check_qos_limits(
    job: &Job,
    qos: &Qos,
    user_running_count: u32,
    user_submitted_count: u32,
    user_running_tres: &TresRecord,
) -> QosCheckResult {
    let limits = &qos.limits;

    // Max jobs per user
    if let Some(max) = limits.max_jobs_per_user {
        if user_running_count >= max {
            return QosCheckResult::Blocked(PendingReason::QoSMaxJobsPerUser);
        }
    }

    // Max submit jobs per user
    if let Some(max) = limits.max_submit_jobs_per_user {
        if user_submitted_count >= max {
            return QosCheckResult::Blocked(PendingReason::QoSMaxJobsPerUser);
        }
    }

    // Max wall time
    if let Some(max_wall) = limits.max_wall_minutes {
        if let Some(job_wall) = job.spec.time_limit {
            if job_wall.num_minutes() > max_wall as i64 {
                return QosCheckResult::Blocked(PendingReason::PartitionTimeLimit);
            }
        }
    }

    // Max TRES per job
    if let Some(ref max_tres) = limits.max_tres_per_job {
        let job_cpus = (job.spec.num_tasks * job.spec.cpus_per_task) as u64;
        if max_tres.get(TresType::Cpu) > 0 && job_cpus > max_tres.get(TresType::Cpu) {
            return QosCheckResult::Blocked(PendingReason::Resources);
        }

        if let Some(mem) = job.spec.memory_per_node_mb {
            let total_mem = mem * job.spec.num_nodes as u64;
            if max_tres.get(TresType::Memory) > 0 && total_mem > max_tres.get(TresType::Memory) {
                return QosCheckResult::Blocked(PendingReason::Resources);
            }
        }
    }

    // Max TRES per user (group limit)
    if let Some(ref max_tres) = limits.max_tres_per_user {
        let job_cpus = (job.spec.num_tasks * job.spec.cpus_per_task) as u64;
        let new_total_cpu = user_running_tres.get(TresType::Cpu) + job_cpus;
        if max_tres.get(TresType::Cpu) > 0 && new_total_cpu > max_tres.get(TresType::Cpu) {
            return QosCheckResult::Blocked(PendingReason::Resources);
        }
    }

    QosCheckResult::Allowed
}

/// Calculate effective priority including QOS priority adjustment.
pub fn qos_adjusted_priority(base_priority: u32, qos: &Qos) -> u32 {
    let adjusted = base_priority as i64 + qos.priority as i64;
    adjusted.max(1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::JobSpec;

    fn make_qos(max_jobs: Option<u32>, max_wall: Option<u32>) -> Qos {
        Qos {
            name: "test".into(),
            limits: QosLimits {
                max_jobs_per_user: max_jobs,
                max_wall_minutes: max_wall,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn make_test_job() -> Job {
        Job::new(1, JobSpec {
            name: "test".into(),
            user: "alice".into(),
            num_tasks: 4,
            cpus_per_task: 1,
            time_limit: Some(chrono::Duration::hours(2)),
            ..Default::default()
        })
    }

    #[test]
    fn test_allowed_when_no_limits() {
        let qos = make_qos(None, None);
        let job = make_test_job();
        let result = check_qos_limits(&job, &qos, 0, 0, &TresRecord::new());
        assert_eq!(result, QosCheckResult::Allowed);
    }

    #[test]
    fn test_blocked_by_max_jobs() {
        let qos = make_qos(Some(5), None);
        let job = make_test_job();
        let result = check_qos_limits(&job, &qos, 5, 5, &TresRecord::new());
        assert_eq!(result, QosCheckResult::Blocked(PendingReason::QoSMaxJobsPerUser));
    }

    #[test]
    fn test_allowed_under_max_jobs() {
        let qos = make_qos(Some(5), None);
        let job = make_test_job();
        let result = check_qos_limits(&job, &qos, 3, 3, &TresRecord::new());
        assert_eq!(result, QosCheckResult::Allowed);
    }

    #[test]
    fn test_blocked_by_max_wall() {
        let qos = make_qos(None, Some(60)); // 1 hour max
        let job = make_test_job(); // 2 hour job
        let result = check_qos_limits(&job, &qos, 0, 0, &TresRecord::new());
        assert_eq!(result, QosCheckResult::Blocked(PendingReason::PartitionTimeLimit));
    }

    #[test]
    fn test_blocked_by_max_tres_per_job() {
        let mut tres = TresRecord::new();
        tres.set(TresType::Cpu, 2); // Max 2 CPUs per job
        let qos = Qos {
            name: "restricted".into(),
            limits: QosLimits {
                max_tres_per_job: Some(tres),
                ..Default::default()
            },
            ..Default::default()
        };
        let job = make_test_job(); // 4 CPUs
        let result = check_qos_limits(&job, &qos, 0, 0, &TresRecord::new());
        assert_eq!(result, QosCheckResult::Blocked(PendingReason::Resources));
    }

    #[test]
    fn test_qos_priority_adjustment() {
        let qos = Qos { priority: 500, ..Default::default() };
        assert_eq!(qos_adjusted_priority(1000, &qos), 1500);

        let qos_neg = Qos { priority: -200, ..Default::default() };
        assert_eq!(qos_adjusted_priority(1000, &qos_neg), 800);
    }

    #[test]
    fn test_qos_priority_floor() {
        let qos = Qos { priority: -2000, ..Default::default() };
        assert_eq!(qos_adjusted_priority(1000, &qos), 1); // Floor at 1
    }
}
