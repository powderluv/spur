//! T50: Core type tests.
//!
//! Tests for Job, Node, ResourceSet, Partition types.
//! Corresponds to Slurm's slurm_unit/common/ tests.

#[cfg(test)]
mod tests {
    use crate::harness::*;
    use spur_core::job::*;
    use spur_core::node::*;
    use spur_core::partition::*;
    use spur_core::resource::*;

    // ── T50.1: Job state machine ──────────────────────────────────

    #[test]
    fn t50_1_job_initial_state_is_pending() {
        reset_job_ids();
        let job = make_job("test");
        assert_job_state(&job, JobState::Pending);
    }

    #[test]
    fn t50_2_job_pending_to_running() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_ok(&mut job, JobState::Running);
        assert_job_state(&job, JobState::Running);
    }

    #[test]
    fn t50_3_job_running_to_completed() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::Completed);
        assert!(job.state.is_terminal());
        assert!(job.end_time.is_some());
    }

    #[test]
    fn t50_4_job_running_to_failed() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::Failed);
        assert!(job.state.is_terminal());
    }

    #[test]
    fn t50_5_job_pending_to_cancelled() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_ok(&mut job, JobState::Cancelled);
        assert!(job.state.is_terminal());
    }

    #[test]
    fn t50_6_job_running_to_timeout() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::Timeout);
        assert!(job.state.is_terminal());
    }

    #[test]
    fn t50_7_job_running_to_node_fail() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::NodeFail);
        assert!(job.state.is_terminal());
    }

    #[test]
    fn t50_8_job_running_to_preempted() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::Preempted);
        assert_eq!(job.state, JobState::Preempted);
    }

    #[test]
    fn t50_9_job_running_to_suspended_and_back() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::Suspended);
        assert!(job.state.is_active());
        assert_transition_ok(&mut job, JobState::Running);
    }

    #[test]
    fn t50_10_invalid_pending_to_completed() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_err(&mut job, JobState::Completed);
    }

    #[test]
    fn t50_11_invalid_completed_to_running() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::Completed);
        assert_transition_err(&mut job, JobState::Running);
    }

    #[test]
    fn t50_12_invalid_pending_to_failed() {
        reset_job_ids();
        let mut job = make_job("test");
        assert_transition_err(&mut job, JobState::Failed);
    }

    // ── T50.13: Job state display ─────────────────────────────────

    #[test]
    fn t50_13_state_codes() {
        assert_eq!(JobState::Pending.code(), "PD");
        assert_eq!(JobState::Running.code(), "R");
        assert_eq!(JobState::Completing.code(), "CG");
        assert_eq!(JobState::Completed.code(), "CD");
        assert_eq!(JobState::Failed.code(), "F");
        assert_eq!(JobState::Cancelled.code(), "CA");
        assert_eq!(JobState::Timeout.code(), "TO");
        assert_eq!(JobState::NodeFail.code(), "NF");
        assert_eq!(JobState::Preempted.code(), "PR");
        assert_eq!(JobState::Suspended.code(), "S");
    }

    #[test]
    fn t50_14_state_display_names() {
        assert_eq!(JobState::Pending.display(), "PENDING");
        assert_eq!(JobState::Running.display(), "RUNNING");
        assert_eq!(JobState::Completed.display(), "COMPLETED");
    }

    // ── T50.15: Job path resolution ───────────────────────────────

    #[test]
    fn t50_15_path_resolve_job_id() {
        reset_job_ids();
        let mut job = make_job("train");
        job.job_id = 42;
        assert_eq!(job.resolved_stdout(), "spur-42.out");
    }

    #[test]
    fn t50_16_path_resolve_custom_pattern() {
        reset_job_ids();
        let mut job = make_job("train");
        job.job_id = 42;
        job.spec.user = "bob".into();
        job.spec.stdout_path = Some("output-%x-%u-%j.log".into());
        assert_eq!(job.resolved_stdout(), "output-train-bob-42.log");
    }

    #[test]
    fn t50_17_path_resolve_node_pattern() {
        reset_job_ids();
        let mut job = make_job("test");
        job.job_id = 10;
        job.allocated_nodes = vec!["gpu001".into()];
        job.spec.stdout_path = Some("out-%N-%j.log".into());
        assert_eq!(job.resolved_stdout(), "out-gpu001-10.log");
    }

    // ── T50.18: Job run time ──────────────────────────────────────

    #[test]
    fn t50_18_run_time_none_when_not_started() {
        let job = make_job("test");
        assert!(job.run_time().is_none());
    }

    #[test]
    fn t50_19_run_time_computed_when_running() {
        let mut job = make_job("test");
        job.start_time = Some(chrono::Utc::now() - chrono::Duration::minutes(5));
        let rt = job.run_time().unwrap();
        // Should be roughly 5 minutes (allow 2 second tolerance)
        assert!(rt.num_seconds() >= 298 && rt.num_seconds() <= 302);
    }

    // ── T50.20: Node state ────────────────────────────────────────

    #[test]
    fn t50_20_node_initial_state() {
        let node = Node::new("node001".into(), ResourceSet::default());
        assert_eq!(node.state, NodeState::Unknown);
    }

    #[test]
    fn t50_21_node_state_from_alloc() {
        let mut node = Node::new(
            "node001".into(),
            ResourceSet {
                cpus: 64,
                memory_mb: 256_000,
                ..Default::default()
            },
        );
        node.state = NodeState::Idle;
        node.update_state_from_alloc();
        assert_eq!(node.state, NodeState::Idle);

        node.alloc_resources.cpus = 32;
        node.update_state_from_alloc();
        assert_eq!(node.state, NodeState::Mixed);

        node.alloc_resources.cpus = 64;
        node.update_state_from_alloc();
        assert_eq!(node.state, NodeState::Allocated);
    }

    #[test]
    fn t50_22_node_admin_state_not_overridden() {
        let mut node = Node::new(
            "node001".into(),
            ResourceSet {
                cpus: 64,
                ..Default::default()
            },
        );
        node.state = NodeState::Drain;
        node.alloc_resources.cpus = 0;
        node.update_state_from_alloc();
        // Should stay Drain, not flip to Idle
        assert_eq!(node.state, NodeState::Drain);
    }

    #[test]
    fn t50_23_node_schedulable() {
        assert!(NodeState::Idle.is_available());
        assert!(NodeState::Mixed.is_available());
        assert!(!NodeState::Down.is_available());
        assert!(!NodeState::Drain.is_available());
        assert!(!NodeState::Allocated.is_available());
    }

    // ── T50.24: ResourceSet ───────────────────────────────────────

    #[test]
    fn t50_24_resource_can_satisfy() {
        let avail = ResourceSet {
            cpus: 64,
            memory_mb: 256_000,
            ..Default::default()
        };
        let req = ResourceSet {
            cpus: 32,
            memory_mb: 128_000,
            ..Default::default()
        };
        assert!(avail.can_satisfy(&req));
    }

    #[test]
    fn t50_25_resource_cannot_satisfy_cpu() {
        let avail = ResourceSet {
            cpus: 32,
            memory_mb: 256_000,
            ..Default::default()
        };
        let req = ResourceSet {
            cpus: 64,
            memory_mb: 128_000,
            ..Default::default()
        };
        assert!(!avail.can_satisfy(&req));
    }

    #[test]
    fn t50_26_resource_cannot_satisfy_memory() {
        let avail = ResourceSet {
            cpus: 64,
            memory_mb: 100_000,
            ..Default::default()
        };
        let req = ResourceSet {
            cpus: 32,
            memory_mb: 200_000,
            ..Default::default()
        };
        assert!(!avail.can_satisfy(&req));
    }

    #[test]
    fn t50_27_resource_subtract() {
        let total = ResourceSet {
            cpus: 64,
            memory_mb: 256_000,
            ..Default::default()
        };
        let used = ResourceSet {
            cpus: 24,
            memory_mb: 100_000,
            ..Default::default()
        };
        let avail = total.subtract(&used);
        assert_eq!(avail.cpus, 40);
        assert_eq!(avail.memory_mb, 156_000);
    }

    // ── T50.28: GRES parsing ──────────────────────────────────────

    #[test]
    fn t50_28_parse_gres_full() {
        let (name, gtype, count) = spur_core::resource::parse_gres("gpu:mi300x:4").unwrap();
        assert_eq!(name, "gpu");
        assert_eq!(gtype.unwrap(), "mi300x");
        assert_eq!(count, 4);
    }

    #[test]
    fn t50_29_parse_gres_no_type() {
        let (name, gtype, count) = spur_core::resource::parse_gres("gpu:2").unwrap();
        assert_eq!(name, "gpu");
        assert!(gtype.is_none());
        assert_eq!(count, 2);
    }

    #[test]
    fn t50_30_parse_gres_bare() {
        let (name, gtype, count) = spur_core::resource::parse_gres("license").unwrap();
        assert_eq!(name, "license");
        assert!(gtype.is_none());
        assert_eq!(count, 1);
    }

    // ── T50.31: Partition state ───────────────────────────────────

    #[test]
    fn t50_31_partition_states() {
        assert_eq!(PartitionState::Up.display(), "up");
        assert_eq!(PartitionState::Down.display(), "down");
        assert_eq!(PartitionState::Drain.display(), "drain");
        assert_eq!(PartitionState::Inactive.display(), "inactive");
    }

    // ── T50.32: Held job ──────────────────────────────────────────

    #[test]
    fn t50_32_held_job_starts_pending() {
        reset_job_ids();
        let job = Job::new(
            99,
            JobSpec {
                name: "held".into(),
                user: "test".into(),
                hold: true,
                ..Default::default()
            },
        );
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.pending_reason, PendingReason::Held);
    }

    // ── T50.33–37: Requeue state transitions ───────────────────

    #[test]
    fn t50_33_requeue_from_timeout() {
        reset_job_ids();
        let mut job = make_job("requeue-timeout");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::Timeout);
        assert!(job.end_time.is_some(), "end_time should be set on Timeout");
        // Requeue: Timeout → Pending should succeed
        assert_transition_ok(&mut job, JobState::Pending);
        assert_job_state(&job, JobState::Pending);
        assert!(
            job.end_time.is_none(),
            "end_time should be cleared on requeue"
        );
    }

    #[test]
    fn t50_34_requeue_from_preempted() {
        reset_job_ids();
        let mut job = make_job("requeue-preempted");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::Preempted);
        // Preempted → Pending should succeed
        assert_transition_ok(&mut job, JobState::Pending);
        assert_job_state(&job, JobState::Pending);
        assert!(
            job.end_time.is_none(),
            "end_time should be cleared on requeue"
        );
    }

    #[test]
    fn t50_35_requeue_from_node_fail() {
        reset_job_ids();
        let mut job = make_job("requeue-nodefail");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::NodeFail);
        // NodeFail → Pending should succeed
        assert_transition_ok(&mut job, JobState::Pending);
        assert_job_state(&job, JobState::Pending);
    }

    #[test]
    fn t50_36_requeue_from_failed() {
        reset_job_ids();
        let mut job = make_job("requeue-failed");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::Failed);
        // Failed → Pending should succeed
        assert_transition_ok(&mut job, JobState::Pending);
        assert_job_state(&job, JobState::Pending);
    }

    #[test]
    fn t50_37_requeue_from_completed_fails() {
        reset_job_ids();
        let mut job = make_job("requeue-completed");
        assert_transition_ok(&mut job, JobState::Running);
        assert_transition_ok(&mut job, JobState::Completed);
        // Completed → Pending should fail (Completed is not retriable)
        assert_transition_err(&mut job, JobState::Pending);
        assert_job_state(&job, JobState::Completed);
    }

    // ── T50.38–40: Drain / Draining node behavior ──────────────

    #[test]
    fn t50_38_drain_preserves_state() {
        let mut node = Node::new(
            "n1".into(),
            ResourceSet {
                cpus: 64,
                memory_mb: 256_000,
                ..Default::default()
            },
        );
        node.state = NodeState::Drain;
        // update_state_from_alloc should not override Drain
        node.update_state_from_alloc();
        assert_eq!(node.state, NodeState::Drain);
    }

    #[test]
    fn t50_39_draining_not_schedulable() {
        let mut node = Node::new(
            "n1".into(),
            ResourceSet {
                cpus: 64,
                memory_mb: 256_000,
                ..Default::default()
            },
        );
        node.state = NodeState::Draining;
        assert!(
            !node.is_schedulable(),
            "Draining node should not be schedulable"
        );
    }

    #[test]
    fn t50_40_draining_preserves_state() {
        let mut node = Node::new(
            "n1".into(),
            ResourceSet {
                cpus: 64,
                memory_mb: 256_000,
                ..Default::default()
            },
        );
        node.state = NodeState::Draining;
        node.alloc_resources.cpus = 32;
        // update_state_from_alloc should not override Draining
        node.update_state_from_alloc();
        assert_eq!(node.state, NodeState::Draining);
    }

    #[test]
    fn t50_41_error_preserves_state() {
        let mut node = Node::new(
            "n1".into(),
            ResourceSet {
                cpus: 64,
                memory_mb: 256_000,
                ..Default::default()
            },
        );
        node.state = NodeState::Error;
        node.update_state_from_alloc();
        assert_eq!(node.state, NodeState::Error);
    }

    // ── T50.42–43: Requeue does not reset from Cancelled ───────

    #[test]
    fn t50_42_requeue_from_cancelled_fails() {
        reset_job_ids();
        let mut job = make_job("requeue-cancelled");
        assert_transition_ok(&mut job, JobState::Cancelled);
        // Cancelled → Pending should fail
        assert_transition_err(&mut job, JobState::Pending);
    }

    #[test]
    fn t50_43_node_available_states() {
        // Comprehensive check: only Idle and Mixed are available
        assert!(NodeState::Idle.is_available());
        assert!(NodeState::Mixed.is_available());
        assert!(!NodeState::Down.is_available());
        assert!(!NodeState::Drain.is_available());
        assert!(!NodeState::Draining.is_available());
        assert!(!NodeState::Allocated.is_available());
        assert!(!NodeState::Error.is_available());
        assert!(!NodeState::Unknown.is_available());
    }

    // ── T50.44: Mail type field ─────────────────────────────────

    #[test]
    fn t50_44_mail_type_field() {
        let spec = spur_core::job::JobSpec {
            mail_type: vec!["BEGIN".into(), "END".into()],
            mail_user: Some("alice@example.com".into()),
            ..Default::default()
        };
        assert_eq!(spec.mail_type.len(), 2);
        assert_eq!(spec.mail_user.as_deref(), Some("alice@example.com"));
    }

    // ── T50.45: Interactive flag ────────────────────────────────

    #[test]
    fn t50_45_interactive_flag() {
        let spec = spur_core::job::JobSpec {
            interactive: true,
            ..Default::default()
        };
        assert!(spec.interactive);
    }

    #[test]
    fn t50_46_interactive_default_false() {
        let spec = spur_core::job::JobSpec::default();
        assert!(!spec.interactive);
    }

    // ── T50.47–49: MPI / distribution fields ────────────────────

    #[test]
    fn t50_47_mpi_field_default_none() {
        let spec = spur_core::job::JobSpec::default();
        assert!(spec.mpi.is_none());
    }

    #[test]
    fn t50_48_distribution_field_default_none() {
        let spec = spur_core::job::JobSpec::default();
        assert!(spec.distribution.is_none());
    }

    #[test]
    fn t50_49_mpi_field_set() {
        let spec = spur_core::job::JobSpec {
            mpi: Some("pmix".into()),
            distribution: Some("cyclic".into()),
            ..Default::default()
        };
        assert_eq!(spec.mpi.as_deref(), Some("pmix"));
        assert_eq!(spec.distribution.as_deref(), Some("cyclic"));
    }
}
