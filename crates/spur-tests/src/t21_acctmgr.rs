//! T21: Accounting management tests.
//!
//! Corresponds to Slurm's test21.x series (sacctmgr).
//! Tests TRES records, QOS limits, accounting data models.

#[cfg(test)]
mod tests {
    use spur_core::accounting::*;
    use spur_core::qos::*;
    use spur_core::job::*;

    // ── T21.1: TRES records ──────────────────────────────────────

    #[test]
    fn t21_1_tres_set_get() {
        let mut rec = TresRecord::new();
        rec.set(TresType::Cpu, 64);
        rec.set(TresType::Memory, 256000);
        assert_eq!(rec.get(TresType::Cpu), 64);
        assert_eq!(rec.get(TresType::Memory), 256000);
        assert_eq!(rec.get(TresType::Gpu), 0); // Not set
    }

    #[test]
    fn t21_2_tres_add() {
        let mut a = TresRecord::new();
        a.set(TresType::Cpu, 10);
        a.set(TresType::Memory, 1000);
        let mut b = TresRecord::new();
        b.set(TresType::Cpu, 20);
        b.set(TresType::Gpu, 4);
        a.add(&b);
        assert_eq!(a.get(TresType::Cpu), 30);
        assert_eq!(a.get(TresType::Memory), 1000);
        assert_eq!(a.get(TresType::Gpu), 4);
    }

    #[test]
    fn t21_3_tres_format() {
        let mut rec = TresRecord::new();
        rec.set(TresType::Cpu, 64);
        rec.set(TresType::Gpu, 8);
        let s = rec.format();
        assert!(s.contains("cpu=64"));
        assert!(s.contains("gres/gpu=8"));
    }

    #[test]
    fn t21_4_tres_parse() {
        let rec = TresRecord::parse("cpu=128,mem=512000,gres/gpu=8");
        assert_eq!(rec.get(TresType::Cpu), 128);
        assert_eq!(rec.get(TresType::Memory), 512000);
        assert_eq!(rec.get(TresType::Gpu), 8);
    }

    #[test]
    fn t21_5_tres_roundtrip() {
        let mut orig = TresRecord::new();
        orig.set(TresType::Cpu, 32);
        orig.set(TresType::Gpu, 4);
        let parsed = TresRecord::parse(&orig.format());
        assert_eq!(parsed.get(TresType::Cpu), 32);
        assert_eq!(parsed.get(TresType::Gpu), 4);
    }

    #[test]
    fn t21_6_tres_type_names() {
        assert_eq!(TresType::Cpu.name(), "cpu");
        assert_eq!(TresType::Memory.name(), "mem");
        assert_eq!(TresType::Gpu.name(), "gres/gpu");
        assert_eq!(TresType::from_name("gpu"), Some(TresType::Gpu));
        assert_eq!(TresType::from_name("cpu"), Some(TresType::Cpu));
        assert_eq!(TresType::from_name("unknown"), None);
    }

    // ── T21.7: Account defaults ──────────────────────────────────

    #[test]
    fn t21_7_account_defaults() {
        let acct = Account::default();
        assert_eq!(acct.fairshare_weight, 1);
        assert!(acct.parent.is_none());
        assert!(acct.limits.max_running_jobs.is_none());
    }

    // ── T21.8: QOS defaults ──────────────────────────────────────

    #[test]
    fn t21_8_qos_defaults() {
        let qos = Qos::default();
        assert_eq!(qos.priority, 0);
        assert_eq!(qos.preempt_mode, QosPreemptMode::Off);
        assert_eq!(qos.usage_factor, 1.0);
    }

    #[test]
    fn t21_9_qos_preempt_modes() {
        assert_eq!(QosPreemptMode::from_str("cancel"), QosPreemptMode::Cancel);
        assert_eq!(QosPreemptMode::from_str("requeue"), QosPreemptMode::Requeue);
        assert_eq!(QosPreemptMode::from_str("suspend"), QosPreemptMode::Suspend);
        assert_eq!(QosPreemptMode::from_str("off"), QosPreemptMode::Off);
    }

    // ── T21.10: QOS limit enforcement ────────────────────────────

    #[test]
    fn t21_10_qos_no_limits_allowed() {
        let qos = Qos::default();
        let job = Job::new(1, JobSpec {
            name: "test".into(),
            user: "alice".into(),
            num_tasks: 4,
            cpus_per_task: 1,
            ..Default::default()
        });
        let result = check_qos_limits(&job, &qos, 0, 0, &TresRecord::new());
        assert_eq!(result, QosCheckResult::Allowed);
    }

    #[test]
    fn t21_11_qos_max_jobs_blocked() {
        let qos = Qos {
            name: "limited".into(),
            limits: QosLimits {
                max_jobs_per_user: Some(2),
                ..Default::default()
            },
            ..Default::default()
        };
        let job = Job::new(1, JobSpec {
            name: "test".into(),
            user: "alice".into(),
            ..Default::default()
        });
        // User already has 2 running
        let result = check_qos_limits(&job, &qos, 2, 2, &TresRecord::new());
        assert_eq!(result, QosCheckResult::Blocked(PendingReason::QoSMaxJobsPerUser));
    }

    #[test]
    fn t21_12_qos_max_wall_blocked() {
        let qos = Qos {
            name: "short".into(),
            limits: QosLimits {
                max_wall_minutes: Some(60),
                ..Default::default()
            },
            ..Default::default()
        };
        let job = Job::new(1, JobSpec {
            name: "test".into(),
            user: "alice".into(),
            time_limit: Some(chrono::Duration::hours(4)),
            ..Default::default()
        });
        let result = check_qos_limits(&job, &qos, 0, 0, &TresRecord::new());
        assert_eq!(result, QosCheckResult::Blocked(PendingReason::PartitionTimeLimit));
    }

    #[test]
    fn t21_13_qos_tres_per_job_blocked() {
        let mut max_tres = TresRecord::new();
        max_tres.set(TresType::Cpu, 8);
        let qos = Qos {
            name: "small".into(),
            limits: QosLimits {
                max_tres_per_job: Some(max_tres),
                ..Default::default()
            },
            ..Default::default()
        };
        let job = Job::new(1, JobSpec {
            name: "big".into(),
            user: "alice".into(),
            num_tasks: 16,
            cpus_per_task: 1,
            ..Default::default()
        });
        let result = check_qos_limits(&job, &qos, 0, 0, &TresRecord::new());
        assert_eq!(result, QosCheckResult::Blocked(PendingReason::Resources));
    }

    // ── T21.14: QOS priority adjustment ──────────────────────────

    #[test]
    fn t21_14_qos_priority_boost() {
        let qos = Qos { priority: 1000, ..Default::default() };
        assert_eq!(qos_adjusted_priority(500, &qos), 1500);
    }

    #[test]
    fn t21_15_qos_priority_penalty() {
        let qos = Qos { priority: -200, ..Default::default() };
        assert_eq!(qos_adjusted_priority(1000, &qos), 800);
    }

    #[test]
    fn t21_16_qos_priority_floor() {
        let qos = Qos { priority: -5000, ..Default::default() };
        assert_eq!(qos_adjusted_priority(1000, &qos), 1);
    }
}
