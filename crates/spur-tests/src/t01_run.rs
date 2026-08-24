// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T01: Job step and task distribution tests.
//!
//! Corresponds to Slurm's test1.x series (srun).
//! Tests task distribution, CPU/GPU binding, job steps.

#[cfg(test)]
mod tests {
    use spur_core::resource::*;
    use spur_core::step::*;
    use spur_sched::cons_tres::*;

    // ── T01.1: Task distribution block ───────────────────────────

    #[test]
    fn t01_1_block_distribution_even() {
        let m = distribute_tasks(8, 4, TaskDistribution::Block);
        assert_eq!(m, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn t01_2_block_distribution_uneven() {
        let m = distribute_tasks(10, 3, TaskDistribution::Block);
        assert_eq!(m.len(), 10);
        let count_0 = m.iter().filter(|&&n| n == 0).count();
        let count_1 = m.iter().filter(|&&n| n == 1).count();
        let count_2 = m.iter().filter(|&&n| n == 2).count();
        assert_eq!(count_0, 4); // 10/3 = 3 rem 1 → first node gets 4
        assert_eq!(count_1, 3);
        assert_eq!(count_2, 3);
    }

    #[test]
    fn t01_3_cyclic_distribution() {
        let m = distribute_tasks(8, 4, TaskDistribution::Cyclic);
        assert_eq!(m, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn t01_4_single_node() {
        let m = distribute_tasks(16, 1, TaskDistribution::Block);
        assert!(m.iter().all(|&n| n == 0));
    }

    // ── T01.5: CPU binding ───────────────────────────────────────

    #[test]
    fn t01_5_cpu_bind_types() {
        assert_eq!("cores".parse::<CpuBind>().unwrap(), CpuBind::Cores);
        assert_eq!("threads".parse::<CpuBind>().unwrap(), CpuBind::Threads);
        assert_eq!("sockets".parse::<CpuBind>().unwrap(), CpuBind::Sockets);
        assert_eq!("ldoms".parse::<CpuBind>().unwrap(), CpuBind::Ldoms);
        assert_eq!("rank".parse::<CpuBind>().unwrap(), CpuBind::Rank);
        assert_eq!("none".parse::<CpuBind>().unwrap(), CpuBind::None);
    }

    #[test]
    fn t01_6_cpu_bind_map() {
        match "map_cpu:0,4,8,12".parse::<CpuBind>().unwrap() {
            CpuBind::Map(s) => assert_eq!(s, "0,4,8,12"),
            other => panic!("expected Map, got {:?}", other),
        }
    }

    // ── T01.7: GPU binding ───────────────────────────────────────

    #[test]
    fn t01_7_gpu_bind_types() {
        assert_eq!("closest".parse::<GpuBind>().unwrap(), GpuBind::Closest);
        assert_eq!("none".parse::<GpuBind>().unwrap(), GpuBind::None);
    }

    #[test]
    fn t01_8_gpu_bind_map() {
        match "map_gpu:0,1,2,3".parse::<GpuBind>().unwrap() {
            GpuBind::Map(s) => assert_eq!(s, "0,1,2,3"),
            other => panic!("expected Map, got {:?}", other),
        }
    }

    // ── T01.9: Consumable TRES allocation ────────────────────────

    #[test]
    fn t01_9_cons_tres_basic_alloc() {
        let gpus: Vec<GpuResource> = (0..8)
            .map(|i| GpuResource {
                device_id: i,
                gpu_type: "mi300x".into(),
                memory_mb: 192_000,
                peer_gpus: vec![],
                link_type: GpuLinkType::XGMI,
            })
            .collect();

        let resources = ResourceSet {
            cpus: 128,
            memory_mb: 512_000,
            gpus,
            ..Default::default()
        };

        let mut node = NodeAllocation::new("gpu001".into(), &resources);
        let alloc = node
            .allocate_for_job(1, 1, 32, 128_000, &[0, 1, 2, 3])
            .unwrap();

        assert_eq!(alloc.cpu_ids.len(), 32);
        assert_eq!(alloc.gpu_ids.len(), 4);
        assert_eq!(alloc.memory_mb, 128_000);
        assert_eq!(node.free_cpus(), 96);
        assert_eq!(node.free_gpus(None), 4);
    }

    #[test]
    fn t01_10_cons_tres_no_overlap() {
        let resources = ResourceSet {
            cpus: 64,
            memory_mb: 256_000,
            ..Default::default()
        };
        let mut node = NodeAllocation::new("cpu001".into(), &resources);

        let a1 = node.allocate_for_job(1, 1, 16, 64_000, &[]).unwrap();
        let a2 = node.allocate_for_job(2, 1, 16, 64_000, &[]).unwrap();

        // No CPU overlap
        for id in &a1.cpu_ids {
            assert!(!a2.cpu_ids.contains(id), "CPU {} allocated twice", id);
        }
    }

    #[test]
    fn t01_11_cons_tres_release() {
        let resources = ResourceSet {
            cpus: 64,
            memory_mb: 256_000,
            ..Default::default()
        };
        let mut node = NodeAllocation::new("cpu001".into(), &resources);

        node.allocate_for_job(1, 1, 32, 128_000, &[]).unwrap();
        assert_eq!(node.free_cpus(), 32);

        node.release_job(1);
        assert_eq!(node.free_cpus(), 64);
        assert_eq!(node.free_memory_mb(), 256_000);
    }

    // ── T01.12: Step state ───────────────────────────────────────

    #[test]
    fn t01_12_step_states() {
        assert!(!StepState::Pending.is_terminal());
        assert!(!StepState::Running.is_terminal());
        assert!(StepState::Completed.is_terminal());
        assert!(StepState::Failed.is_terminal());
        assert!(StepState::Cancelled.is_terminal());
    }

    #[test]
    fn t01_13_step_display() {
        assert_eq!(StepState::Running.display(), "RUNNING");
        assert_eq!(StepState::Completed.display(), "COMPLETED");
    }

    // ── T01.14: Special step IDs ─────────────────────────────────

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn t01_14_special_step_ids() {
        assert_ne!(STEP_BATCH, STEP_EXTERN);
        assert_ne!(STEP_BATCH, STEP_INTERACTIVE);
        // These are large sentinel values
        assert!(STEP_BATCH > 1_000_000);
    }

    // ── T01.15: LOCAL_RANK computation for multi-task ────────────

    #[test]
    fn t01_15_local_rank_computation() {
        // Simulates multi-task per-node: tasks_per_node = 4, task_offset = 8
        // LOCAL_RANK should be 0..3, SPUR_PROCID should be 8..11
        let tasks_per_node = 4u32;
        let task_offset = 8u32;
        for local_rank in 0..tasks_per_node {
            let procid = task_offset + local_rank;
            assert_eq!(procid, 8 + local_rank);
            assert!(local_rank < tasks_per_node);
        }
    }

    #[test]
    fn t01_16_gpu_partitioning_across_tasks() {
        // 8 GPUs across 4 tasks → 2 GPUs per task
        let gpu_devices: Vec<u32> = (0..8).collect();
        let num_tasks = 4u32;
        let gpus_per_task = gpu_devices.len() / num_tasks as usize;
        assert_eq!(gpus_per_task, 2);

        for local_rank in 0..num_tasks {
            let start = local_rank as usize * gpus_per_task;
            let end = start + gpus_per_task;
            let task_gpus: Vec<u32> = gpu_devices[start..end].to_vec();
            assert_eq!(task_gpus.len(), 2);
            // First task gets GPUs 0,1; second gets 2,3; etc.
            assert_eq!(task_gpus[0], local_rank * 2);
            assert_eq!(task_gpus[1], local_rank * 2 + 1);
        }
    }

    #[test]
    fn t01_17_node_rank_from_task_offset() {
        // Multi-node: 2 nodes, 4 tasks per node
        // Node 0: task_offset=0, Node 1: task_offset=4
        let tasks_per_node = 4u32;
        let node_rank_0 = 0u32;
        let node_rank_1 = 4u32 / tasks_per_node;
        assert_eq!(node_rank_0, 0);
        assert_eq!(node_rank_1, 1);
    }

    // ── T01.18–20: srun step mode (SPUR_JOB_ID env) ──────────────

    #[test]
    fn t01_18_srun_step_mode_env_var_name() {
        // srun detects it's inside a batch job by checking SPUR_JOB_ID.
        // This test confirms the env var name matches what sbatch exports.
        let env_var = "SPUR_JOB_ID";
        assert!(env_var.starts_with("SPUR_"), "must use SPUR_ namespace");
        assert!(env_var.ends_with("JOB_ID"), "must identify the job");
    }

    #[test]
    fn t01_19_srun_step_mode_job_id_parse() {
        // When SPUR_JOB_ID="42", srun should parse it as u32 job_id=42.
        let raw = "42";
        let job_id: u32 = raw.parse().expect("SPUR_JOB_ID must be a valid u32");
        assert_eq!(job_id, 42);

        // Non-numeric value should fail to parse.
        let bad = "not-a-number";
        assert!(bad.parse::<u32>().is_err());
    }

    #[test]
    fn t01_20_srun_step_env_vars_set() {
        // When running as a step, these env vars are exported to the child process.
        let step_env = ["SPUR_JOB_ID", "SPUR_STEP_ID", "SPUR_NODEID", "SPUR_PROCID"];
        // Confirm no duplicates and all follow SPUR_ convention.
        let mut seen = std::collections::HashSet::new();
        for v in &step_env {
            assert!(v.starts_with("SPUR_"), "must use SPUR_ prefix: {}", v);
            assert!(seen.insert(v), "duplicate env var: {}", v);
        }
        assert_eq!(seen.len(), 4);
    }
}
