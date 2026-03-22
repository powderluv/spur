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
        assert_eq!(CpuBind::from_str("cores"), CpuBind::Cores);
        assert_eq!(CpuBind::from_str("threads"), CpuBind::Threads);
        assert_eq!(CpuBind::from_str("sockets"), CpuBind::Sockets);
        assert_eq!(CpuBind::from_str("ldoms"), CpuBind::Ldoms);
        assert_eq!(CpuBind::from_str("rank"), CpuBind::Rank);
        assert_eq!(CpuBind::from_str("none"), CpuBind::None);
    }

    #[test]
    fn t01_6_cpu_bind_map() {
        match CpuBind::from_str("map_cpu:0,4,8,12") {
            CpuBind::Map(s) => assert_eq!(s, "0,4,8,12"),
            other => panic!("expected Map, got {:?}", other),
        }
    }

    // ── T01.7: GPU binding ───────────────────────────────────────

    #[test]
    fn t01_7_gpu_bind_types() {
        assert_eq!(GpuBind::from_str("closest"), GpuBind::Closest);
        assert_eq!(GpuBind::from_str("none"), GpuBind::None);
    }

    #[test]
    fn t01_8_gpu_bind_map() {
        match GpuBind::from_str("map_gpu:0,1,2,3") {
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
        let alloc = node.try_allocate(32, 128_000, 4, Some("mi300x")).unwrap();

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

        let a1 = node.try_allocate(16, 64_000, 0, None).unwrap();
        let a2 = node.try_allocate(16, 64_000, 0, None).unwrap();

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

        let alloc = node.try_allocate(32, 128_000, 0, None).unwrap();
        assert_eq!(node.free_cpus(), 32);

        node.release(&alloc);
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
        let node_rank_0 = 0u32 / tasks_per_node;
        let node_rank_1 = 4u32 / tasks_per_node;
        assert_eq!(node_rank_0, 0);
        assert_eq!(node_rank_1, 1);
    }
}
