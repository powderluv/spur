// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Consumable TRES resource selection.
//!
//! Tracks resources at core/socket/GPU granularity within a node.
//! This is the equivalent of Slurm's select/cons_tres plugin.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use spur_core::resource::{GpuResource, ResourceSet};

/// Why a reservation could not be made. Distinguished so the caller can map
/// each to the right gRPC status instead of reporting every failure as GPU
/// exhaustion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    /// A controller-allocated GPU is unknown to this node or already in use.
    GpusUnavailable,
    /// A LaunchJob is already in flight for this job id (its reservation is
    /// still mid-launch, not yet committed or released). A second concurrent
    /// launch would double-count resources, so it is rejected.
    DuplicateJob,
    /// A newer run_attempt already owns this job id's reservation; this
    /// (older or duplicate) attempt lost the race and must not proceed.
    Superseded,
}

/// Per-node resource allocation state.
/// Tracks which specific cores and GPUs are allocated.
#[derive(Debug, Clone)]
pub struct NodeAllocation {
    pub node_name: String,
    /// Total CPUs on this node.
    pub total_cpus: u32,
    /// Bitmap of allocated CPUs (bit N = core N).
    pub allocated_cpus: Vec<bool>,
    /// Total memory in MB.
    pub total_memory_mb: u64,
    /// Allocated memory in MB.
    pub allocated_memory_mb: u64,
    /// GPU allocations: device_id → allocated?
    pub gpu_allocated: Vec<bool>,
    /// GPU info for type matching.
    pub gpus: Vec<GpuResource>,
    /// Per-job ownership, so an allocation can be released by job id and
    /// orphans reconciled. Source of truth; the bitmaps above are a derived
    /// index for fast free-count queries.
    owners: HashMap<u32, Owned>,
    /// Reserved but not yet committed, with reserve time. Reconcile spares
    /// these until they exceed a TTL so a dropped launch can't pin resources.
    launching: HashMap<u32, Instant>,
}

impl NodeAllocation {
    pub fn new(name: String, resources: &ResourceSet) -> Self {
        let num_gpus = resources.gpus.len();
        Self {
            node_name: name,
            total_cpus: resources.cpus,
            allocated_cpus: vec![false; resources.cpus as usize],
            total_memory_mb: resources.memory_mb,
            allocated_memory_mb: 0,
            gpu_allocated: vec![false; num_gpus],
            gpus: resources.gpus.clone(),
            owners: HashMap::new(),
            launching: HashMap::new(),
        }
    }

    /// Available (unallocated) CPU count.
    pub fn free_cpus(&self) -> u32 {
        self.allocated_cpus.iter().filter(|&&a| !a).count() as u32
    }

    /// Available memory.
    pub fn free_memory_mb(&self) -> u64 {
        self.total_memory_mb
            .saturating_sub(self.allocated_memory_mb)
    }

    /// Device ids of all currently-allocated GPUs (for diagnostics).
    pub fn allocated_gpu_ids(&self) -> Vec<u32> {
        self.gpu_allocated
            .iter()
            .enumerate()
            .filter(|(_, &a)| a)
            .filter_map(|(i, _)| self.gpus.get(i).map(|g| g.device_id))
            .collect()
    }

    /// Job ids owning any of `device_ids`, excluding mid-launch owners (a launch
    /// in flight is a real duplicate, not a reclaimable stale owner).
    pub fn conflicting_owners(&self, device_ids: &[u32]) -> Vec<u32> {
        self.owners
            .iter()
            .filter(|(id, _)| !self.launching.contains_key(id))
            .filter(|(_, owned)| owned.result.gpu_ids.iter().any(|g| device_ids.contains(g)))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Available GPU count (optionally filtered by type).
    pub fn free_gpus(&self, gpu_type: Option<&str>) -> u32 {
        self.gpu_allocated
            .iter()
            .enumerate()
            .filter(|(i, allocated)| {
                if **allocated {
                    return false;
                }
                if let Some(gtype) = gpu_type {
                    if gtype != "any"
                        && self.gpus.get(*i).map(|g| g.gpu_type.as_str()) != Some(gtype)
                    {
                        return false;
                    }
                }
                true
            })
            .count() as u32
    }

    /// Free the bitmaps/counters an allocation held. Internal helper: callers
    /// go through `release_job` so per-job ownership stays consistent. GPU ids
    /// are device ids, matched against the node's GPU table the same way
    /// `allocate_for_job` records them.
    fn release(&mut self, alloc: &AllocationResult) {
        for &cpu in &alloc.cpu_ids {
            if let Some(a) = self.allocated_cpus.get_mut(cpu as usize) {
                *a = false;
            }
        }
        self.allocated_memory_mb = self.allocated_memory_mb.saturating_sub(alloc.memory_mb);
        for &device_id in &alloc.gpu_ids {
            if let Some(idx) = self.gpus.iter().position(|g| g.device_id == device_id) {
                self.gpu_allocated[idx] = false;
            }
        }
    }

    /// Reserve resources for a job, keyed by job id. GPU device ids are the
    /// hard gate (unknown or in-use → `GpusUnavailable`); a launch already in
    /// flight for the same job id → `DuplicateJob`. CPU is best-effort since the
    /// controller owns placement. Memory is always accounted so release stays
    /// symmetric. Marked `launching` until `commit_job`/`release_job` so
    /// reconcile spares an in-flight launch. `run_attempt` is recorded so a
    /// later `commit_job` for a different (superseded) attempt is rejected.
    pub fn allocate_for_job(
        &mut self,
        job_id: u32,
        run_attempt: u32,
        cpus: u32,
        memory_mb: u64,
        gpu_device_ids: &[u32],
    ) -> Result<AllocationResult, AllocError> {
        // A launch still in flight (reserved, not yet committed or released) is
        // a genuine concurrent duplicate: a second launch would double-count.
        if self.launching.contains_key(&job_id) {
            return Err(AllocError::DuplicateJob);
        }
        // A committed owner from the same or an older attempt is stale (a prior
        // run's teardown hasn't released yet) — supersede it. A NEWER attempt's
        // owner already won this job id; a late reservation must not clobber it.
        if let Some(existing) = self.owners.get(&job_id) {
            if existing.run_attempt > run_attempt {
                return Err(AllocError::Superseded);
            }
        }
        self.release_job(job_id);

        let mut gpu_indices = Vec::with_capacity(gpu_device_ids.len());
        for &id in gpu_device_ids {
            let idx = self
                .gpus
                .iter()
                .position(|g| g.device_id == id)
                .ok_or(AllocError::GpusUnavailable)?;
            if self.gpu_allocated[idx] || gpu_indices.contains(&idx) {
                return Err(AllocError::GpusUnavailable);
            }
            gpu_indices.push(idx);
        }

        let mut cpu_ids = Vec::new();
        for (i, allocated) in self.allocated_cpus.iter_mut().enumerate() {
            if !*allocated && cpu_ids.len() < cpus as usize {
                *allocated = true;
                cpu_ids.push(i as u32);
            }
        }

        self.allocated_memory_mb += memory_mb;
        for &idx in &gpu_indices {
            self.gpu_allocated[idx] = true;
        }

        let result = AllocationResult {
            cpu_ids,
            gpu_ids: gpu_device_ids.to_vec(),
            memory_mb,
        };
        self.owners.insert(
            job_id,
            Owned {
                run_attempt,
                result: result.clone(),
            },
        );
        self.launching.insert(job_id, Instant::now());
        Ok(result)
    }

    /// Mark a job's allocation as committed (its process is now tracked), so it
    /// is no longer exempt from reconcile. Returns false if the reservation no
    /// longer exists (reconcile reclaimed it after the launch exceeded the TTL)
    /// or now belongs to a different `run_attempt` (a newer reservation for the
    /// same job id superseded this one) — either way the caller must not treat
    /// the job as backed by an allocation.
    pub fn commit_job(&mut self, job_id: u32, run_attempt: u32) -> bool {
        self.launching.remove(&job_id);
        self.owners
            .get(&job_id)
            .is_some_and(|owned| owned.run_attempt == run_attempt)
    }

    /// Release a job's allocation by id, regardless of which attempt owns it.
    /// Idempotent: releasing an unknown or already-released job is a no-op
    /// returning false. Only for callers that genuinely don't have a specific
    /// attempt to compare (reconcile, foreign-owner reclaim) — everyone else
    /// should use `release_job_if`.
    pub fn release_job(&mut self, job_id: u32) -> bool {
        self.launching.remove(&job_id);
        let Some(owned) = self.owners.remove(&job_id) else {
            return false;
        };
        self.release(&owned.result);
        true
    }

    /// Release a job's allocation only if `run_attempt` is still the current
    /// owner, so a stale caller can't free a different, newer attempt's
    /// reservation that has since superseded the one it thinks it owns.
    pub fn release_job_if(&mut self, job_id: u32, run_attempt: u32) -> bool {
        if self
            .owners
            .get(&job_id)
            .is_some_and(|owned| owned.run_attempt == run_attempt)
        {
            self.release_job(job_id)
        } else {
            false
        }
    }

    /// Release owned allocations whose job is neither live nor launching within
    /// `launching_ttl`, returning the reclaimed ids. Recovers a failed teardown
    /// or a dropped launch instead of stranding the node until spurd restart.
    pub fn reconcile(
        &mut self,
        live: &HashSet<u32>,
        now: Instant,
        launching_ttl: Duration,
    ) -> Vec<u32> {
        let orphaned: Vec<u32> = self
            .owners
            .keys()
            .copied()
            .filter(|id| {
                if live.contains(id) {
                    return false;
                }
                match self.launching.get(id) {
                    Some(reserved_at) => {
                        now.saturating_duration_since(*reserved_at) >= launching_ttl
                    }
                    None => true,
                }
            })
            .collect();
        for &id in &orphaned {
            self.release_job(id);
        }
        orphaned
    }
}

/// An owned reservation, tagged with the attempt that created it so a later
/// operation for a superseded attempt can be told apart from the current one.
#[derive(Debug, Clone)]
struct Owned {
    run_attempt: u32,
    result: AllocationResult,
}

/// Result of a successful allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationResult {
    /// Allocated core IDs.
    pub cpu_ids: Vec<u32>,
    /// Allocated GPU device IDs.
    pub gpu_ids: Vec<u32>,
    /// Allocated memory in MB.
    pub memory_mb: u64,
}

impl AllocationResult {
    /// Format CPU IDs as a taskset-compatible string.
    pub fn cpu_list(&self) -> String {
        self.cpu_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Format GPU IDs as ROCR_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES string.
    pub fn gpu_list(&self) -> String {
        self.gpu_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spur_core::resource::GpuLinkType;

    fn make_node(cpus: u32, mem: u64, num_gpus: usize, gpu_type: &str) -> NodeAllocation {
        make_node_with_ids(cpus, mem, (0..num_gpus as u32).collect(), gpu_type)
    }

    fn make_node_with_ids(
        cpus: u32,
        mem: u64,
        device_ids: Vec<u32>,
        gpu_type: &str,
    ) -> NodeAllocation {
        let gpus: Vec<GpuResource> = device_ids
            .into_iter()
            .map(|device_id| GpuResource {
                device_id,
                gpu_type: gpu_type.into(),
                memory_mb: 192_000,
                peer_gpus: vec![],
                link_type: GpuLinkType::XGMI,
            })
            .collect();

        let resources = ResourceSet {
            cpus,
            memory_mb: mem,
            gpus,
            ..Default::default()
        };

        NodeAllocation::new("node001".into(), &resources)
    }

    #[test]
    fn test_initial_state() {
        let node = make_node(64, 256_000, 8, "mi300x");
        assert_eq!(node.free_cpus(), 64);
        assert_eq!(node.free_memory_mb(), 256_000);
        assert_eq!(node.free_gpus(None), 8);
    }

    #[test]
    fn test_allocate_cpus() {
        let mut node = make_node(64, 256_000, 0, "");
        let alloc = node.allocate_for_job(1, 1, 16, 0, &[]).unwrap();
        assert_eq!(alloc.cpu_ids.len(), 16);
        assert_eq!(node.free_cpus(), 48);
    }

    #[test]
    fn test_allocate_gpus_by_device_id() {
        let mut node = make_node(64, 256_000, 8, "mi300x");
        let alloc = node.allocate_for_job(1, 1, 0, 0, &[0, 1, 2, 3]).unwrap();
        assert_eq!(alloc.gpu_ids, vec![0, 1, 2, 3]);
        assert_eq!(node.free_gpus(None), 4);
    }

    #[test]
    fn test_record_then_release_noncontiguous_device_ids() {
        // Real nodes expose GPUs whose device_id != vec position (DRM render
        // ids 128..135). Release keys by device_id, so a released device must
        // return to the free pool or it is rejected forever.
        let mut node = make_node_with_ids(64, 256_000, vec![128, 129, 130, 131], "mi350x");

        assert!(node.allocate_for_job(1, 1, 0, 0, &[129, 131]).is_ok());
        assert_eq!(node.free_gpus(None), 2);

        assert!(node.release_job(1));
        assert_eq!(
            node.free_gpus(None),
            4,
            "released GPUs must return to the free pool"
        );

        // The whole point: the device is re-allocatable after release.
        assert!(
            node.allocate_for_job(2, 1, 0, 0, &[129, 131]).is_ok(),
            "device_ids must be re-allocatable after release"
        );
    }

    #[test]
    fn test_allocate_rejects_unknown_device_id() {
        let mut node = make_node_with_ids(64, 256_000, vec![128, 129], "mi350x");
        assert!(node.allocate_for_job(1, 1, 0, 0, &[200]).is_err());
        // A rejected allocation must not leave partial state behind.
        assert_eq!(node.free_gpus(None), 2);
        assert!(!node.release_job(1));
    }

    #[test]
    fn test_allocate_rolls_back_on_partial_gpu_conflict() {
        // If a multi-GPU allocation hits a conflict partway, it must not leave
        // the earlier device_ids marked allocated — that is itself a leak.
        let mut node = make_node_with_ids(64, 256_000, vec![128, 129, 130], "mi350x");
        assert!(node.allocate_for_job(1, 1, 0, 0, &[129]).is_ok());
        // [128, 129] — 129 already taken, so the whole call must fail and 128
        // must remain free.
        assert!(node.allocate_for_job(2, 1, 0, 0, &[128, 129]).is_err());
        assert!(
            node.allocate_for_job(3, 1, 0, 0, &[128]).is_ok(),
            "128 must remain free after the failed partial allocation"
        );
    }

    #[test]
    fn test_allocate_for_job_release_by_id() {
        let mut node = make_node_with_ids(64, 256_000, vec![128, 129, 130, 131], "mi350x");
        let alloc = node
            .allocate_for_job(1, 1, 8, 32_000, &[129, 131])
            .expect("allocation should succeed");
        assert_eq!(alloc.gpu_ids, vec![129, 131]);
        assert_eq!(node.free_cpus(), 56);
        assert_eq!(node.free_gpus(None), 2);
        assert_eq!(node.free_memory_mb(), 224_000);

        assert!(node.release_job(1));
        assert_eq!(node.free_cpus(), 64);
        assert_eq!(node.free_gpus(None), 4);
        assert_eq!(node.free_memory_mb(), 256_000);
        // Idempotent: releasing again is a no-op.
        assert!(!node.release_job(1));
    }

    #[test]
    fn test_allocate_for_job_rejects_in_flight_duplicate() {
        // A second launch while the first is still in flight (reserved, not yet
        // committed or released) is a genuine concurrent duplicate: rejecting it
        // avoids double-counting CPU/mem and orphaning the prior owner entry.
        let mut node = make_node(64, 256_000, 0, "");
        assert!(node.allocate_for_job(1, 1, 8, 16_000, &[]).is_ok());
        assert_eq!(
            node.allocate_for_job(1, 1, 8, 16_000, &[]),
            Err(AllocError::DuplicateJob)
        );
        assert_eq!(node.free_cpus(), 56);
        assert_eq!(node.free_memory_mb(), 240_000);
        // After release the id is free to reserve again.
        assert!(node.release_job(1));
        assert!(node.allocate_for_job(1, 1, 8, 16_000, &[]).is_ok());
    }

    #[test]
    fn test_allocate_for_job_supersedes_stale_committed_reservation() {
        // A committed reservation whose teardown has not released yet (e.g. a
        // preempted-then-requeued job re-dispatched under the same id before the
        // agent reaped the killed process) must be superseded, not rejected —
        // otherwise the legitimate re-launch fails and the job never runs. The
        // stale reservation is released first, so resources are not double-counted.
        let mut node = make_node(64, 256_000, 0, "");
        node.allocate_for_job(7, 1, 8, 16_000, &[]).unwrap();
        node.commit_job(7, 1); // prior run committed, then preempted (not released).
        assert_eq!(node.free_cpus(), 56);

        // Re-dispatch under the same id succeeds and does not double-count.
        let alloc = node
            .allocate_for_job(7, 1, 8, 16_000, &[])
            .expect("re-dispatch of a stale committed job id must succeed");
        assert_eq!(alloc.cpu_ids.len(), 8);
        assert_eq!(node.free_cpus(), 56);
        assert_eq!(node.free_memory_mb(), 240_000);
        // Exactly one owner entry remains, and the fresh reservation is again
        // treated as in-flight until it commits or releases.
        assert_eq!(
            node.allocate_for_job(7, 1, 8, 16_000, &[]),
            Err(AllocError::DuplicateJob)
        );
    }

    #[test]
    fn test_allocate_for_job_rejects_conflicting_gpu() {
        let mut node = make_node_with_ids(64, 256_000, vec![0, 1], "mi300x");
        assert!(node.allocate_for_job(1, 1, 4, 0, &[0]).is_ok());
        // Second job wanting the same device id must fail with no state change.
        assert_eq!(
            node.allocate_for_job(2, 1, 4, 0, &[0]),
            Err(AllocError::GpusUnavailable)
        );
        assert_eq!(node.free_gpus(None), 1);
        // The failed job left no owner entry.
        assert!(!node.release_job(2));
    }

    #[test]
    fn test_conflicting_owners_reports_committed_but_not_launching() {
        let mut node = make_node_with_ids(64, 256_000, vec![0, 1, 2, 3], "mi300x");
        // job 1: committed, owns GPUs 0 and 1.
        node.allocate_for_job(1, 1, 4, 8_000, &[0, 1]).unwrap();
        node.commit_job(1, 1);
        // job 2: still launching (reserved, not committed), owns GPU 2.
        node.allocate_for_job(2, 1, 4, 8_000, &[2]).unwrap();

        // A new dispatch onto GPU 1 conflicts with committed job 1.
        assert_eq!(node.conflicting_owners(&[1]), vec![1]);
        // A dispatch onto GPU 3 (free) conflicts with nobody.
        assert!(node.conflicting_owners(&[3]).is_empty());
        // GPU 2's owner is still launching — a real duplicate, never reported.
        assert!(node.conflicting_owners(&[2]).is_empty());
        // A dispatch spanning a committed and a launching device reports only the
        // committed owner.
        assert_eq!(node.conflicting_owners(&[1, 2]), vec![1]);
    }

    #[test]
    fn test_reconcile_reclaims_orphans_but_spares_live_and_launching() {
        let mut node = make_node_with_ids(64, 256_000, vec![0, 1, 2, 3], "mi300x");
        // job 1: committed and live.
        node.allocate_for_job(1, 1, 4, 8_000, &[0]).unwrap();
        node.commit_job(1, 1);
        // job 2: committed but NOT live (teardown failed to release — orphan).
        node.allocate_for_job(2, 1, 4, 8_000, &[1]).unwrap();
        node.commit_job(2, 1);
        // job 3: launching within TTL — spared.
        node.allocate_for_job(3, 1, 4, 8_000, &[2]).unwrap();

        let live: HashSet<u32> = [1].into_iter().collect();
        let ttl = Duration::from_secs(120);
        let mut reclaimed = node.reconcile(&live, Instant::now(), ttl);
        reclaimed.sort();
        assert_eq!(reclaimed, vec![2], "only the orphan (job 2) is reclaimed");

        assert!(!node.release_job(2), "job 2 already reconciled");
        assert!(node.release_job(1));
        assert!(node.release_job(3));
        assert_eq!(node.free_gpus(None), 4);
    }

    #[test]
    fn test_reconcile_reclaims_launching_past_ttl() {
        let mut node = make_node_with_ids(64, 256_000, vec![0, 1, 2, 3], "mi300x");
        // A launch reserved but never committed must be reclaimed past the TTL.
        node.allocate_for_job(1, 1, 4, 8_000, &[0]).unwrap();

        let live: HashSet<u32> = HashSet::new();
        let ttl = Duration::from_secs(120);

        assert!(node.reconcile(&live, Instant::now(), ttl).is_empty());
        assert_eq!(node.free_gpus(None), 3, "still reserved within TTL");

        let future = Instant::now() + Duration::from_secs(121);
        assert_eq!(node.reconcile(&live, future, ttl), vec![1]);
        assert_eq!(node.free_gpus(None), 4, "reclaimed after TTL");
    }

    #[test]
    fn test_commit_job_reports_reclaimed_reservation() {
        let mut node = make_node_with_ids(64, 256_000, vec![0, 1], "mi300x");
        node.allocate_for_job(1, 1, 4, 8_000, &[0]).unwrap();
        assert!(
            node.commit_job(1, 1),
            "commit of a live reservation returns true"
        );

        // A launch whose reservation reconcile reclaimed before commit: the
        // owner is gone, so commit must report false and stay a no-op. Keep
        // job 1 in the live set so only the past-TTL launch (job 2) is reclaimed.
        node.allocate_for_job(2, 1, 4, 8_000, &[1]).unwrap();
        let live: HashSet<u32> = [1].into_iter().collect();
        let past = Instant::now() + Duration::from_secs(121);
        node.reconcile(&live, past, Duration::from_secs(120));
        assert!(
            !node.commit_job(2, 1),
            "commit of a reclaimed reservation returns false"
        );
        assert_eq!(
            node.free_gpus(None),
            1,
            "reclaimed GPU stays free after the no-op commit"
        );
    }

    #[test]
    fn test_commit_job_rejects_a_superseded_attempt() {
        // Attempt 1 reserves job 7, then something (a cancel, or the TTL
        // reconcile) releases it believing it's gone before it ever commits —
        // freeing the job id for a redispatch (attempt 2) to reserve and
        // commit. Attempt 1, unaware it was superseded, must not have its own,
        // now-stale commit adopt attempt 2's reservation as its own.
        let mut node = make_node(64, 256_000, 0, "");
        node.allocate_for_job(7, 1, 8, 16_000, &[]).unwrap();
        node.release_job(7);
        node.allocate_for_job(7, 2, 8, 16_000, &[]).unwrap();
        assert!(node.commit_job(7, 2), "the current attempt commits");

        assert!(
            !node.commit_job(7, 1),
            "a superseded attempt's late commit must not succeed"
        );
        assert_eq!(
            node.free_cpus(),
            56,
            "the current attempt's resources must remain allocated"
        );
    }

    #[test]
    fn test_allocate_for_job_rejects_a_stale_attempt_superseding_a_committed_newer_one() {
        // Attempt 2 reserves and commits job 7. A late/duplicate LaunchJob for
        // the older attempt 1 must not be allowed to supersede it — that would
        // hand attempt 2's live, running resources to a stale relaunch.
        let mut node = make_node(64, 256_000, 0, "");
        node.allocate_for_job(7, 2, 8, 16_000, &[]).unwrap();
        assert!(node.commit_job(7, 2));

        assert_eq!(
            node.allocate_for_job(7, 1, 8, 16_000, &[]),
            Err(AllocError::Superseded)
        );
        assert_eq!(
            node.free_cpus(),
            56,
            "the newer attempt's resources must remain allocated"
        );
    }

    #[test]
    fn test_allocate_for_job_supersedes_a_committed_same_or_older_attempt() {
        // A same-attempt retry, or a genuinely newer attempt reserving after a
        // stale committed owner, must still supersede as before.
        let mut node = make_node(64, 256_000, 0, "");
        node.allocate_for_job(7, 1, 8, 16_000, &[]).unwrap();
        assert!(node.commit_job(7, 1));

        assert!(node.allocate_for_job(7, 2, 8, 16_000, &[]).is_ok());
        assert_eq!(node.free_cpus(), 56);
    }

    #[test]
    fn test_release_job_if_spares_a_reused_job_ids_reservation() {
        let mut node = make_node(64, 256_000, 0, "");
        node.allocate_for_job(7, 1, 8, 16_000, &[]).unwrap();
        node.release_job(7);
        node.allocate_for_job(7, 2, 8, 16_000, &[]).unwrap();

        assert!(
            !node.release_job_if(7, 1),
            "a stale attempt must not release a different, current attempt's reservation"
        );
        assert_eq!(node.free_cpus(), 56);
        assert!(
            node.release_job_if(7, 2),
            "the current attempt can release its own"
        );
        assert_eq!(node.free_cpus(), 64);
    }

    #[test]
    fn test_memory_released_symmetrically_with_zero_cpus() {
        // A job with 0 cpus must still have its memory reserved and released
        // symmetrically, or release drives allocated_memory_mb below what was
        // added.
        let mut node = make_node(64, 256_000, 0, "");
        node.allocate_for_job(1, 1, 0, 16_000, &[]).unwrap();
        assert_eq!(node.free_memory_mb(), 240_000);
        node.release_job(1);
        assert_eq!(node.free_memory_mb(), 256_000);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut node = make_node(64, 256_000, 8, "mi300x");
        let a1 = node.allocate_for_job(1, 1, 16, 64_000, &[0, 1]).unwrap();
        let a2 = node.allocate_for_job(2, 1, 16, 64_000, &[2, 3]).unwrap();
        assert_eq!(node.free_cpus(), 32);
        assert_eq!(node.free_gpus(None), 4);

        // CPU IDs should not overlap
        let overlap: Vec<_> = a1
            .cpu_ids
            .iter()
            .filter(|id| a2.cpu_ids.contains(id))
            .collect();
        assert!(overlap.is_empty());
    }

    #[test]
    fn test_allocation_result_format() {
        let alloc = AllocationResult {
            cpu_ids: vec![0, 1, 2, 3],
            gpu_ids: vec![0, 1],
            memory_mb: 128_000,
        };
        assert_eq!(alloc.cpu_list(), "0,1,2,3");
        assert_eq!(alloc.gpu_list(), "0,1");
    }
}
