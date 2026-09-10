// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native k0s cluster controller (leader-gated).
//!
//! Distinct from `crate::cluster` (the raft-backed `ClusterManager` state machine): this drives
//! the SPUR-managed k0s cluster — role selection, IP/CIDR allocation, and the
//! per-node `SlurmAgent` StartClusterComponent/StopClusterComponent fan-out — all
//! gated on Raft leadership. Phase transitions go through `ClusterManager::set_k0s_phase`
//! (WAL-replicated) so a leadership change mid-provision is safe.

use std::collections::{HashMap, HashSet};
use std::net::Ipv4Addr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use tracing::{info, warn};

use spur_core::k0s::{K0sClusterState, K0sPhase, K0sRole};
use spur_net::address::AddressPool;
use spur_net::mesh::{MeshMembership, MeshNode};
use spur_proto::proto::slurm_agent_client::SlurmAgentClient;
use spur_proto::proto::{
    ClusterNodeStatus, CreateK0sJoinTokenRequest, DeleteK8sNodeRequest, DrainK8sNodeRequest,
    GetClusterComponentStatusRequest, GetKubeconfigRequest, StartClusterComponentRequest,
    StopClusterComponentRequest,
};

use crate::cluster::ClusterManager;
use crate::raft::RaftHandle;

/// Per-agent RPC dial timeout.
const AGENT_TIMEOUT: Duration = Duration::from_secs(5);

const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);

/// Network CIDRs the reconcile loop needs (from `[network]` + `[cluster]` config).
#[derive(Clone, Debug)]
pub struct ClusterNetworking {
    /// Whether the WireGuard mesh is enabled (network.wg_enabled). Independent of `[cluster].enabled`:
    /// a k0s cluster can run without the mesh, in which case the reconcile skips reading peer
    /// endpoints from `wg` (nothing to advertise, and `wg show` would just error every tick).
    pub wg_enabled: bool,
    /// WireGuard mesh CIDR (network.wg_cidr) — node mesh IPs are allocated from here.
    pub mesh_cidr: String,
    /// WireGuard interface name (network.wg_interface) — the controller reads its peer endpoints
    /// from this to re-advertise worker↔worker underlay endpoints in the mesh membership.
    pub mesh_interface: String,
    /// Pod CIDR (cluster.pod_cidr) — per-node /24s are carved from here.
    pub pod_cidr: String,
    /// Service CIDR (cluster.service_cidr) — for the generated k0s config.
    pub service_cidr: String,
    /// CNI MTU (cluster.cni_mtu) — emitted into the generated Calico config.
    pub cni_mtu: u16,
    /// CNI mode (cluster.cni): "kuberouter" (default) or "calico" (mesh-native config + node-ip).
    pub cni: String,
    /// Operator-pinned control-plane node (cluster.control_plane_node), if any.
    pub control_plane_node: Option<String>,
    /// How long a k8s (k0s) node may stay non-`active` during provisioning before the loop
    /// marks the cluster `degraded` (cluster.k8s_provisioning_timeout_secs).
    pub provisioning_timeout: Duration,
}

impl ClusterNetworking {
    /// Whether pods route natively over the WireGuard mesh. Only Calico (`bird` mode) can; kube-router
    /// always tunnels over the underlay, so a mesh IP must never be advertised to it.
    fn mesh_native(&self) -> bool {
        self.wg_enabled && self.cni == "calico"
    }
}

/// Leader-gated k0s reconcile loop. Spawned from `main.rs` when `[cluster].enabled`; it still
/// re-checks leadership every tick because leadership can flip at any time.
pub async fn run(cluster: Arc<ClusterManager>, raft: Arc<RaftHandle>, net: ClusterNetworking) {
    info!(mesh = %net.mesh_cidr, pod = %net.pod_cidr, "k0s cluster reconcile loop started");
    let mut interval = tokio::time::interval(RECONCILE_INTERVAL);
    let mut last_mesh: Vec<MeshNode> = Vec::new();
    // Cache of the join token minted per node (worker or secondary control-plane), so we mint once
    // (not every tick) while it joins — re-minting churns k0s server tokens and races the join.
    let mut join_tokens: HashMap<String, String> = HashMap::new();
    // Leader-local provisioning clock: a leader-flip or restart resets it, so the timeout re-arms
    // on the new leader rather than tripping instantly off a persisted start time.
    let mut provisioning_since: Option<Instant> = None;
    loop {
        interval.tick().await;
        if !raft.is_leader() {
            last_mesh.clear(); // forget on leadership loss so a new term re-logs the membership
            provisioning_since = None;
            continue; // only the leader reconciles
        }
        let state = cluster.k0s_state();

        let timed_out = update_provisioning_clock(
            state.phase,
            &mut provisioning_since,
            Instant::now(),
            net.provisioning_timeout,
        );

        // Mesh: derive the authoritative full-mesh membership (pubkey + mesh IP + pod /24) from
        // live inventory and push it to every meshed node's agent (ApplyMesh) so a native-routing
        // CNI can ride the WireGuard mesh. Level-triggered: re-push EVERY tick (the agent's
        // reconcile_mesh is idempotent + prunes) so node-local drift (reboot, wg restart), a failed
        // push, and controller failover all self-heal. Only meaningful with ≥2 meshed nodes; the
        // membership diff gates only the log line, not the push.
        // build_mesh_membership shells out to `wg` (peer_endpoints); keep that off the async
        // runtime with spawn_blocking, matching the agent side's apply_mesh handler.
        let mesh = {
            let cluster = cluster.clone();
            let (wg_enabled, mesh_cidr, mesh_interface) = (
                net.wg_enabled,
                net.mesh_cidr.clone(),
                net.mesh_interface.clone(),
            );
            match tokio::task::spawn_blocking(move || {
                build_mesh_membership(&cluster, wg_enabled, &mesh_cidr, &mesh_interface)
            })
            .await
            {
                Ok(m) => m,
                Err(e) => {
                    warn!(error = %e, "build_mesh_membership task panicked; skipping mesh push this tick");
                    MeshMembership::default()
                }
            }
        };
        if mesh.nodes.len() >= 2 {
            if mesh.nodes != last_mesh {
                info!(
                    members = mesh.nodes.len(),
                    "k0s full-mesh membership changed"
                );
            }
            for node in &mesh.nodes {
                spawn_apply_mesh(&cluster, &node.hostname, &mesh);
            }
        }
        last_mesh = mesh.nodes.clone();

        let started = Instant::now();
        let errored = reconcile_phase(&cluster, &net, &state, &mut join_tokens, timed_out).await;
        let cluster_name = cluster.config().cluster_name.clone();
        cluster
            .k8s_metrics()
            .observe_reconcile_duration(&cluster_name, started.elapsed().as_secs_f64());
        if errored {
            cluster.k8s_metrics().record_reconcile_error(&cluster_name);
        }
    }
}

/// Advance the leader-local provisioning clock and report whether the deadline passed. Arms on the
/// first Provisioning observation and resets on any other phase, so re-entry restarts the timer.
fn update_provisioning_clock(
    phase: K0sPhase,
    since: &mut Option<Instant>,
    now: Instant,
    timeout: Duration,
) -> bool {
    if phase != K0sPhase::Provisioning {
        *since = None;
        return false;
    }
    let started = *since.get_or_insert(now);
    now.saturating_duration_since(started) >= timeout
}

/// Run one reconcile tick for the current phase. Extracted from `run` so it is testable.
///
/// Ready and Provisioning both run the assignment + converge reconcile so the cluster self-heals: a
/// node that is removed then re-added (a spurd restart deregisters on SIGTERM, dropping the node +
/// its k0s assignment) or a node added while Ready gets (re)assigned a role/IP/CIDR, (re)joined, and
/// rejoins the mesh membership on the next ApplyMesh tick. Idempotent — assigned + active nodes are
/// skipped — so a converged cluster does no work beyond the per-node status probes. Without running
/// this in Ready, a re-added node stays un-roled (out of the mesh) until the next manual `spur k8s up`.
pub(crate) async fn reconcile_phase(
    cluster: &ClusterManager,
    net: &ClusterNetworking,
    state: &K0sClusterState,
    join_tokens: &mut HashMap<String, String>,
    timed_out: bool,
) -> bool {
    match state.phase {
        K0sPhase::Ready | K0sPhase::Provisioning => {
            if let Err(e) = provision_assignments(cluster, net, state) {
                warn!(error = %e, "k0s provisioning assignment failed; will retry next tick");
                return true;
            }
            let (errors, active_cp) = converge_provisioning(cluster, net, join_tokens).await;
            // converge may have flipped us to Ready this tick; only degrade a still-provisioning
            // cluster that has blown its deadline. Reuse converge's control-plane liveness (no second
            // per-node status sweep).
            if timed_out && cluster.k0s_state().phase == K0sPhase::Provisioning {
                degrade_stuck_cluster(cluster, net, join_tokens, &active_cp).await;
            }
            errors > 0
        }
        K0sPhase::Down => {
            // Drop cached join tokens: they were minted against this incarnation's CA, which a
            // rebuild regenerates. Keeping them would hand a worker a stale-CA token on the next
            // `up` (roles are re-assigned before converge runs, so its empty-set clear never fires).
            join_tokens.clear();
            stop_all_components(cluster, state.reset_requested).await;
            false
        }
        K0sPhase::Degraded => {
            warn!("k0s cluster degraded");
            false
        }
    }
}

/// Assign role + mesh IP + pod /24 to any node that lacks one. Idempotent: a node's persisted
/// `k0s_role`/`k0s_mesh_ip`/`k0s_pod_cidr` IS the allocation record, so assigned nodes are skipped
/// and never re-allocated. The (in-memory) AddressPool is re-seeded from persisted inventory on
/// every call — skipping that would hand out IPs already in use after a controller restart.
pub(crate) fn provision_assignments(
    cluster: &ClusterManager,
    net: &ClusterNetworking,
    state: &K0sClusterState,
) -> anyhow::Result<()> {
    let mut nodes = cluster.get_nodes();
    nodes.retain(|n| state.is_member(&n.name));
    if nodes.is_empty() {
        return Ok(());
    }
    nodes.sort_by(|a, b| a.name.cmp(&b.name)); // deterministic

    // Bootstrap control-plane (etcd seed, holder of `.1`). Recorded bootstrap outranks a scanned
    // role (secondary CPs also carry `Controller`) so `.1` stays put across a 1->3 grow.
    let bootstrap = state
        .bootstrap()
        .or_else(|| {
            nodes
                .iter()
                .find(|n| matches!(n.k0s_role, Some(K0sRole::Single | K0sRole::Controller)))
                .map(|n| n.name.clone())
        })
        .or_else(|| net.control_plane_node.clone())
        .unwrap_or_else(|| nodes[0].name.clone());

    // Control-plane set: the persisted HA set (from `cluster_up`), or just the bootstrap for the
    // legacy single-CP path where no set was recorded.
    let mut cp_set: HashSet<String> = state.controllers().into_iter().collect();
    if cp_set.is_empty() {
        cp_set.insert(bootstrap.clone());
    }

    // Re-seed the mesh pool from persisted assignments + reserve .1 for the bootstrap controller. An
    // already-assigned node's persisted `k0s_mesh_ip` is authoritative, so remember which node owns each
    // in-mesh address to arbitrate newcomer conflicts below.
    let mut pool = AddressPool::new(&net.mesh_cidr)?;
    let controller_ip = first_host(&net.mesh_cidr)?;
    let _ = pool.allocate_specific(controller_ip); // reserve .1 (ignore if already reserved)
    let mut assigned_addr: HashMap<Ipv4Addr, String> = HashMap::new();
    for n in &nodes {
        if let Some(ip) = &n.k0s_mesh_ip {
            let parsed: Ipv4Addr = ip
                .parse()
                .with_context(|| format!("persisted k0s_mesh_ip {ip} for {}", n.name))?;
            pool.mark_allocated(parsed);
            if n.k0s_role.is_some() {
                assigned_addr.insert(parsed, n.name.clone());
            }
        }
    }

    // Adopt each node's REAL mesh address: a meshed node advertises its `spur0` IP as `Node.address`,
    // and when it falls within `mesh_cidr` that IS its mesh IP — the single source of truth. Re-deriving
    // from an independent pool is what silently corrupts the mesh when the control-plane set is not an
    // alphabetical prefix of join order. An out-of-mesh (underlay) address falls through to pool
    // allocation below.
    //
    // A conflict — two nodes advertising the same in-mesh address — is contained, not fatal: only the
    // conflicting nodes stay unprovisioned (each tagged with the reason for `spur k8s status`), while
    // healthy nodes provision normally. An already-assigned member owns its address, so a newcomer
    // colliding with it is the one refused; two unassigned newcomers on the same address both stay out.
    //
    // Group unassigned claimants by advertised in-mesh address so an N-way conflict resolves in one shot.
    let mut claimants: HashMap<Ipv4Addr, Vec<String>> = HashMap::new();
    for n in &nodes {
        if n.k0s_role.is_some() {
            continue; // already assigned; its persisted mesh IP is authoritative
        }
        if let Some(addr) = real_mesh_address(n, &net.mesh_cidr)? {
            claimants.entry(addr).or_default().push(n.name.clone());
        }
    }

    let mut real_ip: HashMap<String, Ipv4Addr> = HashMap::new();
    let mut refused: HashSet<String> = HashSet::new();
    for (addr, mut names) in claimants {
        if let Some(owner) = assigned_addr.get(&addr) {
            // An in-cluster node already owns this address; every newcomer claiming it is refused.
            for name in &names {
                let reason =
                    format!("network mismatch: mesh address {addr} already owned by {owner}");
                record_node_k0s_error(cluster, name, &reason);
                refused.insert(name.clone());
            }
            continue;
        }
        if names.len() > 1 {
            // Two or more unassigned nodes advertise the same address: none can adopt it. Keep all out
            // and name every claimant on each so status points at the whole conflict.
            names.sort();
            let reason = format!(
                "network mismatch: mesh address {addr} claimed by {}",
                names.join(", ")
            );
            for name in &names {
                record_node_k0s_error(cluster, name, &reason);
                refused.insert(name.clone());
            }
            continue;
        }
        pool.mark_allocated(addr);
        real_ip.insert(names.remove(0), addr);
    }

    // Pod-/24 ordinals already in use (so a new node never collides).
    let pod_base = cidr_base(&net.pod_cidr)?;
    let mut used_ordinals: HashSet<u32> = nodes
        .iter()
        .filter_map(|n| n.k0s_pod_cidr.as_deref())
        .filter_map(|c| pod_ordinal(c, pod_base))
        .collect();

    let single = nodes.len() == 1;
    // Two passes so control planes take the lowest mesh IPs deterministically (bootstrap keeps `.1`,
    // secondary CPs `.2`/`.3`...) regardless of where they sort among workers.
    let ordered: Vec<_> = nodes
        .iter()
        .filter(|n| cp_set.contains(&n.name))
        .chain(nodes.iter().filter(|n| !cp_set.contains(&n.name)))
        .collect();
    for node in ordered {
        if node.k0s_role.is_some() {
            continue; // already assigned
        }
        if refused.contains(&node.name) {
            continue; // conflicting mesh address; stays unprovisioned until the operator resolves it
        }
        let is_cp = cp_set.contains(&node.name);
        let role = if is_cp {
            if single {
                K0sRole::Single
            } else {
                K0sRole::Controller
            }
        } else {
            K0sRole::Worker
        };
        // Adopt the node's real WireGuard address; fall back to `.1` for a not-yet-meshed bootstrap,
        // else the pool.
        let mesh_ip = if let Some(addr) = real_ip.get(&node.name) {
            *addr
        } else if node.name == bootstrap {
            controller_ip
        } else {
            pool.allocate()?
        };
        let ordinal = next_free_ordinal(&used_ordinals);
        used_ordinals.insert(ordinal);
        let pod_cidr = carve_pod_cidr(&net.pod_cidr, ordinal)?;
        cluster.assign_node_k0s(&node.name, role, &mesh_ip.to_string(), &pod_cidr)?;
    }

    // Persist the bootstrap choice if not already recorded (legacy single-CP path; `cluster_up`
    // records the full set up front for HA).
    if state.control_plane_node.as_deref() != Some(bootstrap.as_str()) {
        cluster.set_k0s_phase(
            K0sPhase::Provisioning,
            Some(bootstrap),
            Vec::new(),
            Vec::new(),
            false,
        )?;
    }
    Ok(())
}

/// Resolve the member scope for `spur k8s up` fail-closed: the UNION of a `nodes` hostlist, a
/// `partition`'s members, and a label `selector` (all pairs match). Empty (nothing given) = whole inventory.
pub(crate) fn resolve_member_nodes(
    all_nodes: &[spur_core::node::Node],
    nodes_hostlist: &str,
    partition: &str,
    selector: &HashMap<String, String>,
) -> Result<Vec<String>, String> {
    if nodes_hostlist.is_empty() && partition.is_empty() && selector.is_empty() {
        return Ok(Vec::new());
    }
    let registered: HashSet<&str> = all_nodes.iter().map(|n| n.name.as_str()).collect();
    let mut members: HashSet<String> = HashSet::new();

    if !nodes_hostlist.is_empty() {
        let expanded = spur_core::hostlist::expand(nodes_hostlist)
            .map_err(|e| format!("invalid --nodes hostlist {nodes_hostlist}: {e}"))?;
        for name in expanded {
            if !registered.contains(name.as_str()) {
                return Err(format!("node {name} is not a registered node"));
            }
            members.insert(name);
        }
    }
    if !partition.is_empty() {
        let mut any = false;
        for n in all_nodes {
            if n.partitions.iter().any(|p| p == partition) {
                members.insert(n.name.clone());
                any = true;
            }
        }
        if !any {
            return Err(format!("partition {partition} has no registered nodes"));
        }
    }
    if !selector.is_empty() {
        let mut any = false;
        for n in all_nodes {
            if selector.iter().all(|(k, v)| n.labels.get(k) == Some(v)) {
                members.insert(n.name.clone());
                any = true;
            }
        }
        if !any {
            return Err("--selector matched no registered nodes".to_string());
        }
    }
    if members.is_empty() {
        return Err("node selection matched no registered nodes".to_string());
    }
    let mut out: Vec<String> = members.into_iter().collect();
    out.sort();
    Ok(out)
}

/// Resolve the control-plane set for `spur k8s up`, fail-closed, bootstrap node first: an explicit
/// `nodes` list wins, else the lowest `replicas` candidates. Count must be 1/3/5 and fit the nodes.
pub(crate) fn resolve_control_plane_set(
    mut candidates: Vec<String>,
    explicit: &[String],
    pinned_bootstrap: Option<&str>,
    replicas: u32,
) -> Result<Vec<String>, String> {
    candidates.sort();
    candidates.dedup();
    if !explicit.is_empty() {
        spur_core::k0s::validate_control_plane_replicas(explicit.len() as u32)?;
        let mut seen = HashSet::new();
        for n in explicit {
            if !candidates.contains(n) {
                return Err(format!("control-plane node {n} is not a registered node"));
            }
            if !seen.insert(n.clone()) {
                return Err(format!("duplicate control-plane node {n}"));
            }
        }
        // Fail closed on a contradictory bootstrap: if a bootstrap is pinned (operator override or a
        // previously-recorded CP) but absent from the explicit list, `.1`/etcd-seed would silently
        // land on a different node than intended.
        if let Some(boot) = pinned_bootstrap {
            if !explicit.iter().any(|n| n == boot) {
                return Err(format!(
                    "bootstrap control-plane {boot} is not in the requested set [{}]",
                    explicit.join(", ")
                ));
            }
        }
        let mut set = explicit.to_vec();
        order_bootstrap_first(&mut set, pinned_bootstrap);
        return Ok(set);
    }
    spur_core::k0s::validate_control_plane_replicas(replicas)?;
    if replicas as usize > candidates.len() {
        return Err(format!(
            "requested {replicas} control planes but only {} node(s) are registered",
            candidates.len()
        ));
    }
    // Fail closed on a pinned bootstrap outside the candidate set (e.g. a --control-plane-node not in
    // the requested node scope) — else `.1`/etcd-seed silently lands on a different, in-scope node.
    if let Some(boot) = pinned_bootstrap {
        if !candidates.iter().any(|c| c == boot) {
            return Err(format!(
                "control-plane node {boot} is not among the selected cluster nodes"
            ));
        }
    }
    // Pin the bootstrap into the set first so `.1` lands on it, then fill from the lowest names.
    let mut set: Vec<String> = Vec::new();
    if let Some(boot) = pinned_bootstrap {
        set.push(boot.to_string());
    }
    for c in candidates {
        if set.len() >= replicas as usize {
            break;
        }
        if !set.contains(&c) {
            set.push(c);
        }
    }
    order_bootstrap_first(&mut set, pinned_bootstrap);
    Ok(set)
}

/// Move the pinned bootstrap node to the front of the CP set (it holds `.1` + seeds etcd).
fn order_bootstrap_first(set: &mut [String], pinned_bootstrap: Option<&str>) {
    if let Some(boot) = pinned_bootstrap {
        if let Some(pos) = set.iter().position(|n| n == boot) {
            set.swap(0, pos);
        }
    }
}

/// The `.1` host of a CIDR (mesh controller IP).
fn first_host(cidr: &str) -> anyhow::Result<Ipv4Addr> {
    Ok(Ipv4Addr::from(u32::from(cidr_base(cidr)?) + 1))
}

/// A node's real WireGuard mesh address: the `spur0` IP it advertises (`Node.address`) once meshed
/// (non-empty `wg_pubkey`), but only when it falls within `mesh_cidr`. Ok(None) for an unmeshed node
/// or an out-of-mesh (underlay) address — both fall back to pool allocation. Errs on malformed CIDR.
fn real_mesh_address(
    node: &spur_core::node::Node,
    mesh_cidr: &str,
) -> anyhow::Result<Option<Ipv4Addr>> {
    if node
        .wg_pubkey
        .as_deref()
        .filter(|k| !k.is_empty())
        .is_none()
    {
        return Ok(None);
    }
    let Some(addr) = node
        .address
        .as_deref()
        .and_then(|a| a.parse::<Ipv4Addr>().ok())
    else {
        return Ok(None);
    };
    if cidr_contains(mesh_cidr, addr)? {
        Ok(Some(addr))
    } else {
        Ok(None)
    }
}

/// Whether `ip` falls within `cidr` (same network prefix).
fn cidr_contains(cidr: &str, ip: Ipv4Addr) -> anyhow::Result<bool> {
    let (base, prefix) = cidr
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("{cidr} is not a CIDR"))?;
    let base: Ipv4Addr = base
        .parse()
        .with_context(|| format!("CIDR base in {cidr}"))?;
    let prefix: u8 = prefix
        .parse()
        .with_context(|| format!("CIDR prefix in {cidr}"))?;
    if prefix > 32 {
        anyhow::bail!("{cidr} prefix must be <= 32");
    }
    let mask = if prefix == 0 {
        0
    } else {
        !((1u32 << (32 - prefix)) - 1)
    };
    Ok(u32::from(base) & mask == u32::from(ip) & mask)
}

/// The base address of a CIDR string.
fn cidr_base(cidr: &str) -> anyhow::Result<Ipv4Addr> {
    let (base, _) = cidr
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("{cidr} is not a CIDR"))?;
    base.parse().with_context(|| format!("CIDR base in {cidr}"))
}

/// Carve a per-node pod /24 out of `pod_cidr` by ordinal, e.g. ("10.42.0.0/16", 2) -> "10.42.2.0/24".
fn carve_pod_cidr(pod_cidr: &str, ordinal: u32) -> anyhow::Result<String> {
    let (base, prefix) = pod_cidr
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("{pod_cidr} is not a CIDR"))?;
    let prefix: u8 = prefix
        .parse()
        .with_context(|| format!("pod_cidr prefix in {pod_cidr}"))?;
    if prefix > 24 {
        anyhow::bail!("pod_cidr {pod_cidr} must be /24 or larger to carve per-node /24s");
    }
    let base: Ipv4Addr = base
        .parse()
        .with_context(|| format!("pod_cidr base in {pod_cidr}"))?;
    let num_24s = 1u32 << (24 - prefix);
    if ordinal >= num_24s {
        anyhow::bail!("pod ordinal {ordinal} exceeds {num_24s} /24s in {pod_cidr}");
    }
    let carved = u32::from(base) + (ordinal << 8);
    Ok(format!("{}/24", Ipv4Addr::from(carved)))
}

/// Inverse of `carve_pod_cidr`: the ordinal of a per-node /24 within `pod_base`.
fn pod_ordinal(node_cidr: &str, pod_base: Ipv4Addr) -> Option<u32> {
    let (b, _) = node_cidr.split_once('/')?;
    let nb: Ipv4Addr = b.parse().ok()?;
    Some(u32::from(nb).checked_sub(u32::from(pod_base))? >> 8)
}

/// Smallest non-negative ordinal not already in use.
fn next_free_ordinal(used: &HashSet<u32>) -> u32 {
    let mut o = 0;
    while used.contains(&o) {
        o += 1;
    }
    o
}

/// Proto status string for a cluster phase.
pub fn phase_str(p: K0sPhase) -> String {
    match p {
        K0sPhase::Down => "down",
        K0sPhase::Provisioning => "provisioning",
        K0sPhase::Ready => "ready",
        K0sPhase::Degraded => "degraded",
    }
    .to_string()
}

fn role_str(r: K0sRole) -> String {
    match r {
        K0sRole::Controller => "controller",
        K0sRole::Worker => "worker",
        K0sRole::Single => "single",
    }
    .to_string()
}

/// Per-node status list from persisted (Raft-replicated) k0s state only (no agent round-trip).
pub fn node_statuses(cluster: &ClusterManager) -> Vec<ClusterNodeStatus> {
    cluster
        .get_nodes()
        .into_iter()
        .filter_map(|n| {
            let role = n.k0s_role?;
            Some(ClusterNodeStatus {
                node: n.name,
                role: role_str(role),
                component_state: "unknown".to_string(),
                enabled: true,
                reason: n.k0s_last_error.unwrap_or_default(),
            })
        })
        .collect()
}

/// Build the authoritative full-mesh membership from live node inventory: every node that has both
/// joined the WireGuard mesh (non-empty `wg_pubkey`, reported at registration) and been assigned a
/// mesh IP. Each entry carries the node's pod /24, so a native-routing CNI (Calico `bird`) can ride
/// the mesh — the controller is the source of truth for `MeshNode.public_key`/`pod_cidr`, which an
/// operator feeds to `apply_mesh` on each node via `spur net mesh --config`. Nodes not yet on the
/// mesh (no pubkey) are skipped rather than fabricated, so an incomplete membership is never emitted.
pub fn build_mesh_membership(
    cluster: &ClusterManager,
    wg_enabled: bool,
    mesh_cidr: &str,
    mesh_interface: &str,
) -> MeshMembership {
    // Peer endpoints only matter under WireGuard; when the mesh is off, skip the `wg` shell-out
    // entirely (it would just error every reconcile tick). Otherwise snapshot the controller's
    // peer→endpoint table so worker↔worker peers get an endpoint (net join only wires
    // worker→controller). Log on failure — an empty table silently regresses to hub-and-spoke.
    let endpoints = if wg_enabled {
        spur_net::wireguard::peer_endpoints(mesh_interface).unwrap_or_else(|e| {
            warn!(
                interface = %mesh_interface, error = %e,
                "could not read WireGuard peer endpoints; mesh membership will omit them, so \
                 worker↔worker tunnels may fall back to hub-and-spoke"
            );
            std::collections::HashMap::new()
        })
    } else {
        std::collections::HashMap::new()
    };
    mesh_from_nodes(cluster.get_nodes(), mesh_cidr, &endpoints)
}

/// Pure core of [`build_mesh_membership`] (testable without a `ClusterManager`).
///
/// Includes any meshed node (non-empty `wg_pubkey`) with a known mesh IP — its `k0s_mesh_ip` if it
/// has a role, else its advertised address when that falls inside `mesh_cidr` — so controller/login
/// nodes stay in membership. `endpoints` supplies each peer's underlay endpoint for worker↔worker.
fn mesh_from_nodes(
    nodes: Vec<spur_core::node::Node>,
    mesh_cidr: &str,
    endpoints: &std::collections::HashMap<String, String>,
) -> MeshMembership {
    let mut nodes: Vec<MeshNode> = nodes
        .into_iter()
        .filter_map(|n| {
            let public_key = n.wg_pubkey.clone().filter(|k| !k.is_empty())?;
            let mesh_ip = match n.k0s_mesh_ip.clone() {
                Some(ip) => ip,
                None => real_mesh_address(&n, mesh_cidr).ok().flatten()?.to_string(),
            };
            let endpoint = endpoints.get(&public_key).cloned().unwrap_or_default();
            Some(MeshNode {
                hostname: n.name,
                public_key,
                mesh_ip,
                endpoint,
                pod_cidr: n.k0s_pod_cidr.clone(),
            })
        })
        .collect();
    // Sort numerically by IPv4 — a string sort orders "10.44.0.10" before "10.44.0.2", producing
    // spurious membership diffs (and unnecessary ApplyMesh pushes) between ticks.
    nodes.sort_by(|a, b| {
        a.mesh_ip
            .parse::<std::net::Ipv4Addr>()
            .ok()
            .cmp(&b.mesh_ip.parse::<std::net::Ipv4Addr>().ok())
            .then_with(|| a.mesh_ip.cmp(&b.mesh_ip))
    });
    MeshMembership { nodes }
}

/// Resolve a node's agent endpoint (`http://addr:port`), or None if it has no address.
fn agent_endpoint(cluster: &ClusterManager, node: &str) -> Option<String> {
    let n = cluster.get_node(node)?;
    let addr = n.address?;
    Some(format!("http://{}:{}", addr, n.port))
}

/// Dial a healthy control-plane agent, timing out per `AGENT_TIMEOUT`. Tries the bootstrap node
/// first (its k0s API answers admin/token RPCs) then any other control plane, so admin/kubeconfig/
/// token minting survive the loss of a single control plane while etcd quorum holds. Errors if no
/// control-plane node is assigned yet or none is reachable.
async fn connect_control_plane(
    cluster: &ClusterManager,
) -> anyhow::Result<SlurmAgentClient<crate::agent_client::AgentChannel>> {
    let state = cluster.k0s_state();
    let mut candidates = state.controllers();
    if candidates.is_empty() {
        anyhow::bail!("no control-plane node assigned yet");
    }
    // Bootstrap node first (stable admin endpoint), then the rest as failover.
    if let Some(boot) = &state.control_plane_node {
        if let Some(pos) = candidates.iter().position(|n| n == boot) {
            candidates.swap(0, pos);
        }
    }
    let mut last_err = String::new();
    for cp in &candidates {
        let Some(endpoint) = agent_endpoint(cluster, cp) else {
            last_err = format!("control-plane node {cp} has no agent address");
            continue;
        };
        match tokio::time::timeout(AGENT_TIMEOUT, crate::agent_client::connect(endpoint)).await {
            Ok(Ok(client)) => return Ok(client),
            Ok(Err(e)) => last_err = format!("connect to control-plane agent {cp} failed: {e}"),
            Err(_) => last_err = format!("connect to control-plane agent {cp} timed out"),
        }
    }
    anyhow::bail!("no reachable control-plane agent ({last_err})")
}

/// Mint a join token of `role` ("worker" | "controller") from a control-plane agent (`k0s token
/// create --role <role>`). Errors until a control-plane component is up (its k0s API must answer);
/// the caller retries. A controller token lets a secondary CP join the bootstrap's etcd quorum.
async fn mint_join_token(cluster: &ClusterManager, role: &str) -> anyhow::Result<String> {
    let mut client = connect_control_plane(cluster).await?;
    let resp = client
        .create_k0s_join_token(CreateK0sJoinTokenRequest {
            role: role.to_string(),
            expiry_seconds: 0, // k0s default lifetime
        })
        .await
        .map_err(|e| anyhow::anyhow!("create_k0s_join_token RPC failed: {e}"))?;
    Ok(resp.into_inner().join_token)
}

/// Fetch the admin kubeconfig from the control-plane node's agent (`k0s kubeconfig admin`), for the
/// ClusterKubeconfig RPC. Errors if there is no control-plane node yet or it is unreachable.
pub async fn fetch_admin_kubeconfig(cluster: &ClusterManager) -> anyhow::Result<String> {
    let mut client = connect_control_plane(cluster).await?;
    let resp = client
        .get_kubeconfig(GetKubeconfigRequest::default())
        .await
        .map_err(|e| anyhow::anyhow!("get_kubeconfig RPC failed: {e}"))?;
    Ok(resp.into_inner().kubeconfig)
}

/// Mint a namespace-scoped kubeconfig for a SPUR user: the control-plane agent ensures the
/// ServiceAccount exists in the account namespace and mints a bound token. `namespace` + `sa` are
/// derived by the caller from the user's account via `spur_core::quota_names`.
pub async fn fetch_user_kubeconfig(
    cluster: &ClusterManager,
    user: &str,
    namespace: &str,
    service_account: &str,
) -> anyhow::Result<String> {
    let mut client = connect_control_plane(cluster).await?;
    let resp = client
        .get_kubeconfig(GetKubeconfigRequest {
            user: user.to_string(),
            namespace: namespace.to_string(),
            service_account: service_account.to_string(),
        })
        .await
        .map_err(|e| anyhow::anyhow!("get_kubeconfig (scoped) RPC failed: {e}"))?;
    Ok(resp.into_inner().kubeconfig)
}

/// Query a node's live k0s component state via its agent, with a timeout. Returns None if the node
/// is unreachable or has no component yet.
async fn fetch_component_status(cluster: &ClusterManager, node: &str) -> Option<(String, bool)> {
    let endpoint = agent_endpoint(cluster, node)?;
    let fut = async {
        let mut client = crate::agent_client::connect(endpoint).await.ok()?;
        let resp = client
            .get_cluster_component_status(GetClusterComponentStatusRequest {})
            .await
            .ok()?;
        let r = resp.into_inner();
        Some((r.component_state, r.enabled))
    };
    tokio::time::timeout(AGENT_TIMEOUT, fut)
        .await
        .ok()
        .flatten()
}

async fn fetch_component_state(cluster: &ClusterManager, node: &str) -> Option<String> {
    fetch_component_status(cluster, node)
        .await
        .map(|(state, _)| state)
}

/// The address Calico/kubelet should use for `node`: its mesh IP when the mesh is configured
/// (`mesh_native`), else its real underlay address — a `k0s_mesh_ip` handed out with the mesh
/// disabled is a pool-allocated address never bound to any interface, so it must not be used.
/// `kubelet --node-ip` requires a literal IP (unlike `Node.address`, which may be an FQDN), so the
/// underlay branch only returns a value that parses as one.
fn calico_node_address(mesh_native: bool, node: &spur_core::node::Node) -> Option<&str> {
    if mesh_native {
        node.k0s_mesh_ip.as_deref()
    } else {
        node.address
            .as_deref()
            .filter(|addr| addr.parse::<std::net::IpAddr>().is_ok())
    }
}

/// The k0s controller config for `node`, or None for an unknown CNI / a node without a usable
/// address yet. `cp_count > 1` also enables node-local load balancing. Only Calico rides the mesh
/// ([`ClusterNetworking::mesh_native`]); kube-router always advertises the API on the underlay address.
fn controller_k0s_config(
    net: &ClusterNetworking,
    node: &spur_core::node::Node,
    cp_count: usize,
) -> Option<String> {
    let mesh_native = net.mesh_native();
    let api = calico_node_address(mesh_native, node)?;
    // SANs: the advertised address + the underlay address (so `kubectl` over either works).
    let mut sans = vec![api.to_string()];
    if let Some(addr) = &node.address {
        if addr != api {
            sans.push(addr.clone());
        }
    }
    spur_core::k0s::k0s_controller_config_yaml(
        &net.cni,
        &net.pod_cidr,
        &net.service_cidr,
        net.cni_mtu,
        api,
        &sans,
        cp_count,
        mesh_native,
    )
}

/// One convergence pass: start any assigned component not yet active and, once the control-plane
/// quorum is up, mark the cluster Ready (partial-Ready — workers keep converging in the background).
/// `join_tokens` caches each worker's minted join token across ticks so we mint once per join.
/// Returns the error count for this pass (currently join-token mint failures, surfaced via
/// `spur_k8s_reconcile_errors_total`) plus the set of control-plane nodes observed `active` this
/// pass, so the caller can decide the degrade/Ready edge without a second per-node status sweep.
async fn converge_provisioning(
    cluster: &ClusterManager,
    net: &ClusterNetworking,
    join_tokens: &mut HashMap<String, String>,
) -> (u64, HashSet<String>) {
    let mut errors = 0u64;
    let assigned: Vec<_> = cluster
        .get_nodes()
        .into_iter()
        .filter(|n| n.k0s_role.is_some())
        .collect();
    if assigned.is_empty() {
        join_tokens.clear();
        return (errors, HashSet::new());
    }
    let bootstrap = cluster.k0s_state().bootstrap();
    let cp_count = cluster.k0s_state().controllers().len();
    let mut bootstrap_active = false;
    // Active control-plane nodes this tick: Ready is gated on their quorum (partial-Ready), not on
    // every worker being up.
    let mut active_cp: HashSet<String> = HashSet::new();
    // Bootstrap control-plane first: it seeds etcd (tokenless) and its k0s API must answer before any
    // secondary control-plane or worker can mint a join token. A Single node is always the bootstrap.
    for node in &assigned {
        let role = node.k0s_role.expect("assigned above");
        if role == K0sRole::Worker {
            continue;
        }
        let is_bootstrap = role == K0sRole::Single || bootstrap.as_deref() == Some(&node.name);
        if !is_bootstrap {
            continue;
        }
        if fetch_component_state(cluster, &node.name).await.as_deref() == Some("active") {
            bootstrap_active = true;
            active_cp.insert(node.name.clone());
            clear_node_error(cluster, node);
            continue;
        }
        // The bootstrap seeds etcd — no join token. Its k0s config pins the CIDRs and CNI mode for
        // the whole cluster.
        let k0s_config = controller_k0s_config(net, node, cp_count);
        spawn_start_component(cluster, &node.name, role, None, k0s_config, None);
    }
    // Don't mint join tokens for secondary CPs / workers until the bootstrap's etcd is seeded and its
    // API answers: a controller token minted before then would race the quorum. But if a quorum of
    // control planes is already active (e.g. the bootstrap died after secondaries joined), the cluster
    // is still usable — flip it Ready before returning rather than pinning it in Provisioning forever.
    if !bootstrap_active {
        maybe_mark_ready(cluster, &active_cp);
        return (errors, active_cp);
    }
    // Secondary CPs join the etcd quorum with a `controller` token, then workers with a `worker`
    // token; both mint from a healthy CP agent, and a minting error just retries next tick.
    for node in &assigned {
        let role = node.k0s_role.expect("assigned above");
        let is_bootstrap = role == K0sRole::Single || bootstrap.as_deref() == Some(&node.name);
        if is_bootstrap {
            continue; // handled above
        }
        if fetch_component_state(cluster, &node.name).await.as_deref() == Some("active") {
            join_tokens.remove(&node.name); // joined — drop the cached token
            if role == K0sRole::Controller {
                active_cp.insert(node.name.clone());
            }
            clear_node_error(cluster, node);
            continue;
        }
        let token_role = if role == K0sRole::Controller {
            "controller"
        } else {
            "worker"
        };
        // For Calico, pin the node's kubelet node-ip to whichever address it's actually running on;
        // kube-router picks the underlay address itself.
        let node_ip = if net.cni == "calico" {
            calico_node_address(net.mesh_native(), node).map(String::from)
        } else {
            None
        };
        // A secondary control-plane also needs its own generated k0s config (API SANs per calico_node_address).
        let k0s_config = if role == K0sRole::Controller {
            controller_k0s_config(net, node, cp_count)
        } else {
            None
        };
        // Mint the join token once and cache it: re-minting every tick churns k0s server tokens and
        // races the join. Reuse the cached token on later ticks until the node joins.
        let token = match join_tokens.get(&node.name) {
            Some(cached) => cached.clone(),
            None => match mint_join_token(cluster, token_role).await {
                Ok(token) => {
                    join_tokens.insert(node.name.clone(), token.clone());
                    token
                }
                Err(e) => {
                    warn!(node = %node.name, error = %e, "could not mint {token_role} join token yet; will retry");
                    errors += 1;
                    continue;
                }
            },
        };
        spawn_start_component(cluster, &node.name, role, Some(token), k0s_config, node_ip);
    }
    maybe_mark_ready(cluster, &active_cp);
    (errors, active_cp)
}

/// Flip the cluster to `Ready` on the edge when the control-plane quorum is active (partial-Ready).
/// A cluster whose control plane is up is usable; stragglers keep converging on later ticks (the
/// reconcile loop also runs while Ready). Only transitions on the edge, so an already-Ready cluster
/// does not churn a WAL write + log line each tick. Shared by both the bootstrap-down early return
/// and the end of `converge_provisioning` so the quorum decision lives in one place.
fn maybe_mark_ready(cluster: &ClusterManager, active_cp: &HashSet<String>) {
    let cp_set = cluster.k0s_state().controllers();
    if control_plane_quorum_met(&cp_set, active_cp) && cluster.k0s_state().phase != K0sPhase::Ready
    {
        match cluster.set_k0s_phase(K0sPhase::Ready, None, Vec::new(), Vec::new(), false) {
            Ok(()) => info!(
                control_planes_active = active_cp.len(),
                "k0s control-plane quorum reached -> Ready (workers converge in background)"
            ),
            Err(e) => warn!(error = %e, "failed to mark k0s cluster Ready"),
        }
    }
}

/// Whether a quorum of the control-plane set has its k0s unit `active`: at least `quorum(cp_set.len())`
/// of the control-plane nodes are in `active`. This is measured from systemd unit liveness (a proxy
/// for etcd quorum, not a direct etcd health check) and is what makes the cluster usable, so the
/// reconcile loop gates `Ready` on it (partial-Ready: workers converge in background). An empty
/// control-plane set is never a quorum.
fn control_plane_quorum_met(cp_set: &[String], active: &HashSet<String>) -> bool {
    let need = spur_core::k0s::quorum(cp_set.len());
    if need == 0 {
        return false;
    }
    let have = cp_set.iter().filter(|n| active.contains(*n)).count();
    have >= need
}

/// Union `add` into the current member set, sorted + deduped. Used to grow a scoped cluster's
/// `member_nodes` online (`spur k8s add-nodes`). Adding an already-present node is a no-op. Callers
/// must not use this to "add" to a whole-inventory cluster (empty `member_nodes` = all nodes) — that
/// would narrow the scope; the RPC handler guards that case.
pub(crate) fn merge_member_nodes(current: &[String], add: &[String]) -> Vec<String> {
    let mut set: HashSet<String> = current.iter().cloned().collect();
    set.extend(add.iter().cloned());
    let mut out: Vec<String> = set.into_iter().collect();
    out.sort();
    out
}

/// Remove `drop` from the current member set, keeping it sorted. Used to shrink a scoped cluster's
/// `member_nodes` online (`spur k8s remove-nodes`). Removing an absent node is a no-op. Note: this
/// can return an empty vec — the caller must not let a scoped cluster shrink to empty, since empty
/// `member_nodes` means "whole inventory" (the RPC handler guards that).
pub(crate) fn subtract_member_nodes(current: &[String], drop: &[String]) -> Vec<String> {
    let drop: HashSet<&String> = drop.iter().collect();
    let mut out: Vec<String> = current
        .iter()
        .filter(|n| !drop.contains(n))
        .cloned()
        .collect();
    out.sort();
    out
}

/// Drop a node's stale degrade reason once it is healthy again, so status reports honestly on retry.
/// Guarded so a converged cluster does not churn a WAL write per tick.
fn clear_node_error(cluster: &ClusterManager, node: &spur_core::node::Node) {
    if node.k0s_last_error.is_none() {
        return;
    }
    if let Err(e) = cluster.set_node_k0s_error(&node.name, None) {
        warn!(node = %node.name, error = %e, "failed to clear k0s node error");
    }
}

/// Record a node's k0s error reason so it surfaces in `spur k8s status`. A failed WAL write is logged
/// rather than dropped silently — otherwise the reason for a fail-loud abort would vanish on a Raft
/// write error, leaving status with no explanation.
fn record_node_k0s_error(cluster: &ClusterManager, node: &str, reason: &str) {
    if let Err(e) = cluster.set_node_k0s_error(node, Some(reason.to_string())) {
        warn!(node = %node, error = %e, "failed to record k0s node error");
    }
}

/// Provisioning blew its deadline. Decide by control-plane quorum (partial-Ready), reusing the
/// `active_cp` liveness `converge_provisioning` already gathered this tick (no second status sweep):
/// if a quorum of control planes is active the cluster is usable, so flip it `Ready` right here (the
/// converge early-return path may not have) and let the stragglers keep converging. Only when the
/// quorum is unmet is the cluster truly stuck — record each non-active node's reason, stop its
/// half-started unit (non-reset, keeping the role so `spur k8s up` can retry), and mark `Degraded`.
async fn degrade_stuck_cluster(
    cluster: &ClusterManager,
    net: &ClusterNetworking,
    join_tokens: &mut HashMap<String, String>,
    active_cp: &HashSet<String>,
) {
    let timeout_secs = net.provisioning_timeout.as_secs();

    // Partial-Ready: a control-plane quorum means the cluster is usable. Flip Ready (idempotent on
    // the edge) instead of degrading, and leave the stragglers converging.
    let cp_set = cluster.k0s_state().controllers();
    if control_plane_quorum_met(&cp_set, active_cp) {
        maybe_mark_ready(cluster, active_cp);
        warn!(
            timeout_secs,
            "k0s provisioning past deadline but control-plane quorum holds; \
             staying up, stragglers keep converging (see `spur k8s status` for per-node reasons)"
        );
        return;
    }

    // Control-plane quorum is unmet: the cluster genuinely cannot come up. Record why each non-active
    // node blocked convergence, stop its half-started unit (keep the role for a retry), and degrade.
    for node in cluster.get_nodes() {
        if node.k0s_role.is_none() {
            continue;
        }
        let state = fetch_component_state(cluster, &node.name).await;
        if state.as_deref() == Some("active") {
            clear_node_error(cluster, &node);
            continue;
        }
        let observed = state.as_deref().unwrap_or("unreachable");
        let reason = format!("not active after {timeout_secs}s (component {observed})");
        if let Err(e) = cluster.set_node_k0s_error(&node.name, Some(reason)) {
            warn!(node = %node.name, error = %e, "failed to record k0s node error");
        }
        spawn_stop_component(cluster, &node.name, false);
    }
    join_tokens.clear();
    match cluster.set_k0s_phase(K0sPhase::Degraded, None, Vec::new(), Vec::new(), false) {
        Ok(()) => warn!(
            timeout_secs,
            "k0s provisioning timed out with no control-plane quorum -> Degraded; \
             see `spur k8s status` for per-node reasons"
        ),
        Err(e) => warn!(error = %e, "failed to mark k0s cluster Degraded"),
    }
}

/// Upper bound on a k8s drain RPC: the caller's per-node drain timeout plus slack for the eviction
/// round-trip, so a black-holed control-plane agent can't pin the removal handler indefinitely.
const DRAIN_RPC_SLACK: Duration = Duration::from_secs(30);

/// Cordon + drain a worker's pods via a control-plane agent (which holds the admin kubeconfig). The
/// agent reports a blocked/timed-out drain in-band (`drained=false`); we return that as an error so
/// the caller can decide (retry, or proceed under `--force`). `force` is passed through to the agent
/// (kubectl `--force --disable-eviction`). The whole RPC is bounded (drain timeout + slack) so a
/// hung agent can't stall the serial removal loop.
pub(crate) async fn drain_via_control_plane(
    cluster: &ClusterManager,
    node: &str,
    timeout_secs: u32,
    force: bool,
) -> anyhow::Result<()> {
    let mut client = connect_control_plane(cluster).await?;
    let agent_default = 120u32;
    let bound = Duration::from_secs(u64::from(if timeout_secs == 0 {
        agent_default
    } else {
        timeout_secs
    })) + DRAIN_RPC_SLACK;
    let resp = tokio::time::timeout(
        bound,
        client.drain_k8s_node(DrainK8sNodeRequest {
            node: node.to_string(),
            timeout_secs,
            force,
        }),
    )
    .await
    .map_err(|_| anyhow::anyhow!("drain_k8s_node RPC for {node} timed out"))?
    .map_err(|e| anyhow::anyhow!("drain_k8s_node RPC failed: {e}"))?
    .into_inner();
    if !resp.drained {
        anyhow::bail!("drain of {node} did not complete: {}", resp.message);
    }
    Ok(())
}

/// Delete a drained+stopped worker's k8s Node object via a control-plane agent. Runs AFTER the
/// component is stopped so the kubelet can't re-register a fresh (uncordoned) node in the gap.
async fn delete_node_via_control_plane(cluster: &ClusterManager, node: &str) -> anyhow::Result<()> {
    let mut client = connect_control_plane(cluster).await?;
    let resp = tokio::time::timeout(
        AGENT_TIMEOUT,
        client.delete_k8s_node(DeleteK8sNodeRequest {
            node: node.to_string(),
        }),
    )
    .await
    .map_err(|_| anyhow::anyhow!("delete_k8s_node RPC for {node} timed out"))?
    .map_err(|e| anyhow::anyhow!("delete_k8s_node RPC failed: {e}"))?
    .into_inner();
    if !resp.deleted {
        anyhow::bail!("delete of node {node} did not complete: {}", resp.message);
    }
    Ok(())
}

/// Await a StopClusterComponent on a single node (ordered, unlike the fire-and-forget
/// `spawn_stop_component`). Used by online remove, where declassify must not race the stop. The dial
/// and RPC are bounded so a black-holed agent can't hang the removal handler. Returns an error if
/// the RPC fails or the agent reports the stop/reset did not complete.
async fn stop_component_now(
    cluster: &ClusterManager,
    node: &str,
    reset: bool,
) -> anyhow::Result<()> {
    let endpoint = agent_endpoint(cluster, node)
        .ok_or_else(|| anyhow::anyhow!("{node} has no agent address"))?;
    let fut = async {
        let mut client = crate::agent_client::connect(endpoint)
            .await
            .map_err(|e| anyhow::anyhow!("connect to agent {node} failed: {e}"))?;
        client
            .stop_cluster_component(StopClusterComponentRequest { reset })
            .await
            .map_err(|e| anyhow::anyhow!("stop_cluster_component {node} failed: {e}"))
    };
    let r = tokio::time::timeout(AGENT_TIMEOUT, fut)
        .await
        .map_err(|_| anyhow::anyhow!("stop_cluster_component {node} timed out"))??
        .into_inner();
    if !r.stopped {
        anyhow::bail!("stop/reset of {node} did not complete: {}", r.message);
    }
    Ok(())
}

/// Gracefully remove a worker: drain, then stop+reset, then delete the Node object, then declassify.
/// The ordering matters — see the step comments — and the caller must drop it from `member_nodes`
/// first (re-adding on failure) so the reconcile loop never re-enrolls a mid-removal node.
pub(crate) async fn remove_worker(
    cluster: &ClusterManager,
    node: &str,
    drain_timeout_secs: u32,
    force: bool,
) -> anyhow::Result<()> {
    if let Err(e) = drain_via_control_plane(cluster, node, drain_timeout_secs, force).await {
        if !force {
            record_remove_error(cluster, node, &format!("drain failed: {e}"));
            return Err(e);
        }
        warn!(node = %node, error = %e, "drain did not complete; proceeding under --force");
    }
    // Stop+reset before deleting the Node object: a delete while the kubelet still runs lets it
    // re-register an uncordoned node.
    if let Err(e) = stop_component_now(cluster, node, true).await {
        record_remove_error(cluster, node, &format!("stop/reset failed: {e}"));
        return Err(e);
    }
    if let Err(e) = delete_node_via_control_plane(cluster, node).await {
        record_remove_error(cluster, node, &format!("node delete failed: {e}"));
        return Err(e);
    }
    cluster.clear_node_k0s(node).map_err(|e| {
        record_remove_error(cluster, node, &format!("declassify failed: {e}"));
        anyhow::anyhow!("clear k0s role for {node}: {e}")
    })?;
    info!(node = %node, "worker removed from k0s cluster (drained + reset + deleted + declassified)");
    Ok(())
}

/// Record why a worker removal did not finish, so `spur k8s status` shows the reason instead of the
/// node silently sitting in a half-removed state. Best-effort — a failed write is only logged.
fn record_remove_error(cluster: &ClusterManager, node: &str, reason: &str) {
    if let Err(e) = cluster.set_node_k0s_error(node, Some(reason.to_string())) {
        warn!(node = %node, error = %e, "failed to record k0s remove error");
    }
}

/// Cluster teardown: keep stopping a node's component while k0s still runs, else
/// (stopped/failed/unreachable) clear its role so it is never stranded out of scheduling.
async fn stop_all_components(cluster: &ClusterManager, reset: bool) {
    for node in cluster.get_nodes() {
        if node.k0s_role.is_none() {
            continue;
        }
        let state = fetch_component_state(cluster, &node.name).await;
        let still_running = matches!(
            state.as_deref(),
            Some("active") | Some("activating") | Some("deactivating")
        );
        if still_running {
            spawn_stop_component(cluster, &node.name, reset);
            continue;
        }
        if let Err(e) = cluster.clear_node_k0s(&node.name) {
            warn!(node = %node.name, error = %e, "failed to clear k0s role after teardown");
        }
    }
}

/// Fire-and-forget StartClusterComponent to a node's agent (off the reconcile thread).
fn spawn_start_component(
    cluster: &ClusterManager,
    node: &str,
    role: K0sRole,
    join_token: Option<String>,
    k0s_config: Option<String>,
    node_ip: Option<String>,
) {
    let Some(endpoint) = agent_endpoint(cluster, node) else {
        warn!(node = %node, "no agent address; cannot start k0s component");
        return;
    };
    let node = node.to_string();
    let role = role_str(role);
    tokio::spawn(async move {
        match crate::agent_client::connect(endpoint).await {
            Ok(mut client) => {
                let req = StartClusterComponentRequest {
                    role,
                    join_token,
                    k0s_config,
                    node_ip,
                };
                if let Err(e) = client.start_cluster_component(req).await {
                    warn!(node = %node, error = %e, "start_cluster_component failed");
                }
            }
            Err(e) => warn!(node = %node, error = %e, "connect to agent failed"),
        }
    });
}

/// Fire-and-forget StopClusterComponent to a node's agent.
fn spawn_stop_component(cluster: &ClusterManager, node: &str, reset: bool) {
    let Some(endpoint) = agent_endpoint(cluster, node) else {
        return;
    };
    let node = node.to_string();
    tokio::spawn(async move {
        match crate::agent_client::connect(endpoint).await {
            Ok(mut client) => {
                match client
                    .stop_cluster_component(StopClusterComponentRequest { reset })
                    .await
                {
                    Ok(resp) => {
                        // The agent reports a failed stop/reset in-band (stopped=false): surface it
                        // so `down --reset` isn't a false success. The component stays active, so the
                        // reconcile loop retries and `spur k8s status` still shows the node.
                        let r = resp.into_inner();
                        if !r.stopped {
                            warn!(
                                node = %node,
                                detail = %r.message,
                                "k0s component stop/reset failed; teardown is partial — retrying"
                            );
                        }
                    }
                    Err(e) => warn!(node = %node, error = %e, "stop_cluster_component failed"),
                }
            }
            Err(e) => warn!(node = %node, error = %e, "connect to agent failed"),
        }
    });
}

/// Convert the spur-net mesh membership to its proto mirror for the wire.
fn to_proto_membership(mesh: &MeshMembership) -> spur_proto::proto::MeshMembership {
    spur_proto::proto::MeshMembership {
        nodes: mesh
            .nodes
            .iter()
            .map(|n| spur_proto::proto::MeshNode {
                hostname: n.hostname.clone(),
                public_key: n.public_key.clone(),
                mesh_ip: n.mesh_ip.clone(),
                endpoint: n.endpoint.clone(),
                pod_cidr: n.pod_cidr.clone(),
            })
            .collect(),
    }
}

/// Fire-and-forget ApplyMesh to a node's agent: the agent reconciles the full mesh locally
/// (prune departed peers + add/update the desired set via `wg set`). Idempotent, so the
/// level-triggered per-tick re-push is safe.
fn spawn_apply_mesh(cluster: &ClusterManager, node: &str, mesh: &MeshMembership) {
    let Some(endpoint) = agent_endpoint(cluster, node) else {
        warn!(node = %node, "no agent address; cannot push mesh");
        return;
    };
    let node = node.to_string();
    let proto = to_proto_membership(mesh);
    tokio::spawn(async move {
        // Bound connect + RPC so a hung/blackholed agent can't leak accumulating detached tasks
        // (this fires every reconcile tick).
        let fut = async {
            let mut client = crate::agent_client::connect(endpoint)
                .await
                .map_err(|e| tonic::Status::unavailable(e.to_string()))?;
            client.apply_mesh(proto).await
        };
        match tokio::time::timeout(AGENT_TIMEOUT, fut).await {
            Ok(Ok(resp)) => {
                let r = resp.into_inner();
                if !r.applied {
                    warn!(node = %node, message = %r.message, "apply_mesh not applied");
                }
            }
            Ok(Err(e)) => warn!(node = %node, error = %e, "apply_mesh RPC failed"),
            Err(_) => warn!(node = %node, "apply_mesh timed out"),
        }
    });
}

/// Per-node status with LIVE component_state fetched from each agent (for the ClusterStatus RPC).
pub async fn live_node_statuses(cluster: &ClusterManager) -> Vec<ClusterNodeStatus> {
    let mut out = Vec::new();
    for n in cluster.get_nodes() {
        let Some(role) = n.k0s_role else { continue };
        // Report the agent's real (state, enabled) — not a hard-coded enabled=true.
        let (component_state, enabled) = fetch_component_status(cluster, &n.name)
            .await
            .unwrap_or_else(|| ("unknown".to_string(), false));
        out.push(ClusterNodeStatus {
            node: n.name,
            role: role_str(role),
            component_state,
            enabled,
            reason: n.k0s_last_error.unwrap_or_default(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active_set(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn cp_quorum_met_for_single_cp() {
        let cp = vec!["cp-1".to_string()];
        assert!(control_plane_quorum_met(&cp, &active_set(&["cp-1"])));
        assert!(!control_plane_quorum_met(&cp, &active_set(&[])));
    }

    #[test]
    fn cp_quorum_met_needs_majority_of_three() {
        let cp = vec!["cp-1".to_string(), "cp-2".to_string(), "cp-3".to_string()];
        // 2 of 3 is quorum; 1 of 3 is not.
        assert!(control_plane_quorum_met(
            &cp,
            &active_set(&["cp-1", "cp-2"])
        ));
        assert!(!control_plane_quorum_met(&cp, &active_set(&["cp-1"])));
        // A down worker does not affect the count; extra actives outside the CP set are ignored.
        assert!(control_plane_quorum_met(
            &cp,
            &active_set(&["cp-1", "cp-3", "worker-9"])
        ));
    }

    #[test]
    fn cp_quorum_met_needs_majority_of_five() {
        let cp: Vec<String> = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(control_plane_quorum_met(&cp, &active_set(&["a", "b", "c"])));
        assert!(!control_plane_quorum_met(&cp, &active_set(&["a", "b"])));
    }

    #[test]
    fn cp_quorum_not_met_for_empty_cp_set() {
        assert!(!control_plane_quorum_met(&[], &active_set(&[])));
    }

    #[test]
    fn merge_member_nodes_unions_and_sorts() {
        let cur = names(&["b", "a"]);
        let add = names(&["c", "a"]); // "a" already present
        assert_eq!(merge_member_nodes(&cur, &add), names(&["a", "b", "c"]));
    }

    #[test]
    fn merge_member_nodes_all_present_is_unchanged() {
        let cur = names(&["a", "b"]);
        assert_eq!(merge_member_nodes(&cur, &names(&["a"])), names(&["a", "b"]));
    }

    #[test]
    fn subtract_member_nodes_removes_and_keeps_sorted() {
        let cur = names(&["a", "b", "c"]);
        assert_eq!(
            subtract_member_nodes(&cur, &names(&["b"])),
            names(&["a", "c"])
        );
    }

    #[test]
    fn subtract_member_nodes_absent_is_unchanged() {
        let cur = names(&["a", "c"]);
        assert_eq!(
            subtract_member_nodes(&cur, &names(&["x"])),
            names(&["a", "c"])
        );
    }

    #[test]
    fn subtract_member_nodes_can_empty_the_set() {
        // Pure-function contract: subtraction can return empty. Preventing a scoped cluster from
        // actually shrinking to empty (which would flip it to whole-inventory) is the handler's job —
        // see `cluster_remove_nodes_rejects_emptying_the_member_set` in server.rs.
        let cur = names(&["a"]);
        assert_eq!(
            subtract_member_nodes(&cur, &names(&["a"])),
            Vec::<String>::new()
        );
    }

    #[test]
    fn provisioning_clock_arms_then_trips_at_deadline() {
        let start = Instant::now();
        let timeout = Duration::from_secs(600);
        let mut since = None;
        // First Provisioning observation arms the clock; not yet timed out.
        assert!(!update_provisioning_clock(
            K0sPhase::Provisioning,
            &mut since,
            start,
            timeout
        ));
        assert!(since.is_some());
        // Before the deadline: still false. At/after: true.
        assert!(!update_provisioning_clock(
            K0sPhase::Provisioning,
            &mut since,
            start + Duration::from_secs(599),
            timeout
        ));
        assert!(update_provisioning_clock(
            K0sPhase::Provisioning,
            &mut since,
            start + timeout,
            timeout
        ));
    }

    #[test]
    fn provisioning_clock_resets_off_provisioning() {
        let start = Instant::now();
        let timeout = Duration::from_secs(600);
        let mut since = Some(start);
        // A non-Provisioning phase disarms the clock and never reports timed out.
        assert!(!update_provisioning_clock(
            K0sPhase::Ready,
            &mut since,
            start + timeout,
            timeout
        ));
        assert!(since.is_none());
        // Re-entering Provisioning re-arms from the new now, not the old start.
        assert!(!update_provisioning_clock(
            K0sPhase::Provisioning,
            &mut since,
            start + timeout,
            timeout
        ));
        assert_eq!(since, Some(start + timeout));
    }

    #[test]
    fn carve_pod_cidr_from_16() {
        assert_eq!(carve_pod_cidr("10.42.0.0/16", 0).unwrap(), "10.42.0.0/24");
        assert_eq!(carve_pod_cidr("10.42.0.0/16", 2).unwrap(), "10.42.2.0/24");
        assert_eq!(
            carve_pod_cidr("10.42.0.0/16", 255).unwrap(),
            "10.42.255.0/24"
        );
        // /16 has exactly 256 /24s -> ordinal 256 overflows.
        assert!(carve_pod_cidr("10.42.0.0/16", 256).is_err());
        // /25 is too small to carve a /24.
        assert!(carve_pod_cidr("10.42.0.0/25", 0).is_err());
    }

    #[test]
    fn pod_ordinal_inverts_carve() {
        let base: Ipv4Addr = "10.42.0.0".parse().unwrap();
        assert_eq!(pod_ordinal("10.42.0.0/24", base), Some(0));
        assert_eq!(pod_ordinal("10.42.7.0/24", base), Some(7));
        // below the pod base -> None (not one of ours)
        assert_eq!(pod_ordinal("10.41.0.0/24", base), None);
    }

    #[test]
    fn next_free_ordinal_skips_used() {
        let used: HashSet<u32> = [0, 1, 3].into_iter().collect();
        assert_eq!(next_free_ordinal(&used), 2);
        assert_eq!(next_free_ordinal(&HashSet::new()), 0);
    }

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn resolve_cp_set_replicas_picks_lowest_bootstrap_first() {
        let set =
            resolve_control_plane_set(names(&["d", "b", "a", "c", "e"]), &[], None, 3).unwrap();
        // lowest 3 names, sorted; no pin so bootstrap-first is a no-op.
        assert_eq!(set, names(&["a", "b", "c"]));
    }

    #[test]
    fn resolve_cp_set_pins_bootstrap_to_front_and_into_the_set() {
        let set =
            resolve_control_plane_set(names(&["a", "b", "c", "d"]), &[], Some("c"), 3).unwrap();
        assert_eq!(set[0], "c", "pinned bootstrap leads (holds .1)");
        assert_eq!(set.len(), 3);
        assert!(set.contains(&"a".to_string()) && set.contains(&"b".to_string()));
    }

    #[test]
    fn resolve_cp_set_explicit_list_wins_and_orders_bootstrap() {
        let set = resolve_control_plane_set(
            names(&["a", "b", "c", "d", "e"]),
            &names(&["e", "c", "a"]),
            Some("c"),
            5, // ignored when explicit list is present
        )
        .unwrap();
        assert_eq!(set[0], "c");
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn resolve_cp_set_rejects_even_count() {
        assert!(resolve_control_plane_set(names(&["a", "b"]), &[], None, 2).is_err());
        assert!(
            resolve_control_plane_set(names(&["a", "b", "c", "d"]), &names(&["a", "b"]), None, 1)
                .is_err(),
            "explicit even list rejected"
        );
    }

    #[test]
    fn resolve_cp_set_rejects_more_cps_than_nodes() {
        let err = resolve_control_plane_set(names(&["a", "b"]), &[], None, 3).unwrap_err();
        assert!(err.contains("only 2 node"), "got: {err}");
    }

    #[test]
    fn resolve_cp_set_rejects_unknown_or_duplicate_explicit_node() {
        assert!(
            resolve_control_plane_set(names(&["a", "b", "c"]), &names(&["a", "x", "c"]), None, 3)
                .is_err(),
            "unknown node rejected"
        );
        assert!(
            resolve_control_plane_set(names(&["a", "b"]), &names(&["a", "a", "b"]), None, 3)
                .is_err(),
            "duplicate node rejected"
        );
    }

    #[test]
    fn resolve_cp_set_rejects_pinned_bootstrap_outside_explicit_list() {
        // A recorded/overridden bootstrap not in the requested set would silently move `.1`.
        let err = resolve_control_plane_set(
            names(&["a", "b", "c", "d"]),
            &names(&["a", "b", "c"]),
            Some("d"),
            3,
        )
        .unwrap_err();
        assert!(err.contains("bootstrap control-plane d"), "got: {err}");
        // In-list pinned bootstrap is fine and leads the set.
        let set = resolve_control_plane_set(
            names(&["a", "b", "c"]),
            &names(&["a", "b", "c"]),
            Some("b"),
            3,
        )
        .unwrap();
        assert_eq!(set[0], "b");
    }

    #[test]
    fn resolve_cp_set_legacy_single_cp_reup_is_idempotent() {
        // A 1-CP cluster re-upped with replicas=1 resolves to the same single-node set (guard allows).
        let set = resolve_control_plane_set(names(&["a", "b"]), &[], Some("a"), 1).unwrap();
        assert_eq!(set, names(&["a"]));
    }

    #[test]
    fn resolve_cp_set_rejects_singular_pin_outside_candidates() {
        // With candidates narrowed to the member scope, a --control-plane-node outside it must error
        // rather than be silently dropped and a different in-scope node elected.
        let err = resolve_control_plane_set(names(&["a", "b"]), &[], Some("z"), 1).unwrap_err();
        assert!(
            err.contains("control-plane node z is not among the selected"),
            "got: {err}"
        );
    }

    #[test]
    fn first_host_is_dot_one() {
        assert_eq!(
            first_host("10.44.0.0/16").unwrap(),
            "10.44.0.1".parse::<Ipv4Addr>().unwrap()
        );
    }

    fn scope_node(name: &str, parts: &[&str], labels: &[(&str, &str)]) -> spur_core::node::Node {
        let mut n = spur_core::node::Node::new(name.to_string(), Default::default());
        n.partitions = parts.iter().map(|s| s.to_string()).collect();
        n.labels = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        n
    }

    fn sel(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn resolve_members_empty_selection_is_whole_inventory() {
        let nodes = vec![scope_node("a", &[], &[]), scope_node("b", &[], &[])];
        assert_eq!(
            resolve_member_nodes(&nodes, "", "", &HashMap::new()).unwrap(),
            Vec::<String>::new(),
            "no selection = empty = whole inventory"
        );
    }

    #[test]
    fn resolve_members_hostlist_expands_and_sorts() {
        let nodes = vec![
            scope_node("gpu01", &[], &[]),
            scope_node("gpu02", &[], &[]),
            scope_node("gpu03", &[], &[]),
        ];
        let out = resolve_member_nodes(&nodes, "gpu[01-02]", "", &HashMap::new()).unwrap();
        assert_eq!(out, names(&["gpu01", "gpu02"]));
    }

    #[test]
    fn resolve_members_hostlist_rejects_unregistered() {
        let nodes = vec![scope_node("a", &[], &[])];
        let err = resolve_member_nodes(&nodes, "a,ghost", "", &HashMap::new()).unwrap_err();
        assert!(err.contains("ghost is not a registered node"), "got: {err}");
    }

    #[test]
    fn resolve_members_partition_selects_members() {
        let nodes = vec![
            scope_node("a", &["gpu"], &[]),
            scope_node("b", &["cpu"], &[]),
            scope_node("c", &["gpu"], &[]),
        ];
        let out = resolve_member_nodes(&nodes, "", "gpu", &HashMap::new()).unwrap();
        assert_eq!(out, names(&["a", "c"]));
    }

    #[test]
    fn resolve_members_empty_partition_rejected() {
        let nodes = vec![scope_node("a", &["gpu"], &[])];
        let err = resolve_member_nodes(&nodes, "", "nope", &HashMap::new()).unwrap_err();
        assert!(err.contains("partition nope has no"), "got: {err}");
    }

    #[test]
    fn resolve_members_selector_matches_all_pairs() {
        let nodes = vec![
            scope_node("a", &[], &[("zone", "z1"), ("gpu", "mi300")]),
            scope_node("b", &[], &[("zone", "z1"), ("gpu", "mi200")]),
            scope_node("c", &[], &[("zone", "z2"), ("gpu", "mi300")]),
        ];
        let out = resolve_member_nodes(&nodes, "", "", &sel(&[("zone", "z1"), ("gpu", "mi300")]))
            .unwrap();
        assert_eq!(out, names(&["a"]), "only the node matching BOTH pairs");
    }

    #[test]
    fn resolve_members_union_dedups_across_surfaces() {
        let nodes = vec![
            scope_node("a", &["gpu"], &[("fast", "1")]),
            scope_node("b", &["gpu"], &[]),
            scope_node("c", &[], &[("fast", "1")]),
            scope_node("d", &[], &[]),
        ];
        // hostlist {a} ∪ partition gpu {a,b} ∪ selector fast=1 {a,c} = {a,b,c}, a not duplicated.
        let out = resolve_member_nodes(&nodes, "a", "gpu", &sel(&[("fast", "1")])).unwrap();
        assert_eq!(out, names(&["a", "b", "c"]));
    }

    #[test]
    fn resolve_members_selector_no_match_rejected() {
        let nodes = vec![scope_node("a", &[], &[("zone", "z1")])];
        let err = resolve_member_nodes(&nodes, "", "", &sel(&[("zone", "z9")])).unwrap_err();
        assert!(err.contains("matched no registered nodes"), "got: {err}");
    }

    #[test]
    fn resolve_members_bogus_selector_rejected_even_when_other_surface_matches() {
        // A supplied selector that matches nothing must error even if --nodes/--partition matched,
        // so a typo'd selector isn't silently ignored.
        let nodes = vec![scope_node("a", &["gpu"], &[("zone", "z1")])];
        let err = resolve_member_nodes(&nodes, "a", "", &sel(&[("zone", "z9")])).unwrap_err();
        assert!(err.contains("--selector matched no"), "got: {err}");
    }

    fn mesh_node(
        name: &str,
        mesh_ip: Option<&str>,
        pubkey: Option<&str>,
        addr: Option<&str>,
        pod: Option<&str>,
    ) -> spur_core::node::Node {
        let mut n = spur_core::node::Node::new(name.to_string(), Default::default());
        n.k0s_mesh_ip = mesh_ip.map(String::from);
        n.wg_pubkey = pubkey.map(String::from);
        n.address = addr.map(String::from);
        n.k0s_pod_cidr = pod.map(String::from);
        n
    }

    #[test]
    fn mesh_membership_skips_unmeshed_and_carries_pod_cidr() {
        let nodes = vec![
            // controller: meshed, pod CIDR set
            mesh_node(
                "cp",
                Some("10.44.0.1"),
                Some("pk-cp"),
                Some("198.51.100.1"),
                Some("10.42.0.0/24"),
            ),
            // worker: meshed
            mesh_node(
                "w2",
                Some("10.44.0.2"),
                Some("pk-w2"),
                Some("198.51.100.2"),
                Some("10.42.1.0/24"),
            ),
            // assigned a mesh IP but hasn't reported a pubkey yet -> not on the mesh, skip
            mesh_node("w3", Some("10.44.0.3"), None, Some("198.51.100.3"), None),
            // empty pubkey is treated as absent -> skip
            mesh_node(
                "w4",
                Some("10.44.0.4"),
                Some(""),
                Some("198.51.100.4"),
                None,
            ),
            // no k0s mesh IP and an out-of-mesh (underlay) address -> skip
            mesh_node("w5", None, Some("pk-w5"), Some("198.51.100.5"), None),
        ];
        let m = mesh_from_nodes(nodes, "10.44.0.0/16", &std::collections::HashMap::new());
        assert_eq!(m.nodes.len(), 2, "only fully-meshed nodes included");
        // sorted by mesh_ip
        assert_eq!(m.nodes[0].mesh_ip, "10.44.0.1");
        assert_eq!(m.nodes[0].public_key, "pk-cp");
        // no endpoint known for this peer -> left empty; apply_mesh preserves the existing tunnel.
        assert_eq!(m.nodes[0].endpoint, "");
        assert_eq!(m.nodes[0].pod_cidr.as_deref(), Some("10.42.0.0/24"));
        assert_eq!(m.nodes[1].mesh_ip, "10.44.0.2");
        // the resulting membership feeds apply_mesh: pod CIDR folds into AllowedIPs
        assert_eq!(
            spur_net::mesh::peer_allowed_ips(&m.nodes[1]),
            "10.44.0.2/32,10.42.1.0/24"
        );
    }

    #[test]
    fn mesh_membership_includes_meshed_node_without_k0s_role() {
        // The controller/head (and login nodes) join the mesh via `net join` but never get a
        // k0s role, so `k0s_mesh_ip`/`k0s_pod_cidr` are None. They must still be in the
        // membership — derived from the mesh-range address they advertise — or the agent's
        // reconcile prunes them and severs the control-plane path.
        let nodes = vec![
            // controller: meshed (pubkey + spur0 address in mesh range), no k0s role
            mesh_node("cp", None, Some("pk-cp"), Some("10.44.0.1"), None),
            // worker: assigned a k0s role
            mesh_node(
                "w2",
                Some("10.44.0.2"),
                Some("pk-w2"),
                Some("10.44.0.2"),
                Some("10.42.1.0/24"),
            ),
        ];
        // Controller knows w2's underlay endpoint from its own peer table; it must be folded in.
        let endpoints = std::collections::HashMap::from([(
            "pk-w2".to_string(),
            "198.51.100.2:51820".to_string(),
        )]);
        let m = mesh_from_nodes(nodes, "10.44.0.0/16", &endpoints);
        assert_eq!(
            m.nodes.len(),
            2,
            "controller kept despite having no k0s role"
        );
        assert_eq!(m.nodes[0].mesh_ip, "10.44.0.1");
        assert_eq!(m.nodes[0].public_key, "pk-cp");
        assert_eq!(m.nodes[0].pod_cidr, None);
        // cp has no endpoint in the map -> empty; w2's underlay endpoint is carried through.
        assert_eq!(m.nodes[0].endpoint, "");
        assert_eq!(m.nodes[1].endpoint, "198.51.100.2:51820");
        // /32-only peer: no pod CIDR folded in.
        assert_eq!(
            spur_net::mesh::peer_allowed_ips(&m.nodes[0]),
            "10.44.0.1/32"
        );
    }

    fn test_net(wg_enabled: bool, cni: &str) -> ClusterNetworking {
        ClusterNetworking {
            wg_enabled,
            mesh_cidr: "10.44.0.0/16".into(),
            mesh_interface: "spur0".into(),
            pod_cidr: "10.42.0.0/16".into(),
            service_cidr: "10.43.0.0/16".into(),
            cni_mtu: 1450,
            cni: cni.into(),
            control_plane_node: None,
            provisioning_timeout: Duration::from_secs(600),
        }
    }

    #[test]
    fn calico_node_address_uses_mesh_ip_when_meshed() {
        let net = test_net(true, "calico");
        let mut n = spur_core::node::Node::new("cp".into(), Default::default());
        n.k0s_mesh_ip = Some("10.44.0.1".into());
        n.address = Some("203.0.113.9".into());
        assert_eq!(
            calico_node_address(net.mesh_native(), &n),
            Some("10.44.0.1")
        );
    }

    #[test]
    fn calico_node_address_falls_back_to_underlay_without_mesh() {
        let net = test_net(false, "calico");
        let mut n = spur_core::node::Node::new("cp".into(), Default::default());
        // A pool-allocated mesh IP can still be present (assigned regardless of wg_enabled), but it's
        // never bound to a real interface here, so it must not be used.
        n.k0s_mesh_ip = Some("10.44.0.1".into());
        n.address = Some("203.0.113.9".into());
        assert_eq!(
            calico_node_address(net.mesh_native(), &n),
            Some("203.0.113.9")
        );
    }

    #[test]
    fn calico_node_address_rejects_an_fqdn_underlay_address() {
        let net = test_net(false, "calico");
        let mut n = spur_core::node::Node::new("cp".into(), Default::default());
        // Node.address may be an FQDN (docs/deployment/native-host.rst), but kubelet --node-ip
        // requires a literal IP -- must defer rather than pass a hostname through.
        n.address = Some("cp.example.internal".into());
        assert_eq!(calico_node_address(net.mesh_native(), &n), None);
    }

    #[test]
    fn controller_k0s_config_picks_vxlan_and_underlay_api_without_mesh() {
        let net = test_net(false, "calico");
        let mut n = spur_core::node::Node::new("cp".into(), Default::default());
        n.k0s_mesh_ip = Some("10.44.0.1".into());
        n.address = Some("203.0.113.9".into());
        let y = controller_k0s_config(&net, &n, 1).unwrap();
        assert!(y.contains("address: 203.0.113.9"));
        assert!(y.contains("mode: vxlan"));
        assert!(
            !y.contains("10.44.0.1"),
            "the unbound mesh IP must not leak into the config"
        );
    }

    #[test]
    fn controller_k0s_config_keeps_kuberouter_on_the_underlay_when_meshed() {
        let net = test_net(true, "kuberouter");
        let mut n = spur_core::node::Node::new("cp".into(), Default::default());
        n.k0s_mesh_ip = Some("10.44.0.1".into());
        n.address = Some("203.0.113.9".into());
        let y = controller_k0s_config(&net, &n, 1).unwrap();
        assert!(y.contains("address: 203.0.113.9"));
        assert!(y.contains("overlay-type: full"));
        assert!(
            !y.contains("10.44.0.1"),
            "kube-router tunnels over the underlay, so the mesh IP must not be advertised"
        );
    }

    #[test]
    fn controller_k0s_config_keeps_bird_and_mesh_api_when_meshed() {
        let net = test_net(true, "calico");
        let mut n = spur_core::node::Node::new("cp".into(), Default::default());
        n.k0s_mesh_ip = Some("10.44.0.1".into());
        n.address = Some("203.0.113.9".into());
        let y = controller_k0s_config(&net, &n, 1).unwrap();
        assert!(y.contains("address: 10.44.0.1"));
        assert!(y.contains("mode: bird"));
        // underlay still carried as an alternate SAN so kubectl works over either address.
        assert!(y.contains("203.0.113.9"));
    }
}
