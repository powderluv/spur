// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::Mutex;
use tonic::metadata::MetadataValue;
use tonic::{Code, Request, Response, Status};
use tracing::{info, warn};

use spur_core::job::NodeCompleteError;
use spur_core::mpi::MPI_PMIX;
use spur_core::reservation::Reservation;
use spur_core::task_launch::{
    batch_script_uses_step_launch, build_step_task_plan, step_needs_pmix_prepare,
};
use spur_proto::proto::slurm_controller_client::SlurmControllerClient;
use spur_proto::proto::slurm_controller_server::SlurmController;
use spur_proto::proto::*;

use crate::accounting::{txn, TxnAction, TxnEntity, TxnRecord, TxnSource};
use crate::cluster::{ClusterManager, JobFilter, PartitionError, ReservationError};
use crate::pmix_dispatch::{self, PmixPrepareNode};
use crate::raft::RaftHandle;
use crate::rpc_middleware::RpcStatsLayer;
use crate::rpc_stats::RpcStatsCollector;
use crate::sched_stats::SchedStatsCollector;

const FORWARDED_HEADER: &str = "x-spur-forwarded";
const LEADER_HEADER: &str = "x-spur-leader";

/// Resolve the comm address for an agent registration.
///
/// Tries the advertised address first, then the gRPC peer IP (dynamic registration
/// fallback). Loopback or link-local results are deferred when a routable
/// candidate is also available.
fn resolve_registration_comm_addr(
    advertised: &str,
    remote_addr: &str,
    reject_loopback: bool,
) -> Result<String, Status> {
    let mut candidates = Vec::new();
    if !advertised.is_empty() {
        candidates.push(advertised);
    }
    if !remote_addr.is_empty() && remote_addr != advertised {
        candidates.push(remote_addr);
    }

    if candidates.is_empty() {
        return Err(Status::invalid_argument(
            "no comm address available for registration",
        ));
    }

    let mut last_err = None;
    let mut unusable_result = None;
    for candidate in candidates {
        match spur_net::validate_comm_address(candidate, reject_loopback) {
            Ok(addr) if spur_net::normalized_comm_addr_is_unusable(&addr) => {
                if unusable_result.is_none() {
                    unusable_result = Some((candidate.to_string(), addr));
                }
            }
            Ok(addr) => {
                if candidate != addr {
                    info!(
                        candidate = %candidate,
                        comm_addr = %addr,
                        "normalized comm address for registration"
                    );
                }
                return Ok(addr);
            }
            Err(e) => last_err = Some(e),
        }
    }

    if let Some((candidate, addr)) = unusable_result {
        warn!(
            comm_addr = %addr,
            candidate = %candidate,
            "node registered with loopback or link-local comm address"
        );
        return Ok(addr);
    }

    Err(Status::invalid_argument(match last_err {
        Some(e) => format!("invalid comm address: {e}"),
        None => "no comm address candidate resolved".into(),
    }))
}

fn node_comm_http_url(node: &spur_core::node::Node, node_name: &str) -> Result<String, Status> {
    let host = node
        .comm_addr()
        .ok_or_else(|| Status::unavailable(format!("node {node_name} has no comm address")))?;
    Ok(spur_net::format_comm_http_url(host, node.port))
}

fn node_comm_socket(node: &spur_core::node::Node, node_name: &str) -> Result<String, Status> {
    let host = node
        .comm_addr()
        .ok_or_else(|| Status::unavailable(format!("node {node_name} has no comm address")))?;
    Ok(spur_net::format_comm_socket(host, node.port))
}

/// Forwarding decision for read RPCs, split out so it's unit-testable.
fn read_forwarding_policy(is_leader: bool, is_forwarded: bool) -> bool {
    !is_leader && !is_forwarded
}

pub struct ControllerService {
    cluster: Arc<ClusterManager>,
    raft: Arc<RaftHandle>,
    leader_proxy: LeaderProxy,
    /// Node ID → client API address (host:6817) for the x-spur-leader header.
    client_addrs: BTreeMap<u64, String>,
    rpc_stats: Arc<RpcStatsCollector>,
    sched_stats: Arc<SchedStatsCollector>,
    /// Default HA control-plane count (`[cluster] control_plane_replicas`) when `spur k8s up`
    /// requests neither `--replicas` nor an explicit node set.
    control_plane_replicas: u32,
    /// JWT signing key for node tokens, captured at startup. Deliberately NOT
    /// re-read on `scontrol reconfigure`: swapping it live would instantly fail
    /// verification of every outstanding node token (7-day TTL), silently
    /// partitioning healthy nodes. Like Slurm's AuthType, it is restart-only.
    jwt_key: String,
}

struct LeaderProxy {
    raft: Arc<RaftHandle>,
    client_addrs: BTreeMap<u64, String>,
    cached_client: Mutex<Option<(u64, SlurmControllerClient<tonic::transport::Channel>)>>,
}

impl LeaderProxy {
    fn new(raft: Arc<RaftHandle>, client_addrs: BTreeMap<u64, String>) -> Self {
        Self {
            raft,
            client_addrs,
            cached_client: Mutex::new(None),
        }
    }

    async fn get_leader_client(
        &self,
    ) -> Result<SlurmControllerClient<tonic::transport::Channel>, Status> {
        let leader_id = self
            .raft
            .current_leader()
            .ok_or_else(|| Status::unavailable("no leader elected yet"))?;

        let mut cached = self.cached_client.lock().await;

        if let Some((id, ref client)) = *cached {
            if id == leader_id {
                return Ok(client.clone());
            }
        }

        let addr = self
            .client_addrs
            .get(&leader_id)
            .ok_or_else(|| Status::unavailable("leader address unknown"))?;

        let url = if addr.starts_with("http") {
            addr.clone()
        } else {
            format!("http://{}", addr)
        };

        let client = SlurmControllerClient::connect(url)
            .await
            .map_err(|e| Status::unavailable(format!("cannot reach leader: {e}")))?
            .max_decoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(spur_proto::MAX_GRPC_REQUEST_SIZE);

        *cached = Some((leader_id, client.clone()));
        Ok(client)
    }

    /// `None` instead of an error when no leader is reachable, so read RPCs can
    /// fall back to local state rather than failing.
    async fn try_get_leader_client(
        &self,
    ) -> Option<SlurmControllerClient<tonic::transport::Channel>> {
        self.get_leader_client().await.ok()
    }
}

/// Resolve the node-token signing key from config at startup. Captured once by
/// `serve` into `ControllerService::jwt_key`; deliberately not re-read on
/// `reconfigure` (see the field doc). Falls back to a shared default so
/// key-less dev clusters interoperate.
fn resolve_startup_jwt_key(config: &spur_core::config::SlurmConfig) -> String {
    if let Some(key) = &config.auth.jwt_key {
        return key.clone();
    }
    // Token admission signs/verifies node tokens with this key. A well-known
    // default is trivially forgeable by anyone who can reach the controller.
    if matches!(
        config.admission.mode,
        spur_core::config::AdmissionMode::Token
    ) {
        warn!(
            "admission.mode=Token but auth.jwt_key is unset: node tokens are signed with a \
             well-known default key and are forgeable. Set auth.jwt_key to a secret value."
        );
    }
    "spur-default-key".to_string()
}

impl ControllerService {
    // tonic::Status is 176 bytes (over clippy's 128-byte threshold); fixed upstream in tonic 0.13+
    #[allow(clippy::result_large_err)]
    fn check_leader<T>(&self, request: &Request<T>) -> Result<(), Status> {
        if self.raft.is_leader() {
            return Ok(());
        }

        if request.metadata().get(FORWARDED_HEADER).is_some() {
            return Err(self.not_leader_status());
        }

        Err(self.not_leader_status())
    }

    /// Re-send a cancel to a node still holding an allocation the controller no
    /// longer believes belongs to it there — job finished, job moved to
    /// another node, or an id older than any this controller issued. Best-effort.
    fn reclaim_stale_agent_jobs(&self, node: &str, reported: &[RunningJobStatus]) {
        let stale = stale_reported_jobs(&self.cluster, node, reported);
        if stale.is_empty() {
            return;
        }
        let cluster = self.cluster.clone();
        let node = node.to_string();
        tokio::spawn(async move {
            for job_id in stale {
                // Re-check: a requeue since the snapshot above would otherwise
                // send an unguarded cancel into the job's new run.
                if !is_reclaimable(&cluster, &node, job_id) {
                    continue;
                }
                warn!(
                    job_id,
                    node = %node,
                    "agent still holds an allocation the controller no longer believes belongs to it there — re-sending cancel to reclaim it"
                );
                // Signal 0 is a no-op on an unknown id, but SIGTERM/SIGKILLs a
                // live process for the active-elsewhere case. Not epoch-gated.
                crate::scheduler_loop::cancel_job_on_nodes(
                    &cluster,
                    job_id,
                    std::slice::from_ref(&node),
                    0,
                )
                .await;
            }
        });
    }

    /// Feed a node's heartbeat-reported k0s status into the metric accumulator.
    fn record_k0s_node_status(&self, node: &str, status: &spur_proto::proto::K0sNodeStatus) {
        let cluster = self.cluster.config().cluster_name.clone();
        let metrics = self.cluster.k8s_metrics();
        metrics.set_node_up(&cluster, node, status.unit_active);
        metrics.set_node_restart_total(&cluster, node, status.restart_count);
        if status.install_duration_seconds > 0.0 {
            metrics.observe_node_install_duration(&cluster, node, status.install_duration_seconds);
        }
    }

    /// Reads never require the leader (every node applies the committed log),
    /// so forwarding is only an optional freshness hop. Skipping already-
    /// forwarded requests avoids forward loops between non-leaders.
    fn read_should_forward<T>(&self, request: &Request<T>) -> bool {
        read_forwarding_policy(
            self.raft.is_leader(),
            request.metadata().get(FORWARDED_HEADER).is_some(),
        )
    }

    /// Best-effort forward of a read to the leader: `Some(payload)` forwards
    /// (clone lazily via `forward.then(|| ...)`), `None` serves local state. Any
    /// forward error is swallowed to local — safe only while read handlers return
    /// Ok/NotFound; a handler with a real error (e.g. InvalidArgument) masks it.
    async fn forward_read_optional<T, R, F, Fut>(
        &self,
        meta: tonic::metadata::MetadataMap,
        payload: Option<T>,
        rpc: &str,
        call: F,
    ) -> Option<Response<R>>
    where
        F: FnOnce(SlurmControllerClient<tonic::transport::Channel>, Request<T>) -> Fut,
        Fut: std::future::Future<Output = Result<Response<R>, Status>>,
    {
        let payload = payload?;
        let client = self.leader_proxy.try_get_leader_client().await?;
        let mut fwd = Request::new(payload);
        // Carry the caller's credential, so a forwarded read is authorized as the original caller
        // rather than arriving anonymous — otherwise `auth.mode = required` would reject every hop
        // and silently degrade these reads to local (stale) answers.
        *fwd.metadata_mut() = Self::forwarded_metadata_preserving(&meta);
        match call(client, fwd).await {
            Ok(resp) => Some(resp),
            Err(e) => {
                warn!("forwarding {rpc} to leader failed, serving locally: {e}");
                None
            }
        }
    }

    fn not_leader_status(&self) -> Status {
        let mut status = Status::unavailable("not the Raft leader");
        if let Some(leader_id) = self.raft.current_leader() {
            if let Some(addr) = self.client_addrs.get(&leader_id) {
                if let Ok(val) = addr.parse::<MetadataValue<tonic::metadata::Ascii>>() {
                    status.metadata_mut().insert(LEADER_HEADER, val);
                }
            }
        }
        status
    }

    fn forwarded_metadata() -> tonic::metadata::MetadataMap {
        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert(FORWARDED_HEADER, "true".parse().unwrap());
        meta
    }

    /// The verified identity for this request, if the auth layer authenticated one.
    ///
    /// `None` means the caller was not authenticated — which under `permissive` is allowed and the
    /// handler falls back to the client-asserted fields. Only [`crate::auth_middleware`] inserts
    /// this, so its presence always means "verified".
    fn verified_identity<T>(request: &Request<T>) -> Option<&spur_core::auth::Identity> {
        request.extensions().get::<spur_core::auth::Identity>()
    }

    /// Replace a client-asserted identity field with the authenticated one.
    ///
    /// Applied at handler entry so every downstream read of that field (ownership checks, admin
    /// checks, accounting attribution) sees the verified user without each call site having to know
    /// about authentication. A `None` identity leaves the field alone: that is an unauthenticated
    /// caller under `permissive`/`disabled`, and `required` never reaches a handler unauthenticated
    /// because the auth layer rejects first.
    fn authoritative_user(asserted: &mut String, identity: Option<&spur_core::auth::Identity>) {
        let Some(id) = identity else { return };
        if !asserted.is_empty() && *asserted != id.user {
            warn!(
                claimed = %asserted,
                authenticated = %id.user,
                "request asserted a different user than its credential; using the authenticated one"
            );
        }
        *asserted = id.user.clone();
    }

    /// Bind a submitted spec to the authenticated caller.
    ///
    /// Overwrites `user`/`uid`/`gid` from the verified identity rather than trusting what the client
    /// sent, and derives uid/gid from the username through NSS (the token carries no gid, and a
    /// client-chosen uid is what allowed a job to run as an arbitrary user). Unauthenticated callers
    /// are left as-is so `permissive` keeps working; `required` never reaches here without an
    /// identity because the auth layer rejects first.
    fn bind_spec_to_identity(
        spec: &mut spur_core::job::JobSpec,
        identity: Option<&spur_core::auth::Identity>,
    ) -> Result<(), Status> {
        let Some(id) = identity else { return Ok(()) };
        let (uid, gid) = spur_core::auth::resolve_unix_credentials(&id.user).map_err(|e| {
            // Fail closed: never fall back to the wire's uid (or to 0) for a user we cannot resolve.
            Status::failed_precondition(format!(
                "cannot resolve UNIX credentials for authenticated user '{}': {e}",
                id.user
            ))
        })?;
        if spec.user != id.user && !spec.user.is_empty() {
            warn!(
                claimed = %spec.user,
                authenticated = %id.user,
                "job spec claimed a different user than the credential; using the authenticated one"
            );
        }
        spec.user = id.user.clone();
        spec.uid = uid;
        spec.gid = gid;
        Ok(())
    }

    /// Whether the verified caller is an administrator for policy decisions that are neither
    /// job-ownership nor k0s-cluster gates: priority ceilings, job-info visibility.
    ///
    /// Accepts either the token's `admin` claim or the accounting `Admin` level, so the check works
    /// before the accounting DB is populated and stays consistent with the rest of the control plane
    /// (`is_k0s_admin`). An unauthenticated caller (permissive/disabled) is not an admin.
    fn caller_is_admin(&self, identity: Option<&spur_core::auth::Identity>) -> bool {
        identity.is_some_and(|id| {
            id.is_admin || is_k0s_admin(self.cluster.association_cache(), &id.user)
        })
    }

    /// Fire a best-effort audit record and a structured log line for a reservation
    /// admin action, so operators keep attribution even when the accounting DB is
    /// down. Never alters the RPC result.
    fn audit_reservation(
        &self,
        action: TxnAction,
        entity_name: &str,
        actor: &str,
        identity: Option<&spur_core::auth::Identity>,
        details_base: serde_json::Value,
        result: &Result<(), Status>,
    ) {
        let record =
            Self::build_reservation_txn(action, entity_name, actor, identity, details_base, result);
        info!(
            actor = %record.actor,
            entity = %record.entity_name,
            action = record.action.as_str(),
            outcome = record.outcome.as_str(),
            "reservation admin action"
        );
        self.cluster.record_txn(record);
    }

    /// Build the audit record from the resolved outcome and caller identity.
    /// `verified` is true only for a verified JWT identity, and the actor uid is
    /// taken from that identity — never the forgeable wire value.
    fn build_reservation_txn(
        action: TxnAction,
        entity_name: &str,
        actor: &str,
        identity: Option<&spur_core::auth::Identity>,
        details_base: serde_json::Value,
        result: &Result<(), Status>,
    ) -> TxnRecord {
        TxnRecord {
            ts: Utc::now(),
            actor: actor.to_string(),
            actor_uid: identity.map(|id| i64::from(id.uid)),
            verified: identity.is_some(),
            source: TxnSource::Api,
            action,
            entity_type: TxnEntity::Reservation,
            entity_name: entity_name.to_string(),
            outcome: txn::outcome_from_status(result),
            details: txn::finalize_details(
                details_base,
                result.as_ref().err().map(|s| s.message()),
            ),
        }
    }

    /// Parse, validate, and submit a create-reservation request. Split out so the
    /// handler audits the outcome uniformly, including `invalid_argument` parse
    /// failures that occur before the cluster call.
    fn build_and_create_reservation(&self, req: CreateReservationRequest) -> Result<(), Status> {
        let start_time = if req.start_time.is_empty() || req.start_time.eq_ignore_ascii_case("now")
        {
            Utc::now()
        } else {
            req.start_time
                .parse::<DateTime<Utc>>()
                .map_err(|e| Status::invalid_argument(format!("invalid start_time: {}", e)))?
        };
        let end_time = start_time + chrono::Duration::minutes(req.duration_minutes as i64);
        let flags = spur_core::reservation::ReservationFlags::parse_list(&req.flags)
            .map_err(Status::invalid_argument)?;
        let reservation = Reservation {
            name: req.name,
            start_time,
            end_time,
            nodes: req.nodes,
            accounts: req.accounts,
            users: req.users,
            flags,
            owner: req.user,
        };
        self.cluster
            .create_reservation(reservation)
            .map_err(reservation_rpc_status)
    }

    /// Rejects an identified non-admin from a cluster-tenancy mutation (partitions, node placement,
    /// tokens); anonymous is allowed, matching `caller_is_privileged`. Call as a prologue, before
    /// `into_inner`. Reservations use the stricter [`Self::require_reservation_manager`].
    #[allow(clippy::result_large_err)]
    fn require_admin<T>(&self, request: &Request<T>, op: &str) -> Result<(), Status> {
        if self.caller_is_privileged(Self::verified_identity(request)) {
            Ok(())
        } else {
            Err(Status::permission_denied(format!(
                "{op} requires cluster admin"
            )))
        }
    }

    /// Whether a caller is exempt from the non-admin restrictions (the priority ceiling): an admin,
    /// or a caller with no verified identity. The latter keeps the pre-auth behaviour — `disabled`,
    /// or `permissive` with no credential, trusts the client — so restricting it would break no-auth
    /// deployments. `required` never reaches here without an identity, so real users are still bound.
    fn caller_is_privileged(&self, identity: Option<&spur_core::auth::Identity>) -> bool {
        identity.is_none() || self.caller_is_admin(identity)
    }

    /// Stricter form of [`Self::require_admin`] for reservations: the admin bar, or the
    /// root-or-`sudo`/`wheel` rule the CLI states, resolved from the caller's name on this host.
    /// `require_admin` alone would be inert under the default `permissive` mode, since it waves an
    /// unidentified caller through, and would deny a `sudo`/`wheel` operator under `required`.
    async fn require_reservation_manager(
        &self,
        user: &str,
        identity: Option<&spur_core::auth::Identity>,
    ) -> Result<(), Status> {
        // Skipping the lookup for a verified admin keeps a credential working on a controller that
        // shares no user directory with the login nodes and cannot resolve the name at all.
        if self.caller_is_admin(identity) {
            return Ok(());
        }
        // NSS can reach sssd/LDAP over the network, so it must not run on a runtime worker.
        let owned = user.to_string();
        let privileged = tokio::task::spawn_blocking(move || {
            spur_core::privilege::named_user_is_privileged(&owned)
        })
        .await
        .map_err(|e| Status::internal(format!("privilege lookup failed: {e}")))?;
        reservation_manager_ruling(user, privileged)
    }

    /// Clamp a non-privileged caller's base priority to `[scheduler] max_user_priority`, mirroring
    /// Slurm's operator-only priority raising (an ordinary user may only lower it). Clamped down, not
    /// rejected, so a fat-fingered `--priority` still runs. Privileged callers and an unset priority
    /// are untouched. Returns a warning string; `submit_job` surfaces it in the response, and
    /// `update_job` logs it (UpdateJob has no response field). Operates on the raw `Option<u32>` so
    /// submit and `scontrol update` share one policy — the ceiling can't be dodged by boosting later.
    fn clamp_priority(
        priority: &mut Option<u32>,
        caller_is_privileged: bool,
        max_user_priority: u32,
    ) -> Option<String> {
        if caller_is_privileged {
            return None;
        }
        match *priority {
            Some(p) if p > max_user_priority => {
                *priority = Some(max_user_priority);
                Some(format!(
                    "requested priority {p} exceeds the non-admin ceiling {max_user_priority}; \
                     clamped to {max_user_priority} (raising priority is operator-only)"
                ))
            }
            _ => None,
        }
    }

    /// Apply `[controller] job_info_visibility` for `get_job`. Owner, admin, and unauthenticated
    /// callers (see [`viewer_is_privileged`]) get the full record; an identified non-owner gets
    /// `Full` (legacy), `None` → `NOT_FOUND` (`OwnerOnly`), or the targeting-sensitive fields blanked
    /// (`Redacted`, the default).
    /// Disclosure level for one job. Split out so the single-job and list handlers share one policy
    /// decision; the list path annotates in a single batch and so cannot reuse `scoped_job_info`.
    fn disclosure_for(
        &self,
        owner: &str,
        identity: Option<&spur_core::auth::Identity>,
    ) -> JobInfoDisclosure {
        let privileged = viewer_is_privileged(identity, owner, self.caller_is_admin(identity));
        job_info_disclosure(
            privileged,
            self.cluster.config().controller.job_info_visibility,
        )
    }

    fn scoped_job_info(
        &self,
        job: &spur_core::job::Job,
        identity: Option<&spur_core::auth::Identity>,
    ) -> Option<JobInfo> {
        match self.disclosure_for(&job.spec.user, identity) {
            JobInfoDisclosure::Hidden => None,
            disclosure => {
                let mut info = job_to_proto(job);
                // Annotate before redacting, so the redacted branch strips the
                // planned nodelist rather than having it added back after.
                annotate_jobs_with_planned_reservations(
                    std::slice::from_mut(&mut info),
                    &self.cluster,
                );
                if disclosure == JobInfoDisclosure::Redacted {
                    redact_sensitive_job_info(&mut info);
                }
                Some(info)
            }
        }
    }

    /// Forwarding metadata that carries the caller's credential through to the leader.
    ///
    /// The leader is what authorizes the request, so it must see the ORIGINAL caller — not the
    /// forwarding follower. Building a fresh metadata map (as `forwarded_metadata` does) drops the
    /// `authorization` header, which would make every forwarded call anonymous and break auth the
    /// moment HA is enabled.
    fn forwarded_metadata_preserving(
        orig: &tonic::metadata::MetadataMap,
    ) -> tonic::metadata::MetadataMap {
        let mut meta = Self::forwarded_metadata();
        if let Some(auth) = orig.get(http::header::AUTHORIZATION.as_str()) {
            meta.insert(http::header::AUTHORIZATION.as_str(), auth.clone());
        }
        meta
    }

    /// Re-wrap a request for forwarding to the leader, preserving the caller's credential.
    fn forward_request<T>(request: Request<T>) -> Request<T> {
        let meta = Self::forwarded_metadata_preserving(request.metadata());
        let mut fwd = Request::new(request.into_inner());
        *fwd.metadata_mut() = meta;
        fwd
    }

    fn spawn_cancel_for_evicted(&self, evicted: &[crate::raft::JobFinalized]) {
        for fin in evicted {
            if let Some(job) = self.cluster.get_job(fin.job_id) {
                let cluster = self.cluster.clone();
                tokio::spawn(async move {
                    crate::scheduler_loop::send_cancel_to_agents(&cluster, &job, 9).await;
                });
            }
        }
    }

    /// Validate an admission token if token mode is enabled.
    /// Returns the node_token JWT to include in the registration response,
    /// or an empty string if admission mode is open.
    #[allow(clippy::result_large_err)]
    fn validate_admission(&self, join_token: &str, hostname: &str) -> Result<String, Status> {
        use spur_core::config::AdmissionMode;

        if !matches!(self.cluster.config().admission.mode, AdmissionMode::Token) {
            return Ok(String::new());
        }

        if join_token.is_empty() {
            return Err(Status::unauthenticated("admission token required"));
        }

        let (token_id, secret) = spur_core::admission::parse_token(join_token)
            .map_err(|e| Status::permission_denied(e.to_string()))?;

        let token_store = self.cluster.get_tokens();
        spur_core::admission::validate_token(token_id, secret, &token_store)
            .map_err(|e| Status::permission_denied(e.to_string()))?;

        spur_core::admission::generate_node_token(hostname, self.jwt_key.as_bytes())
            .map_err(|e| Status::internal(e.to_string()))
    }
}

/// Resolve a user to the (namespace, ServiceAccount) its scoped kubeconfig must be bound to.
/// Fails closed if associations aren't loaded yet — the cache resolves fail-open, which would
/// otherwise mint an unscoped token — and rejects a user with no account.
fn resolve_user_namespace_sa(
    cache: &crate::association_cache::AssociationCache,
    user: &str,
) -> Result<(String, String), Status> {
    if !cache.is_loaded() {
        return Err(Status::unavailable(
            "associations not loaded yet; retry shortly",
        ));
    }
    let (account, _qos, _allowed_qos) = cache.resolve(user, None);
    let account = account.ok_or_else(|| {
        Status::not_found(format!("user '{user}' is not associated with any account"))
    })?;
    Ok((
        spur_core::quota_names::account_namespace(&account),
        spur_core::quota_names::user_service_account(user),
    ))
}

/// Whether `caller` may perform k0s cluster-admin ops: `root` always, otherwise an accounting
/// `Admin`. An empty caller is NOT privileged. Fails closed when accounting is off (the cache
/// reports no admins), leaving only `root`.
fn is_k0s_admin(cache: &crate::association_cache::AssociationCache, caller: &str) -> bool {
    // NOTE: `caller` is supplied by the client and is not authenticated, so this check is an
    // operator-error guard, not a security boundary. Anything that hands out a credential must
    // carry its own opt-in (see `cluster.allow_admin_kubeconfig`) until user auth is enforced.
    //
    // The former `caller.is_empty()` bypass is gone: nothing in-tree sends an empty caller (the CLI
    // always fills it, falling back to "unknown"), so it granted admin to a two-character forgery
    // and bought nothing. `root` is retained deliberately — with accounting disabled the cache
    // reports no admins at all, and removing it would strand `spur k8s up` on every cluster that
    // does not run accounting.
    caller == "root" || cache.is_admin(caller)
}

/// Ruling for a non-admin caller, split from the lookup so the policy is testable without the
/// host's user database. An error denies: a name the controller cannot resolve is a name it cannot
/// vouch for.
fn reservation_manager_ruling(
    user: &str,
    privileged: Result<bool, spur_core::auth::AuthError>,
) -> Result<(), Status> {
    match privileged {
        Ok(true) => Ok(()),
        Ok(false) => Err(Status::permission_denied(format!(
            "user '{user}' may not manage reservations: {}",
            spur_core::privilege::PRIVILEGE_REQUIREMENT
        ))),
        Err(e) => Err(Status::permission_denied(format!(
            "cannot verify that '{user}' may manage reservations ({e}); reservation management {}",
            spur_core::privilege::PRIVILEGE_REQUIREMENT
        ))),
    }
}

/// Whether `node` may release what it holds for `job_id`: the run is over, the
/// job is active elsewhere, or the id is untracked but was issued by us.
fn is_reclaimable(cluster: &ClusterManager, node: &str, job_id: u32) -> bool {
    match cluster.get_job(job_id) {
        Some(job) if job.state.is_terminal() => true,
        // An active job's nodelist is authoritative only once populated: state
        // and allocation commit as two separate WAL entries, so a job can be
        // observed as Running with `allocated_nodes` still empty. Spare it, same
        // as a job earlier than active that may be mid-dispatch to this node.
        Some(job) => {
            job.state.is_active()
                && !job.allocated_nodes.is_empty()
                && !job.allocated_nodes.iter().any(|n| n == node)
        }
        None => job_id < cluster.peek_next_job_id(),
    }
}

/// Reported ids `node` may release; ids still allocated here, not yet started,
/// and never issued by this controller are spared.
fn stale_reported_jobs(
    cluster: &ClusterManager,
    node: &str,
    reported: &[RunningJobStatus],
) -> Vec<u32> {
    reported
        .iter()
        .filter_map(|r| is_reclaimable(cluster, node, r.job_id).then_some(r.job_id))
        .collect()
}

#[tonic::async_trait]
impl SlurmController for ControllerService {
    async fn submit_job(
        &self,
        request: Request<SubmitJobRequest>,
    ) -> Result<Response<SubmitJobResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.submit_job(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward submit_job to leader: {e}");
                    return Err(status);
                }
            }
        }

        // Bind to the authenticated caller BEFORE the spec reaches the cluster, so a stale or
        // hostile client cannot choose the user/uid the job runs as.
        let identity = Self::verified_identity(&request).cloned();
        let spec = request
            .into_inner()
            .spec
            .ok_or_else(|| Status::invalid_argument("missing job spec"))?;

        let mut core_spec = proto_to_job_spec(spec)?;
        Self::bind_spec_to_identity(&mut core_spec, identity.as_ref())?;

        // Clamp a non-privileged caller's base priority to the configured ceiling before it reaches
        // the scheduler, so one submission cannot front-run the whole queue. Non-fatal: clamp + warn.
        let priority_warning = Self::clamp_priority(
            &mut core_spec.priority,
            self.caller_is_privileged(identity.as_ref()),
            self.cluster.config().scheduler.max_user_priority,
        );

        let outcome = self
            .cluster
            .submit_job(core_spec)
            .map_err(submit_rpc_status)?;

        let mut warnings = outcome.warnings;
        warnings.extend(priority_warning);
        Ok(Response::new(SubmitJobResponse {
            job_id: outcome.job_id,
            warnings,
        }))
    }

    async fn get_jobs(
        &self,
        request: Request<GetJobsRequest>,
    ) -> Result<Response<GetJobsResponse>, Status> {
        let forward = self.read_should_forward(&request);
        let meta = request.metadata().clone();
        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        if let Some(resp) = self
            .forward_read_optional(
                meta,
                forward.then(|| req.clone()),
                "get_jobs",
                |mut c, r| async move { c.get_jobs(r).await },
            )
            .await
        {
            return Ok(resp);
        }

        let states: Vec<spur_core::job::JobState> = req
            .states
            .iter()
            .filter_map(|s| spur_core::job::JobState::from_proto_i32(*s))
            .collect();

        let user = if req.user.is_empty() {
            None
        } else {
            Some(req.user.as_str())
        };
        let partition = if req.partition.is_empty() {
            None
        } else {
            Some(req.partition.as_str())
        };
        let account = if req.account.is_empty() {
            None
        } else {
            Some(req.account.as_str())
        };
        let name = if req.name.is_empty() {
            None
        } else {
            Some(req.name.as_str())
        };

        let jobs = self.cluster.get_jobs(&JobFilter {
            states: &states,
            user,
            partition,
            account,
            name,
            job_ids: &req.job_ids,
            nodes: &req.nodes,
        });

        let mut proto_jobs: Vec<JobInfo> = jobs.iter().map(job_to_proto).collect();
        annotate_jobs_with_planned_reservations(&mut proto_jobs, &self.cluster);

        // An identified caller is already scoped to their own jobs by the user filter above, so
        // this only bites if that scoping is ever loosened — but it keeps the two read paths from
        // disagreeing about what a non-owner may see.
        let proto_jobs: Vec<JobInfo> = jobs
            .iter()
            .zip(proto_jobs)
            .filter_map(|(job, mut info)| {
                match self.disclosure_for(&job.spec.user, __identity.as_ref()) {
                    JobInfoDisclosure::Hidden => None,
                    JobInfoDisclosure::Redacted => {
                        redact_sensitive_job_info(&mut info);
                        Some(info)
                    }
                    JobInfoDisclosure::Full => Some(info),
                }
            })
            .collect();

        Ok(Response::new(GetJobsResponse { jobs: proto_jobs }))
    }

    async fn get_job(&self, request: Request<GetJobRequest>) -> Result<Response<JobInfo>, Status> {
        let forward = self.read_should_forward(&request);
        let meta = request.metadata().clone();
        // Capture identity before the forward so the serving node (leader or read-allowed follower)
        // can scope the record to the caller; the credential is preserved on forward, so a forwarded
        // read is scoped on the leader instead.
        let identity = Self::verified_identity(&request).cloned();
        let req = request.into_inner();
        let job_id = req.job_id;
        if let Some(resp) = self
            .forward_read_optional(
                meta,
                forward.then_some(req),
                "get_job",
                |mut c, r| async move { c.get_job(r).await },
            )
            .await
        {
            return Ok(resp);
        }

        let job = self
            .cluster
            .get_job_for_display(job_id)
            .ok_or_else(|| Status::not_found(format!("job {} not found", job_id)))?;

        // A non-owner, non-admin caller does not get another tenant's work_dir, command, stdio
        // paths, or (the targeting-sensitive one) allocated nodelist. Policy is configurable; the
        // default redacts those fields while leaving the Slurm-standard queue view intact.
        self.scoped_job_info(&job, identity.as_ref())
            .map(Response::new)
            .ok_or_else(|| Status::not_found(format!("job {} not found", job_id)))
    }

    async fn cancel_job(&self, request: Request<CancelJobRequest>) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.cancel_job(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward cancel_job to leader: {e}");
                    return Err(status);
                }
            }
        }

        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        let job_id = req.job_id;

        // Snapshot the job before cancelling so we have allocated_nodes
        let job = self.cluster.get_job(job_id);

        self.cluster
            .cancel_job(job_id, &req.user)
            .map_err(cluster_err_to_status)?;

        // Send cancel signal to agents so the process is actually killed
        if let Some(job) = job {
            let cluster = self.cluster.clone();
            tokio::spawn(async move {
                crate::scheduler_loop::send_cancel_to_agents(&cluster, &job, 0).await;
            });
        }

        Ok(Response::new(()))
    }

    async fn complete_job(
        &self,
        request: Request<CompleteJobRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.complete_job(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward complete_job to leader: {e}");
                    return Err(status);
                }
            }
        }

        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        let job = self
            .cluster
            .finish_srun_job(req.job_id, req.exit_code, &req.user)
            .map_err(|e| match e {
                crate::cluster::SrunCompleteError::NotFound(id) => {
                    Status::not_found(format!("job {id} not found"))
                }
                crate::cluster::SrunCompleteError::NotSrunJob(id) => {
                    Status::failed_precondition(format!("job {id} is not an srun allocation"))
                }
                crate::cluster::SrunCompleteError::NotStepDispatch(id) => {
                    Status::failed_precondition(format!(
                        "job {id} does not use native step dispatch"
                    ))
                }
                crate::cluster::SrunCompleteError::AlreadyTerminal { job_id, state } => {
                    Status::failed_precondition(format!("job {job_id} is already {state:?}"))
                }
                crate::cluster::SrunCompleteError::NotOwner { job_id, user } => {
                    Status::permission_denied(format!(
                        "user {user} is not permitted to complete job {job_id}"
                    ))
                }
                crate::cluster::SrunCompleteError::Internal { job_id, message } => {
                    Status::internal(format!("job {job_id}: {message}"))
                }
            })?;

        let cluster = self.cluster.clone();
        tokio::spawn(async move {
            crate::scheduler_loop::release_srun_allocation_on_agents(&cluster, &job).await;
        });

        Ok(Response::new(()))
    }

    async fn job_keepalive(
        &self,
        request: Request<JobKeepaliveRequest>,
    ) -> Result<Response<JobKeepaliveResponse>, Status> {
        // Recorded only on the leader, where the reaper reads it.
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.job_keepalive(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward job_keepalive to leader: {e}");
                    return Err(status);
                }
            }
        }

        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        // A real keepalive always carries the caller's username. Reject an empty
        // one explicitly: an empty user must not bypass ownership checks.
        if req.user.is_empty() {
            return Err(Status::permission_denied(
                "keepalive requires a user".to_string(),
            ));
        }
        let job = self
            .cluster
            .get_job(req.job_id)
            .ok_or_else(|| Status::not_found(format!("job {} not found", req.job_id)))?;
        spur_core::auth::check_job_caller(
            &req.user,
            None,
            self.caller_is_admin(__identity.as_ref()),
            &job.spec.user,
            job.spec.uid,
            __identity.as_ref(),
            "send keepalive for",
        )
        .map_err(|e| Status::permission_denied(e.to_string()))?;

        // Only interactive allocations are reaped, so only they need tracking.
        if !(job.spec.interactive || job.spec.srun_job) {
            return Ok(Response::new(JobKeepaliveResponse {}));
        }

        self.cluster.record_job_keepalive(req.job_id);
        Ok(Response::new(JobKeepaliveResponse {}))
    }

    async fn suspend_job(
        &self,
        request: Request<SuspendJobRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.suspend_job(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward suspend_job to leader: {e}");
                    return Err(status);
                }
            }
        }
        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        let job_id = req.job_id;
        // Unknown job ids are NOT_FOUND (consistent with get_job), not a
        // precondition failure. Snapshot up-front for agent dispatch.
        let job = self
            .cluster
            .get_job(job_id)
            .ok_or_else(|| Status::not_found(format!("job {job_id} not found")))?;
        self.cluster
            .suspend_job(job_id, &req.user)
            .map_err(cluster_err_to_precondition_status)?;
        let cluster = self.cluster.clone();
        tokio::spawn(async move {
            crate::scheduler_loop::send_suspend_to_agents(&cluster, &job, false).await;
        });
        Ok(Response::new(()))
    }

    async fn resume_job(&self, request: Request<ResumeJobRequest>) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.resume_job(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward resume_job to leader: {e}");
                    return Err(status);
                }
            }
        }
        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        let job_id = req.job_id;
        // Unknown job ids are NOT_FOUND (consistent with get_job), not a
        // precondition failure. Allocation is retained across resume, so this
        // up-front snapshot's allocated_nodes is still valid for agent dispatch.
        let job = self
            .cluster
            .get_job(job_id)
            .ok_or_else(|| Status::not_found(format!("job {job_id} not found")))?;
        self.cluster
            .resume_job(job_id, &req.user)
            .map_err(cluster_err_to_precondition_status)?;
        let cluster = self.cluster.clone();
        tokio::spawn(async move {
            crate::scheduler_loop::send_suspend_to_agents(&cluster, &job, true).await;
        });
        Ok(Response::new(()))
    }

    async fn update_job(&self, request: Request<UpdateJobRequest>) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.update_job(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward update_job to leader: {e}");
                    return Err(status);
                }
            }
        }

        let __identity = Self::verified_identity(&request).cloned();
        let caller_is_privileged = self.caller_is_privileged(__identity.as_ref());
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());

        // Reject a caller who does not own the target job before any mutation —
        // including the hold/release branch below. Mirrors cancel_job / exec_in_job:
        // update touches placement, account and time limit, so it must be gated the
        // same way. `is_internal` is a verified admin only — derived from the
        // verified identity, never the wire `user` string; an unauthenticated
        // non-owner is still denied, ownership checked against the claimed user.
        let job = self
            .cluster
            .get_job(req.job_id)
            .ok_or_else(|| Status::not_found(format!("job {} not found", req.job_id)))?;
        spur_core::auth::check_job_owner(
            &req.user,
            self.caller_is_admin(__identity.as_ref()),
            &job.spec.user,
            "modify",
        )
        .map_err(|e| Status::permission_denied(e.to_string()))?;

        // Handle hold/release via priority
        if let Some(hold) = req.hold {
            if hold {
                self.cluster
                    .hold_job(req.job_id)
                    .map_err(|e| Status::internal(e.to_string()))?;
            } else {
                self.cluster
                    .release_job(req.job_id)
                    .map_err(|e| Status::internal(e.to_string()))?;
            }
            return Ok(Response::new(()));
        }

        // Same ceiling as submit: a non-privileged caller cannot raise priority above the cap
        // post-submit. UpdateJob returns Empty, so there is no response field to carry a warning —
        // log it instead, so a clamp on this path is at least observable.
        if let Some(w) = Self::clamp_priority(
            &mut req.priority,
            caller_is_privileged,
            self.cluster.config().scheduler.max_user_priority,
        ) {
            warn!(job_id = req.job_id, "{w}");
        }

        let time_limit = req.time_limit.map(|d| chrono::Duration::seconds(d.seconds));

        self.cluster
            .update_job(
                req.job_id,
                time_limit,
                req.priority,
                req.partition,
                req.comment,
                req.account,
                req.qos,
            )
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(()))
    }

    async fn requeue_job(
        &self,
        request: Request<RequeueJobRequest>,
    ) -> Result<Response<RequeueJobResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let mut fwd = Request::new(request.into_inner());
                    *fwd.metadata_mut() = Self::forwarded_metadata();
                    return client.requeue_job(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward requeue_job to leader: {e}");
                    return Err(status);
                }
            }
        }

        let __identity = Self::verified_identity(&request).cloned();
        let caller_is_admin = self.caller_is_admin(__identity.as_ref());
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        let outcome = self
            .cluster
            .requeue_job_by_user(req.job_id, &req.user, caller_is_admin, req.hold)
            .map_err(cluster_err_to_precondition_status)?;

        // Kill the old processes for jobs that were Running/Suspended; the
        // requeue already freed their allocations and re-pended them.
        for job in outcome.killed {
            let cluster = self.cluster.clone();
            tokio::spawn(async move {
                crate::scheduler_loop::send_cancel_to_agents(&cluster, &job, 0).await;
            });
        }

        Ok(Response::new(RequeueJobResponse {
            requeued: outcome.requeued,
            skipped: outcome.skipped,
        }))
    }

    async fn get_nodes(
        &self,
        request: Request<GetNodesRequest>,
    ) -> Result<Response<GetNodesResponse>, Status> {
        let forward = self.read_should_forward(&request);
        let meta = request.metadata().clone();
        let req = request.into_inner();
        if let Some(resp) = self
            .forward_read_optional(
                meta,
                forward.then(|| req.clone()),
                "get_nodes",
                |mut c, r| async move { c.get_nodes(r).await },
            )
            .await
        {
            return Ok(resp);
        }

        let nodes = self.cluster.get_nodes();

        let nodelist = req.nodelist.trim();
        let allowed_names: Option<HashSet<String>> = (!nodelist.is_empty()).then(|| {
            spur_sched::node_match::expand_hostlist_or_split(nodelist)
                .into_iter()
                .collect()
        });

        // Honour the request filters; without this, `scontrol show node X` and
        // `sinfo -n X` return the whole cluster.
        let mut proto_nodes: Vec<NodeInfo> = nodes
            .iter()
            .filter(|n| node_matches_filter(n, allowed_names.as_ref(), &req.partition))
            .map(node_to_proto)
            .filter(|n| req.states.is_empty() || req.states.contains(&n.state))
            .collect();

        let reservations = self.cluster.get_reservations();
        annotate_nodes_with_reservations(&mut proto_nodes, &reservations, Utc::now());
        annotate_nodes_with_planned_reservations(&mut proto_nodes, &self.cluster);
        Ok(Response::new(GetNodesResponse { nodes: proto_nodes }))
    }

    async fn get_node(
        &self,
        request: Request<GetNodeRequest>,
    ) -> Result<Response<NodeInfo>, Status> {
        let forward = self.read_should_forward(&request);
        let meta = request.metadata().clone();
        let req = request.into_inner();
        if let Some(resp) = self
            .forward_read_optional(
                meta,
                forward.then(|| req.clone()),
                "get_node",
                |mut c, r| async move { c.get_node(r).await },
            )
            .await
        {
            return Ok(resp);
        }

        let name = req.name;
        let node = self
            .cluster
            .get_node(&name)
            .ok_or_else(|| Status::not_found(format!("node {} not found", name)))?;
        let mut proto_node = node_to_proto(&node);
        let reservations = self.cluster.get_reservations();
        annotate_nodes_with_reservations(
            std::slice::from_mut(&mut proto_node),
            &reservations,
            Utc::now(),
        );
        annotate_nodes_with_planned_reservations(
            std::slice::from_mut(&mut proto_node),
            &self.cluster,
        );
        Ok(Response::new(proto_node))
    }

    async fn update_node(
        &self,
        request: Request<UpdateNodeRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.update_node(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward update_node to leader: {e}");
                    return Err(status);
                }
            }
        }

        self.require_admin(&request, "update node")?;

        let reason_uid = Self::verified_identity(&request).map(|id| id.uid);
        let req = request.into_inner();
        if let Some(state) = req.state {
            let node_state = spur_core::node::NodeState::from_proto_i32(state)
                .ok_or_else(|| Status::invalid_argument("invalid node state"))?;
            self.cluster
                .update_node_state(&req.name, node_state, req.reason, reason_uid)
                .map_err(|e| Status::internal(e.to_string()))?;
        }
        if !req.labels.is_empty() || !req.remove_labels.is_empty() {
            self.cluster
                .update_node_labels(&req.name, req.labels, &req.remove_labels)
                .map_err(|e| Status::internal(e.to_string()))?;
        }
        Ok(Response::new(()))
    }

    async fn drain_node(
        &self,
        request: Request<spur_proto::proto::DrainNodeRequest>,
    ) -> Result<Response<spur_proto::proto::DrainNodeResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.drain_node(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward drain_node to leader: {e}");
                    return Err(status);
                }
            }
        }
        let reason_uid = Self::verified_identity(&request).map(|id| id.uid);
        let req = request.into_inner();
        let reason = if req.reason.is_empty() {
            None
        } else {
            Some(req.reason)
        };
        if self.cluster.get_node(&req.name).is_none() {
            return Err(Status::not_found(format!("node {} not found", req.name)));
        }
        let (actual_state, running_jobs) = self
            .cluster
            .drain_node(&req.name, reason, reason_uid)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(spur_proto::proto::DrainNodeResponse {
            actual_state: actual_state.to_string(),
            running_jobs,
        }))
    }

    async fn deregister_node(
        &self,
        request: Request<spur_proto::proto::DeregisterNodeRequest>,
    ) -> Result<Response<spur_proto::proto::DeregisterNodeResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.deregister_node(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward deregister_node to leader: {e}");
                    return Err(status);
                }
            }
        }
        let req = request.into_inner();
        let reason = if req.reason.is_empty() {
            None
        } else {
            Some(req.reason)
        };
        if self.cluster.get_node(&req.name).is_none() {
            return Err(Status::not_found(format!("node {} not found", req.name)));
        }
        let evicted = self
            .cluster
            .remove_node(&req.name, req.force, reason)
            .map_err(|e| Status::failed_precondition(e.to_string()))?;
        self.spawn_cancel_for_evicted(&evicted);
        self.cluster.complete_evicted_steps(&evicted);
        Ok(Response::new(spur_proto::proto::DeregisterNodeResponse {
            evicted_jobs_count: evicted.len() as u32,
        }))
    }

    async fn deregister_agent(
        &self,
        request: Request<spur_proto::proto::DeregisterAgentRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.deregister_agent(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward deregister_agent to leader: {e}");
                    return Err(status);
                }
            }
        }
        let req = request.into_inner();

        if matches!(
            self.cluster.config().admission.mode,
            spur_core::config::AdmissionMode::Token
        ) {
            if req.node_token.is_empty() {
                return Err(Status::unauthenticated("node token required"));
            }
            let identity =
                spur_core::admission::verify_node_token(&req.node_token, self.jwt_key.as_bytes())
                    .map_err(|e| Status::unauthenticated(e.to_string()))?;
            if identity.hostname != req.hostname {
                return Err(Status::permission_denied("node token hostname mismatch"));
            }
        }

        if self.cluster.get_node(&req.hostname).is_none() {
            return Ok(Response::new(()));
        }
        // A stopping agent is Down, not gone. Deleting the record here would
        // drop the node from the WireGuard membership, every other node would
        // prune its peer, and the node could not register again over the mesh
        // after a reboot. See `ClusterManager::mark_node_down`.
        let evicted = self
            .cluster
            .mark_node_down(
                &req.hostname,
                Some(req.reason.clone()).filter(|r| !r.is_empty()),
            )
            .map_err(|e| Status::internal(e.to_string()))?;
        self.spawn_cancel_for_evicted(&evicted);
        self.cluster.complete_evicted_steps(&evicted);
        Ok(Response::new(()))
    }

    async fn get_partitions(
        &self,
        request: Request<GetPartitionsRequest>,
    ) -> Result<Response<GetPartitionsResponse>, Status> {
        let forward = self.read_should_forward(&request);
        let meta = request.metadata().clone();
        if let Some(resp) = self
            .forward_read_optional(
                meta,
                forward.then(|| request.into_inner()),
                "get_partitions",
                |mut c, r| async move { c.get_partitions(r).await },
            )
            .await
        {
            return Ok(resp);
        }

        let partitions = self.cluster.get_partitions();
        let proto: Vec<PartitionInfo> = partitions.iter().map(partition_to_proto).collect();
        Ok(Response::new(GetPartitionsResponse { partitions: proto }))
    }

    async fn ping(&self, _request: Request<()>) -> Result<Response<PingResponse>, Status> {
        let hostname: String = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".into());

        let federation_peers: Vec<String> = self
            .cluster
            .config()
            .federation
            .clusters
            .iter()
            .map(|p| format!("{}@{}", p.name, p.address))
            .collect();

        Ok(Response::new(PingResponse {
            hostname,
            server_time: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            version: env!("CARGO_PKG_VERSION").into(),
            federation_peers,
            cluster_name: self.cluster.config().cluster_name.clone(),
        }))
    }

    async fn get_job_metrics(&self, request: Request<()>) -> Result<Response<JobMetrics>, Status> {
        let forward = self.read_should_forward(&request);
        let meta = request.metadata().clone();
        if let Some(resp) = self
            .forward_read_optional(
                meta,
                forward.then_some(()),
                "get_job_metrics",
                |mut c, r| async move { c.get_job_metrics(r).await },
            )
            .await
        {
            return Ok(resp);
        }

        let snap = self.cluster.job_metrics();
        Ok(Response::new(crate::metrics_proto::job_metrics_to_proto(
            &snap,
        )))
    }

    async fn get_node_metrics(
        &self,
        request: Request<()>,
    ) -> Result<Response<NodeMetrics>, Status> {
        let forward = self.read_should_forward(&request);
        let meta = request.metadata().clone();
        if let Some(resp) = self
            .forward_read_optional(
                meta,
                forward.then_some(()),
                "get_node_metrics",
                |mut c, r| async move { c.get_node_metrics(r).await },
            )
            .await
        {
            return Ok(resp);
        }

        let snap = self.cluster.node_metrics();
        Ok(Response::new(crate::metrics_proto::node_metrics_to_proto(
            &snap,
        )))
    }

    async fn get_rpc_stats(&self, request: Request<()>) -> Result<Response<RpcStats>, Status> {
        if self.check_leader(&request).is_err() {
            {
                let proxy = &self.leader_proxy;
                let mut client = proxy.get_leader_client().await?;
                let fwd = Self::forward_request(request);
                return client.get_rpc_stats(fwd).await;
            }
        }

        Ok(Response::new(crate::metrics_proto::rpc_stats_to_proto(
            &self.rpc_stats.snapshot(),
        )))
    }

    async fn get_sched_stats(&self, request: Request<()>) -> Result<Response<SchedStats>, Status> {
        if self.check_leader(&request).is_err() {
            {
                let proxy = &self.leader_proxy;
                let mut client = proxy.get_leader_client().await?;
                let fwd = Self::forward_request(request);
                return client.get_sched_stats(fwd).await;
            }
        }

        Ok(Response::new(crate::metrics_proto::sched_stats_to_proto(
            &self.sched_stats.snapshot(),
        )))
    }

    async fn get_assoc_mgr_info(
        &self,
        request: Request<GetAssocMgrInfoRequest>,
    ) -> Result<Response<GetAssocMgrInfoResponse>, Status> {
        // Usage is read from the leader's job table, the same one admission reads.
        if self.check_leader(&request).is_err() {
            let proxy = &self.leader_proxy;
            let mut client = proxy.get_leader_client().await?;
            let fwd = Self::forward_request(request);
            return client.get_assoc_mgr_info(fwd).await;
        }

        let identity = Self::verified_identity(&request).cloned();
        let caller_is_admin = self.caller_is_admin(identity.as_ref());
        let req = request.into_inner();
        let filter = assoc_mgr_scope_user(identity.as_ref(), &req.user, caller_is_admin);
        let info = self.cluster.assoc_mgr_info(filter.as_deref());

        Ok(Response::new(GetAssocMgrInfoResponse {
            qos_records: info
                .qos_records
                .iter()
                .map(|u| assoc_mgr_to_proto(u, AssocMgrScope::Qos))
                .collect(),
            assoc_records: info
                .assoc_records
                .iter()
                .map(|u| assoc_mgr_to_proto(u, AssocMgrScope::Association))
                .collect(),
            limits_readable: info.limits_readable,
        }))
    }

    async fn reset_diag_stats(&self, request: Request<()>) -> Result<Response<()>, Status> {
        if self.check_leader(&request).is_err() {
            {
                let proxy = &self.leader_proxy;
                let mut client = proxy.get_leader_client().await?;
                let fwd = Self::forward_request(request);
                return client.reset_diag_stats(fwd).await;
            }
        }

        self.rpc_stats.reset();
        self.sched_stats.reset();
        Ok(Response::new(()))
    }

    async fn register_agent(
        &self,
        request: Request<RegisterAgentRequest>,
    ) -> Result<Response<RegisterAgentResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.register_agent(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward register_agent to leader: {e}");
                    return Err(status);
                }
            }
        }

        // Extract the remote IP from the gRPC connection as fallback
        let remote_addr = request
            .remote_addr()
            .map(|a| {
                let ip = a.ip();
                match ip {
                    std::net::IpAddr::V6(v6) => {
                        if let Some(v4) = v6.to_ipv4_mapped() {
                            v4.to_string()
                        } else {
                            ip.to_string()
                        }
                    }
                    _ => ip.to_string(),
                }
            })
            .unwrap_or_default();

        // Decided here, enforced atomically inside `register_node` against the label diff — a
        // pre-check here instead would race a concurrent registration for the same node.
        let caller_privileged = self.caller_is_privileged(Self::verified_identity(&request));

        let req = request.into_inner();
        let resources = req.resources.map(proto_to_resource_set).unwrap_or_default();

        let reject_loopback = self.cluster.config().network.reject_loopback_comm_addr;
        let advertised = req.address.clone();
        let agent_addr = tokio::task::spawn_blocking(move || {
            resolve_registration_comm_addr(&advertised, &remote_addr, reject_loopback)
        })
        .await
        .map_err(|e| Status::internal(format!("comm address resolution task failed: {e}")))??;

        let agent_port = if req.port > 0 { req.port as u16 } else { 6818 };

        let node_token_response = self.validate_admission(&req.join_token, &req.hostname)?;

        let source = spur_core::node::node_source_from_registration(&req.version, &req.labels);
        self.cluster
            .register_node(
                // NodeName and NodeHostname are the same until agents can supply both.
                req.hostname.clone(),
                req.hostname.clone(),
                resources,
                agent_addr,
                agent_port,
                req.wg_pubkey,
                req.version,
                source,
                req.labels,
                caller_privileged,
            )
            .map_err(register_node_rpc_status)?;

        Ok(Response::new(RegisterAgentResponse {
            accepted: true,
            message: "registered".into(),
            node_token: node_token_response,
        }))
    }

    async fn report_job_status(
        &self,
        request: Request<ReportJobStatusRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.report_job_status(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward report_job_status to leader: {e}");
                    return Err(status);
                }
            }
        }

        let req = request.into_inner();
        let state = spur_core::job::JobState::from_proto_i32(req.state)
            .ok_or_else(|| Status::invalid_argument("invalid job state"))?;

        // Non-empty `reporting_node` means a per-node completion report. The final
        // job outcome is still derived from aggregated exit codes in
        // `Job::derived_completion`.
        let completion_result = if !req.reporting_node.is_empty() {
            validate_completion_report_state_for_rpc(state, req.exit_code)?;
            Some(self.cluster.node_complete(
                req.job_id,
                &req.reporting_node,
                req.exit_code,
                req.signal,
                req.run_attempt,
            ))
        } else {
            None
        };

        if req.drain_node && !req.reporting_node.is_empty() {
            warn!(
                node = %req.reporting_node,
                reason = %req.drain_reason,
                job_id = req.job_id,
                "agent requested node drain"
            );
            if let Err(e) = self.cluster.update_node_state(
                &req.reporting_node,
                spur_core::node::NodeState::Drain,
                Some(req.drain_reason),
                // Unauthenticated RPC with client-supplied node/reason; render
                // Unknown rather than attributing to root (uid 0).
                None,
            ) {
                warn!(
                    node = %req.reporting_node,
                    error = %e,
                    "failed to drain node on agent request"
                );
            }
        }

        use crate::cluster::NodeCompleteResult;

        match completion_result {
            Some(Ok(NodeCompleteResult::AllDone { .. })) => Ok(Response::new(())),
            Some(Ok(NodeCompleteResult::Completing)) => {
                if let Some(job) = self.cluster.get_job(req.job_id) {
                    if job
                        .spec
                        .script
                        .as_deref()
                        .is_some_and(batch_script_uses_step_launch)
                    {
                        let missing: Vec<String> = job
                            .allocated_nodes
                            .iter()
                            .filter(|node| !job.node_completions.contains_key(*node))
                            .cloned()
                            .collect();
                        if !missing.is_empty() {
                            let cluster = self.cluster.clone();
                            let job_id = req.job_id;
                            tokio::spawn(async move {
                                crate::scheduler_loop::cancel_job_on_nodes(
                                    &cluster, job_id, &missing, 15,
                                )
                                .await;
                            });
                        }
                    }
                }
                Ok(Response::new(()))
            }
            Some(Ok(NodeCompleteResult::AlreadyTerminal)) => {
                warn!(
                    job_id = req.job_id,
                    node = %req.reporting_node,
                    "duplicate completion report for terminal job"
                );
                Ok(Response::new(()))
            }
            Some(Ok(NodeCompleteResult::StaleReport)) => {
                warn!(
                    job_id = req.job_id,
                    node = %req.reporting_node,
                    run_attempt = req.run_attempt,
                    "ignoring completion report from superseded run"
                );
                Ok(Response::new(()))
            }
            Some(Err(e)) => {
                warn!(
                    job_id = req.job_id,
                    node = %req.reporting_node,
                    error = %e,
                    "node_complete failed"
                );
                Err(node_complete_to_status(e))
            }
            None => Ok(Response::new(())),
        }
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.heartbeat(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward heartbeat to leader: {e}");
                    return Err(status);
                }
            }
        }

        let req = request.into_inner();

        if matches!(
            self.cluster.config().admission.mode,
            spur_core::config::AdmissionMode::Token
        ) {
            if req.node_token.is_empty() {
                return Err(Status::unauthenticated("node token required"));
            }
            let identity =
                spur_core::admission::verify_node_token(&req.node_token, self.jwt_key.as_bytes())
                    .map_err(|e| Status::unauthenticated(e.to_string()))?;
            if identity.hostname != req.hostname {
                return Err(Status::permission_denied("node token hostname mismatch"));
            }
        }

        if self
            .cluster
            .update_heartbeat(&req.hostname, req.cpu_load, req.free_memory_mb)
        {
            // Learn a mesh key that appeared/changed after registration so the node joins ApplyMesh
            // without a restart. Only meaningful once the node is known (heartbeat accepted).
            if self
                .cluster
                .update_node_wg_pubkey(&req.hostname, &req.wg_pubkey)
            {
                info!(node = %req.hostname, "learned updated WireGuard mesh key from heartbeat");
            }
            self.reclaim_stale_agent_jobs(&req.hostname, &req.running_jobs);
            if let Some(k0s) = &req.k0s_status {
                self.record_k0s_node_status(&req.hostname, k0s);
            }
            Ok(Response::new(HeartbeatResponse {}))
        } else {
            Err(Status::not_found(format!(
                "node {} not found — is the node registered?",
                req.hostname
            )))
        }
    }

    async fn create_token(
        &self,
        request: Request<CreateTokenRequest>,
    ) -> Result<Response<CreateTokenResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    // Preserve the caller's credential (and set the forwarded marker) so the leader
                    // authorizes the ORIGINAL caller, not an anonymous forwarded request.
                    let fwd = Self::forward_request(request);
                    return client.create_token(fwd).await;
                }
                Err(_) => return Err(status),
            }
        }

        self.require_admin(&request, "create token")?;

        let req = request.into_inner();
        let ttl_secs = req.ttl_secs.filter(|&v| v > 0);

        let (token, full_string) = self
            .cluster
            .create_token(ttl_secs)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(CreateTokenResponse {
            token: full_string,
            token_id: token.id,
        }))
    }

    async fn list_tokens(
        &self,
        request: Request<ListTokensRequest>,
    ) -> Result<Response<ListTokensResponse>, Status> {
        use spur_proto::proto::TokenInfo;

        // Admission tokens are cluster secrets; only an admin may enumerate them.
        self.require_admin(&request, "list tokens")?;

        let tokens = self.cluster.list_tokens();
        let infos = tokens
            .into_iter()
            .map(|t| TokenInfo {
                id: t.id,
                created_at: t.created_at.to_rfc3339(),
                expires_at: t.expires_at.map(|e| e.to_rfc3339()).unwrap_or_default(),
                revoked: t.revoked,
            })
            .collect();

        Ok(Response::new(ListTokensResponse { tokens: infos }))
    }

    async fn revoke_token(
        &self,
        request: Request<RevokeTokenRequest>,
    ) -> Result<Response<RevokeTokenResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    // Preserve the caller's credential (and set the forwarded marker) so the leader
                    // authorizes the ORIGINAL caller, not an anonymous forwarded request.
                    let fwd = Self::forward_request(request);
                    return client.revoke_token(fwd).await;
                }
                Err(_) => return Err(status),
            }
        }

        self.require_admin(&request, "revoke token")?;

        let req = request.into_inner();
        self.cluster
            .revoke_token(&req.token_id)
            .map_err(|e| Status::not_found(e.to_string()))?;

        Ok(Response::new(RevokeTokenResponse {}))
    }

    async fn get_job_steps(
        &self,
        request: Request<GetJobStepsRequest>,
    ) -> Result<Response<GetJobStepsResponse>, Status> {
        let forward = self.read_should_forward(&request);
        let meta = request.metadata().clone();
        let identity = Self::verified_identity(&request).cloned();
        let req = request.into_inner();
        let job_id = req.job_id;
        if let Some(resp) = self
            .forward_read_optional(
                meta,
                forward.then_some(req),
                "get_job_steps",
                |mut c, r| async move { c.get_job_steps(r).await },
            )
            .await
        {
            return Ok(resp);
        }

        // Scope step visibility to the same policy as get_job: step names can leak intent, so they
        // are blanked under `redacted` and the list is empty under `owner_only`. Resolve the parent
        // job first and return NOT_FOUND if it is gone — matching get_job, and so the owner is never
        // mis-derived as `""` (which would treat the true owner as unprivileged).
        let job = self
            .cluster
            .get_job_for_display(job_id)
            .ok_or_else(|| Status::not_found(format!("job {} not found", job_id)))?;
        let privileged = viewer_is_privileged(
            identity.as_ref(),
            &job.spec.user,
            self.caller_is_admin(identity.as_ref()),
        );
        let redact_names = match job_info_disclosure(
            privileged,
            self.cluster.config().controller.job_info_visibility,
        ) {
            JobInfoDisclosure::Full => false,
            JobInfoDisclosure::Redacted => true,
            // Owner-only: the job is invisible to this caller, so are its steps.
            JobInfoDisclosure::Hidden => {
                return Ok(Response::new(GetJobStepsResponse { steps: Vec::new() }));
            }
        };

        let steps = self.cluster.get_steps(job_id);
        let step_infos: Vec<JobStepInfo> = steps
            .iter()
            .map(|s| JobStepInfo {
                job_id: s.job_id,
                step_id: s.step_id,
                name: if redact_names {
                    String::new()
                } else {
                    s.name.clone()
                },
                state: s.state.display().to_string(),
                num_tasks: s.num_tasks,
            })
            .collect();
        Ok(Response::new(GetJobStepsResponse { steps: step_infos }))
    }

    async fn create_job_step(
        &self,
        request: Request<CreateJobStepRequest>,
    ) -> Result<Response<CreateJobStepResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.create_job_step(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward create_job_step to leader: {e}");
                    return Err(status);
                }
            }
        }

        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        let job_id = req.job_id;

        let job = self
            .cluster
            .get_job(job_id)
            .ok_or_else(|| Status::not_found(format!("job {} not found", job_id)))?;

        spur_core::auth::check_job_caller(
            &req.user,
            Some(req.uid),
            self.caller_is_admin(__identity.as_ref()),
            &job.spec.user,
            job.spec.uid,
            __identity.as_ref(),
            "attach to",
        )
        .map_err(|e| Status::permission_denied(e.to_string()))?;

        if job.state != spur_core::job::JobState::Running {
            return Err(Status::failed_precondition(format!(
                "job {} is not running (state: {:?})",
                job_id, job.state
            )));
        }

        let target_node =
            select_step_node(&job.allocated_nodes, &req.node).map_err(Status::invalid_argument)?;

        // Resolve the target agent address BEFORE creating the step: an unregistered node returns a
        // retryable Unavailable and the client retries the whole call, so resolving first keeps a
        // failed attempt from leaking a step per retry.
        let node = self.cluster.get_node(target_node).ok_or_else(|| {
            Status::unavailable(format!("node {target_node} is not currently registered"))
        })?;
        let node_addr = node_comm_socket(&node, target_node)?;

        let existing_steps = self.cluster.get_steps(job_id);
        let step_id = existing_steps
            .iter()
            .filter(|s| s.step_id < 0xFFFF_FFF0)
            .count() as u32;

        let step = spur_core::step::JobStep {
            job_id,
            step_id,
            name: req.command.join(" "),
            state: spur_core::step::StepState::Running,
            num_tasks: req.num_tasks.max(1),
            cpus_per_task: req.cpus_per_task.max(1),
            resources: spur_core::resource::ResourceAllocations::default(),
            nodes: job.allocated_nodes.clone(),
            distribution: spur_core::step::TaskDistribution::Block,
            start_time: Some(chrono::Utc::now()),
            end_time: None,
            exit_code: None,
        };

        self.cluster
            .create_step(step)
            .map_err(|e| Status::internal(format!("failed to create job step: {e}")))?;

        // Resolve the effective container so the interactive-PTY client can forward it
        // to the agent: the agent connects directly (bypassing the controller) and cannot
        // see the parent job spec to inherit `salloc/sbatch --container-image`.
        let container = resolve_step_container(req.container.clone(), &job.spec);

        Ok(Response::new(CreateJobStepResponse {
            step_id,
            node_addr,
            container,
        }))
    }

    async fn complete_job_step(
        &self,
        request: Request<CompleteJobStepRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.complete_job_step(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward complete_job_step to leader: {e}");
                    return Err(status);
                }
            }
        }

        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        let job_id = req.job_id;

        let job = self
            .cluster
            .get_job(job_id)
            .ok_or_else(|| Status::not_found(format!("job {} not found", job_id)))?;

        spur_core::auth::check_job_caller(
            &req.user,
            Some(req.uid),
            self.caller_is_admin(__identity.as_ref()),
            &job.spec.user,
            job.spec.uid,
            __identity.as_ref(),
            "complete a step in",
        )
        .map_err(|e| Status::permission_denied(e.to_string()))?;

        // Owning the job is not licence to address any step id: reserved ids
        // belong to the batch, extern, and interactive steps.
        if req.step_id >= spur_core::step::STEP_RESERVED_MIN {
            return Err(Status::invalid_argument(format!(
                "step {} is reserved and cannot be completed by a client",
                req.step_id
            )));
        }
        let step = self.cluster.get_step(job_id, req.step_id).ok_or_else(|| {
            Status::not_found(format!("step {}.{} not found", job_id, req.step_id))
        })?;
        // Best-effort fast path; the WAL apply enforces this once the job ends.
        if step.state.is_terminal() {
            return Ok(Response::new(()));
        }

        self.cluster
            .record_step_complete(job_id, req.step_id, req.exit_code)
            .map_err(|e| Status::internal(format!("failed to record step completion: {e}")))?;

        Ok(Response::new(()))
    }

    async fn create_partition(
        &self,
        request: Request<CreatePartitionRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.create_partition(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward create_partition to leader: {e}");
                    return Err(status);
                }
            }
        }

        self.require_admin(&request, "create partition")?;

        let req = request.into_inner();

        if req.nodes.is_empty() && req.selector.is_empty() {
            return Err(Status::invalid_argument(
                "partition must specify at least one of nodes or selector",
            ));
        }

        let max_time_minutes = if req.max_time.is_empty()
            || req.max_time.eq_ignore_ascii_case("INFINITE")
            || req.max_time.eq_ignore_ascii_case("UNLIMITED")
        {
            None
        } else {
            Some(
                spur_core::config::parse_time_minutes(&req.max_time).ok_or_else(|| {
                    Status::invalid_argument(format!("invalid max_time: {}", req.max_time))
                })?,
            )
        };

        let default_time_minutes = if req.default_time.is_empty() {
            None
        } else {
            Some(
                spur_core::config::parse_time_minutes(&req.default_time).ok_or_else(|| {
                    Status::invalid_argument(format!("invalid default_time: {}", req.default_time))
                })?,
            )
        };

        let state = match req.state.to_uppercase().as_str() {
            "" | "UP" => spur_core::partition::PartitionState::Up,
            "DOWN" => spur_core::partition::PartitionState::Down,
            "DRAIN" => spur_core::partition::PartitionState::Drain,
            "INACTIVE" => spur_core::partition::PartitionState::Inactive,
            other => {
                return Err(Status::invalid_argument(format!(
                    "unknown partition state '{}'; expected UP, DOWN, DRAIN, or INACTIVE",
                    other
                )))
            }
        };

        let preempt_mode = match req.preempt_mode.to_uppercase().as_str() {
            "" | "OFF" => spur_core::partition::PreemptMode::Off,
            "CANCEL" => spur_core::partition::PreemptMode::Cancel,
            "REQUEUE" => spur_core::partition::PreemptMode::Requeue,
            "SUSPEND" => spur_core::partition::PreemptMode::Suspend,
            other => {
                return Err(Status::invalid_argument(format!(
                    "unknown preempt_mode '{}'; expected OFF, CANCEL, REQUEUE, or SUSPEND",
                    other
                )))
            }
        };

        let partition = spur_core::partition::Partition {
            name: req.name,
            state,
            is_default: req.is_default,
            nodes: req.nodes,
            selector: req.selector.into_iter().collect(),
            max_time_minutes,
            default_time_minutes,
            // A literal 0 means "no limit" (matches the update-partition
            // contract) rather than a partition that can never run a job.
            max_nodes: req.max_nodes.filter(|&n| n != 0),
            min_nodes: if req.min_nodes == 0 { 1 } else { req.min_nodes },
            allow_accounts: req.allow_accounts,
            allow_groups: req.allow_groups,
            deny_accounts: req.deny_accounts,
            deny_qos: req.deny_qos,
            allow_qos: req.allow_qos,
            priority_tier: req.priority_tier,
            preempt_mode,
            preempt_exempt_time: req.preempt_exempt_time,
            ..Default::default()
        };

        self.cluster
            .create_partition(partition)
            .map_err(partition_rpc_status)?;

        Ok(Response::new(()))
    }

    async fn update_partition(
        &self,
        request: Request<UpdatePartitionRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.update_partition(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward update_partition to leader: {e}");
                    return Err(status);
                }
            }
        }

        self.require_admin(&request, "update partition")?;

        let req = request.into_inner();

        let state = req
            .state
            .map(|s| match s.to_uppercase().as_str() {
                v @ ("UP" | "DOWN" | "DRAIN" | "INACTIVE") => Ok(v.to_string()),
                other => Err(Status::invalid_argument(format!(
                    "unknown partition state '{}'; expected UP, DOWN, DRAIN, or INACTIVE",
                    other
                ))),
            })
            .transpose()?;

        let preempt_mode = req
            .preempt_mode
            .map(|pm| match pm.to_uppercase().as_str() {
                v @ ("OFF" | "CANCEL" | "REQUEUE" | "SUSPEND") => Ok(v.to_string()),
                other => Err(Status::invalid_argument(format!(
                    "unknown preempt_mode '{}'; expected OFF, CANCEL, REQUEUE, or SUSPEND",
                    other
                ))),
            })
            .transpose()?;

        let max_time = req.max_time;
        let allow_accounts = if req.set_allow_accounts {
            Some(req.allow_accounts)
        } else {
            None
        };
        let allow_groups = if req.set_allow_groups {
            Some(req.allow_groups)
        } else {
            None
        };
        let deny_accounts = if req.set_deny_accounts {
            Some(req.deny_accounts)
        } else {
            None
        };
        let deny_qos = if req.set_deny_qos {
            Some(req.deny_qos)
        } else {
            None
        };
        let allow_qos = if req.set_allow_qos {
            Some(req.allow_qos)
        } else {
            None
        };
        let (max_nodes, clear_max_nodes) =
            resolve_max_nodes_update(req.max_nodes_value, req.clear_max_nodes);

        let selector = if req.set_selector || !req.selector.is_empty() {
            Some(req.selector.into_iter().collect())
        } else {
            None
        };
        // Match create_partition's "0 means unset, not literally zero" rule.
        let min_nodes = req.min_nodes.map(|mn| if mn == 0 { 1 } else { mn });

        self.cluster
            .update_partition(
                &req.name,
                req.nodes,
                selector,
                state,
                req.is_default,
                max_time,
                req.default_time,
                max_nodes,
                clear_max_nodes,
                min_nodes,
                allow_accounts,
                allow_groups,
                deny_accounts,
                deny_qos,
                allow_qos,
                req.priority_tier,
                preempt_mode,
                if req.clear_preempt_exempt_time {
                    Some(None)
                } else {
                    req.preempt_exempt_time.map(Some)
                },
            )
            .map_err(partition_rpc_status)?;

        Ok(Response::new(()))
    }

    async fn delete_partition(
        &self,
        request: Request<DeletePartitionRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.delete_partition(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward delete_partition to leader: {e}");
                    return Err(status);
                }
            }
        }

        self.require_admin(&request, "delete partition")?;

        let name = request.into_inner().name;
        self.cluster
            .delete_partition(&name)
            .map_err(partition_rpc_status)?;

        Ok(Response::new(()))
    }

    async fn reconfigure(&self, request: Request<()>) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.reconfigure(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward reconfigure to leader: {e}");
                    return Err(status);
                }
            }
        }

        self.require_admin(&request, "reconfigure")?;

        self.cluster
            .reconfigure()
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(()))
    }

    async fn create_reservation(
        &self,
        request: Request<CreateReservationRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.create_reservation(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward create_reservation to leader: {e}");
                    return Err(status);
                }
            }
        }

        let identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, identity.as_ref());
        self.require_reservation_manager(&req.user, identity.as_ref())
            .await?;

        let details = txn::create_details(
            &req.start_time,
            req.duration_minutes,
            &req.nodes,
            &req.accounts,
            &req.users,
            &req.flags,
        );
        let entity_name = req.name.clone();
        let actor = req.user.clone();

        let result = self.build_and_create_reservation(req);
        self.audit_reservation(
            TxnAction::Create,
            &entity_name,
            &actor,
            identity.as_ref(),
            details,
            &result,
        );
        result.map(|()| Response::new(()))
    }

    async fn update_reservation(
        &self,
        request: Request<UpdateReservationRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.update_reservation(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward update_reservation to leader: {e}");
                    return Err(status);
                }
            }
        }

        let identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, identity.as_ref());
        self.require_reservation_manager(&req.user, identity.as_ref())
            .await?;

        let details = txn::update_details(
            req.duration_minutes,
            &req.add_nodes,
            &req.remove_nodes,
            &req.add_users,
            &req.remove_users,
            &req.add_accounts,
            &req.remove_accounts,
        );
        let entity_name = req.name.clone();
        let actor = req.user.clone();

        let result = self
            .cluster
            .update_reservation(
                &req.name,
                req.duration_minutes,
                &req.add_nodes,
                &req.remove_nodes,
                &req.add_users,
                &req.remove_users,
                &req.add_accounts,
                &req.remove_accounts,
            )
            .map_err(reservation_rpc_status);
        self.audit_reservation(
            TxnAction::Update,
            &entity_name,
            &actor,
            identity.as_ref(),
            details,
            &result,
        );
        result.map(|()| Response::new(()))
    }

    async fn delete_reservation(
        &self,
        request: Request<DeleteReservationRequest>,
    ) -> Result<Response<()>, Status> {
        if let Err(status) = self.check_leader(&request) {
            let proxy = &self.leader_proxy;
            match proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.delete_reservation(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward delete_reservation to leader: {e}");
                    return Err(status);
                }
            }
        }

        let identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, identity.as_ref());
        self.require_reservation_manager(&req.user, identity.as_ref())
            .await?;

        let entity_name = req.name.clone();
        let actor = req.user.clone();

        let result = self
            .cluster
            .delete_reservation(&req.name)
            .map_err(reservation_rpc_status);
        self.audit_reservation(
            TxnAction::Delete,
            &entity_name,
            &actor,
            identity.as_ref(),
            txn::delete_details(None),
            &result,
        );
        result.map(|()| Response::new(()))
    }

    async fn list_reservations(
        &self,
        request: Request<ListReservationsRequest>,
    ) -> Result<Response<ListReservationsResponse>, Status> {
        let forward = self.read_should_forward(&request);
        let meta = request.metadata().clone();
        let req = request.into_inner();
        if let Some(resp) = self
            .forward_read_optional(
                meta,
                forward.then(|| req.clone()),
                "list_reservations",
                |mut c, r| async move { c.list_reservations(r).await },
            )
            .await
        {
            return Ok(resp);
        }
        let name = req.name.trim();
        let reservations = self.cluster.get_reservations();
        let now = Utc::now();
        let infos: Vec<ReservationInfo> = reservations
            .iter()
            .filter(|r| name.is_empty() || r.name == name)
            .map(|r| ReservationInfo {
                name: r.name.clone(),
                start_time: r.start_time.to_rfc3339(),
                end_time: r.end_time.to_rfc3339(),
                nodes: r.nodes.join(","),
                accounts: r.accounts.join(","),
                users: r.users.join(","),
                flags: r.flags.display_csv(),
                state: r.state_label(now).into(),
                owner: r.owner.clone(),
            })
            .collect();
        Ok(Response::new(ListReservationsResponse {
            reservations: infos,
        }))
    }

    async fn exec_in_job(
        &self,
        request: Request<ExecInJobRequest>,
    ) -> Result<Response<ExecInJobResponse>, Status> {
        if self.check_leader(&request).is_err() {
            {
                let proxy = &self.leader_proxy;
                let mut client = proxy.get_leader_client().await?;
                let fwd = Self::forward_request(request);
                return client.exec_in_job(fwd).await;
            }
        }

        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        let job_id = req.job_id;

        let job = self
            .cluster
            .get_job(job_id)
            .ok_or_else(|| Status::not_found(format!("job {} not found", job_id)))?;

        spur_core::auth::check_job_caller(
            &req.user,
            None,
            self.caller_is_admin(__identity.as_ref()),
            &job.spec.user,
            job.spec.uid,
            __identity.as_ref(),
            "exec into",
        )
        .map_err(|e| Status::permission_denied(e.to_string()))?;

        if job.state != spur_core::job::JobState::Running {
            return Err(Status::failed_precondition(format!(
                "job {} is not running (state: {})",
                job_id, job.state
            )));
        }

        let node_name = job
            .allocated_nodes
            .first()
            .ok_or_else(|| Status::internal(format!("job {} has no allocated nodes", job_id)))?
            .clone();

        let node = self
            .cluster
            .get_node(&node_name)
            .ok_or_else(|| Status::not_found(format!("node {} not found", node_name)))?;
        let agent_addr = node_comm_http_url(&node, &node_name)?;

        let mut agent = crate::agent_client::connect(agent_addr.clone())
            .await
            .map_err(|e| {
                Status::unavailable(format!("cannot reach agent at {}: {}", agent_addr, e))
            })?
            .max_decoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(spur_proto::MAX_GRPC_REQUEST_SIZE);

        let resp = agent
            .exec_in_job(ExecInJobRequest {
                job_id,
                command: req.command,
                user: req.user,
            })
            .await
            .map_err(|e| Status::internal(format!("exec failed: {}", e)))?;

        Ok(resp)
    }

    /// Route a step from srun to the job's allocated nodes. Unlike ExecInJob,
    /// the job may not have a tracked process — salloc allocations only exist
    /// as scheduler bookkeeping.
    async fn run_step(
        &self,
        request: Request<RunStepRequest>,
    ) -> Result<Response<RunStepResponse>, Status> {
        if self.check_leader(&request).is_err() {
            let proxy = &self.leader_proxy;
            let mut client = proxy.get_leader_client().await?;
            let fwd = Self::forward_request(request);
            return client.run_step(fwd).await;
        }

        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.user, __identity.as_ref());
        let job_id = req.job_id;

        let job = self
            .cluster
            .get_job(job_id)
            .ok_or_else(|| Status::not_found(format!("job {} not found", job_id)))?;

        // A step executes arbitrary commands on the job's allocated nodes, so the
        // caller must own the target job — same gate as create_job_step / exec_in_job.
        spur_core::auth::check_job_caller(
            &req.user,
            Some(req.uid),
            self.caller_is_admin(__identity.as_ref()),
            &job.spec.user,
            job.spec.uid,
            __identity.as_ref(),
            "run a step in",
        )
        .map_err(|e| Status::permission_denied(e.to_string()))?;

        if job.allocated_nodes.is_empty() {
            return Err(Status::failed_precondition(format!(
                "job {} has no allocated nodes — is the allocation still active?",
                job_id
            )));
        }

        let step = self
            .cluster
            .get_steps(job_id)
            .into_iter()
            .find(|s| s.step_id == req.step_id)
            .ok_or_else(|| {
                Status::not_found(format!("step {} not found for job {}", req.step_id, job_id))
            })?;

        let num_nodes = job.allocated_nodes.len() as u32;
        let plan = build_step_task_plan(step.num_tasks, num_nodes, step.distribution);
        if plan.is_empty() {
            return Err(Status::failed_precondition(format!(
                "step {} for job {} has no tasks to run",
                req.step_id, job_id
            )));
        }

        let step_num_tasks = step.num_tasks;
        let command = req.command.clone();
        let work_dir = req.work_dir.clone();
        let environment = req.environment.clone();
        let uid = req.uid;
        let gid = req.gid;
        let step_id = req.step_id;
        let label = req.label;
        let job_mpi = job.spec.mpi.as_deref().unwrap_or(spur_core::mpi::MPI_NONE);
        let mpi = spur_core::mpi::resolve_step_mpi(req.mpi.as_str(), job_mpi).to_string();
        let step_container = resolve_step_container(req.container.clone(), &job.spec);
        let pmix_tmpdir = self.cluster.config().mpi.pmix_tmpdir.clone();
        let modex_connect_timeout_secs = self.cluster.config().mpi.modex_connect_timeout_secs;
        let modex_fence_timeout_secs = self.cluster.config().mpi.modex_fence_timeout_secs;
        let modex_verify_timeout_secs = self.cluster.config().mpi.modex_verify_timeout_secs;

        struct NodeDispatch {
            node_name: String,
            agent_addr: String,
            node_tasks: spur_core::task_launch::NodeStepTasks,
        }

        let mut dispatches = Vec::new();
        let mut dispatch_errors = Vec::new();

        for node_tasks in plan {
            let node_name = match job.allocated_nodes.get(node_tasks.node_index as usize) {
                Some(name) => name.clone(),
                None => {
                    dispatch_errors.push(format!(
                        "step plan references node index {} but job {} has {} nodes",
                        node_tasks.node_index,
                        job_id,
                        job.allocated_nodes.len()
                    ));
                    continue;
                }
            };
            let Some(node) = self.cluster.get_node(&node_name) else {
                dispatch_errors.push(format!("node {node_name} not found"));
                continue;
            };
            let agent_addr = match node_comm_http_url(&node, &node_name) {
                Ok(url) => url,
                Err(e) => {
                    dispatch_errors.push(format!("node {node_name}: {e}"));
                    continue;
                }
            };
            dispatches.push(NodeDispatch {
                node_name,
                agent_addr,
                node_tasks,
            });
        }

        if !dispatch_errors.is_empty() {
            return Err(Status::failed_precondition(format!(
                "srun step dispatch setup failed: {}",
                dispatch_errors.join("; ")
            )));
        }

        let pmix_peers = if mpi == MPI_PMIX {
            Some(
                spur_core::mpi::PmixStepPeers::from_participants(
                    dispatches
                        .iter()
                        .map(|dispatch| dispatch.node_tasks.node_index)
                        .collect(),
                    |idx| {
                        let name = job.allocated_nodes.get(idx as usize)?;
                        self.cluster
                            .get_node(name)
                            .and_then(|n| n.comm_addr().map(str::to_string))
                    },
                )
                .map_err(Status::failed_precondition)?,
            )
        } else {
            None
        };
        let needs_pmix_prepare = pmix_peers.as_ref().is_some_and(|peers| {
            step_needs_pmix_prepare(
                peers.num_nodes,
                job.spec.mpi.as_deref(),
                job.allocated_nodes.len() as u32,
                job.spec.script.as_deref(),
            )
        });
        let run_attempt = job.run_attempt;

        let dispatch_pmix_plans: Vec<Option<spur_proto::proto::PmixLaunchPlan>> =
            if let Some(peers) = pmix_peers.as_ref() {
                dispatches
                    .iter()
                    .map(|node_dispatch| {
                        let local = spur_core::mpi::pmix_local_dispatch_for_step(
                            peers,
                            node_dispatch.node_tasks.node_index,
                            job_id,
                            step_num_tasks,
                            node_dispatch.node_tasks.task_offset,
                            node_dispatch.node_tasks.tasks_on_node,
                            pmix_tmpdir.clone(),
                            uid,
                            gid,
                            modex_connect_timeout_secs,
                            modex_fence_timeout_secs,
                            modex_verify_timeout_secs,
                        )
                        .map_err(Status::failed_precondition)?;
                        spur_core::mpi::build_validated_pmix_plan_proto(mpi.as_str(), local)
                            .map_err(Status::failed_precondition)?
                            .ok_or_else(|| {
                                Status::failed_precondition("job is not configured for PMIx")
                            })
                            .map(Some)
                    })
                    .collect::<Result<Vec<_>, Status>>()?
            } else {
                vec![None; dispatches.len()]
            };

        let mut pmix_prepare_guard = None;

        if needs_pmix_prepare {
            if let Some(detail) = pmix_dispatch::multi_node_pmix_unsupported(
                dispatches.iter().filter_map(|dispatch| {
                    self.cluster
                        .get_node(&dispatch.node_name)
                        .map(|node| node.source.clone())
                }),
            ) {
                return Err(Status::failed_precondition(detail));
            }

            let prepare_nodes: Vec<PmixPrepareNode> = dispatches
                .iter()
                .zip(&dispatch_pmix_plans)
                .map(|(node_dispatch, pmix_plan)| {
                    Ok(PmixPrepareNode {
                        node_name: node_dispatch.node_name.clone(),
                        agent_addr: node_dispatch.agent_addr.clone(),
                        pmix_plan: pmix_plan.clone().ok_or_else(|| {
                            Status::failed_precondition("job is not configured for PMIx")
                        })?,
                    })
                })
                .collect::<Result<Vec<_>, Status>>()?;
            if prepare_nodes.len() != dispatches.len() {
                return Err(Status::failed_precondition(
                    "multi-node PMIx step missing launch plan for one or more nodes",
                ));
            }
            if let Err(detail) = pmix_dispatch::prepare_pmix_on_nodes(
                job_id,
                run_attempt,
                prepare_nodes,
                crate::scheduler_loop::dispatch_deadline(&self.cluster),
            )
            .await
            {
                return Err(Status::failed_precondition(format!(
                    "PMIx prepare failed: {detail}"
                )));
            }
            let addrs: Vec<String> = dispatches.iter().map(|d| d.agent_addr.clone()).collect();
            pmix_prepare_guard = Some(pmix_dispatch::PmixPreparedReleaseGuard::new(job_id, addrs));
        }

        let mut set = tokio::task::JoinSet::new();
        for (dispatch, pmix_plan) in dispatches.iter().zip(dispatch_pmix_plans) {
            let node_name = dispatch.node_name.clone();
            let agent_addr = dispatch.agent_addr.clone();
            let node_tasks = dispatch.node_tasks.clone();
            let command = command.clone();
            let work_dir = work_dir.clone();
            let environment = environment.clone();
            let step_mpi = mpi.clone();
            let container = step_container.clone();
            set.spawn(async move {
                let mut agent = crate::agent_client::connect(agent_addr.clone())
                    .await
                    .map_err(|e| {
                        Status::unavailable(format!("cannot reach agent at {}: {}", agent_addr, e))
                    })?
                    .max_decoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE)
                    .max_encoding_message_size(spur_proto::MAX_GRPC_REQUEST_SIZE);

                let agent_resp = agent
                    .run_command(RunCommandRequest {
                        command: command.clone(),
                        uid,
                        gid,
                        work_dir: work_dir.clone(),
                        environment: environment.clone(),
                        job_id,
                        num_tasks: node_tasks.tasks_on_node,
                        task_offset: node_tasks.task_offset,
                        step_id,
                        step_num_tasks,
                        label,
                        pmix_plan,
                        mpi: step_mpi.clone(),
                        pmix_prepared: needs_pmix_prepare,
                        container,
                    })
                    .await
                    .map_err(|e| {
                        Status::internal(format!("run_command on {} failed: {}", node_name, e))
                    })?
                    .into_inner();

                Ok::<_, Status>((node_name, agent_resp))
            });
        }

        let mut max_exit = 0i32;
        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut ran_nodes = Vec::new();
        let step_node_names: Vec<String> = dispatches.iter().map(|d| d.node_name.clone()).collect();
        let mut step_abort_sent = false;

        while let Some(result) = set.join_next().await {
            match result {
                Ok(Ok((node_name, agent_resp))) => {
                    max_exit = max_exit.max(agent_resp.exit_code);
                    stdout.push_str(&agent_resp.stdout);
                    stderr.push_str(&agent_resp.stderr);
                    ran_nodes.push(node_name);
                }
                Ok(Err(e)) => {
                    warn!(job_id, step_id, error = %e, "step dispatch failed on one node");
                    dispatch_errors.push(e.to_string());
                    max_exit = max_exit.max(1);
                    if !step_abort_sent {
                        step_abort_sent = true;
                        crate::scheduler_loop::cancel_step_on_nodes(
                            &self.cluster,
                            job_id,
                            step_id,
                            &step_node_names,
                            15,
                        )
                        .await;
                    }
                }
                Err(e) => {
                    warn!(job_id, step_id, error = %e, "step dispatch task panicked");
                    dispatch_errors.push(format!("step dispatch task panicked: {e}"));
                    max_exit = max_exit.max(1);
                    if !step_abort_sent {
                        step_abort_sent = true;
                        crate::scheduler_loop::cancel_step_on_nodes(
                            &self.cluster,
                            job_id,
                            step_id,
                            &step_node_names,
                            15,
                        )
                        .await;
                    }
                }
            }
        }

        if !dispatch_errors.is_empty() {
            max_exit = max_exit.max(1);
            stderr.push_str(&format!(
                "srun step dispatch errors:\n{}\n",
                dispatch_errors.join("\n")
            ));
        }

        if needs_pmix_prepare {
            if let Some(guard) = pmix_prepare_guard.as_mut() {
                guard.disarm();
            }
            let addrs: Vec<String> = dispatches.iter().map(|d| d.agent_addr.clone()).collect();
            pmix_dispatch::release_pmix_on_agents(&addrs, job_id).await;
        }

        if let Err(e) = self.cluster.record_step_complete(job_id, step_id, max_exit) {
            warn!(
                job_id,
                step_id,
                error = %e,
                "failed to record step completion"
            );
        }

        Ok(Response::new(RunStepResponse {
            exit_code: max_exit,
            stdout,
            stderr,
            node: ran_nodes.join(","),
        }))
    }

    // -- Native cluster (k0s) lifecycle. Leader-gated; record intent in the
    //    replicated k0s state and let the reconcile loop (cluster_k8s.rs) converge in 5b/5c. --
    async fn cluster_up(
        &self,
        request: Request<ClusterUpRequest>,
    ) -> Result<Response<ClusterUpResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            match self.leader_proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.cluster_up(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward cluster_up to leader: {e}");
                    return Err(status);
                }
            }
        }
        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.caller, __identity.as_ref());
        if !is_k0s_admin(self.cluster.association_cache(), &req.caller) {
            return Err(Status::permission_denied(
                "k0s cluster up requires cluster admin",
            ));
        }
        let state = self.cluster.k0s_state();
        let nodes = self.cluster.get_nodes();
        let assigned = nodes.iter().any(|n| n.k0s_role.is_some());

        // Teardown clears the recorded scope/CP but roles drain on later reconcile ticks; block a
        // re-up in that window so it can't reuse the emptied scope and silently enroll every node.
        if state.phase == spur_core::k0s::K0sPhase::Down && assigned {
            return Err(Status::failed_precondition(
                "cluster teardown is in progress; re-up is safe once no nodes show a k0s role \
                 in `spur k8s status`",
            ));
        }

        // Resolve the node scope fail-closed; a bare re-up of an assigned cluster keeps the recorded
        // scope, a fresh up with no selection = whole inventory. CP candidates are the in-scope members.
        let scope_requested =
            !req.nodes.is_empty() || !req.partition.is_empty() || !req.selector.is_empty();
        let member_nodes = if assigned && !scope_requested {
            state.member_nodes.clone()
        } else {
            crate::cluster_k8s::resolve_member_nodes(
                &nodes,
                &req.nodes,
                &req.partition,
                &req.selector,
            )
            .map_err(Status::invalid_argument)?
        };
        let candidates: Vec<String> = if member_nodes.is_empty() {
            nodes.iter().map(|n| n.name.clone()).collect()
        } else {
            member_nodes.clone()
        };

        // Resolve the HA control-plane set fail-closed BEFORE recording intent: an explicit node
        // list wins, else `--replicas` (or the config default) picks the lowest-named nodes.
        let explicit_override =
            !req.control_plane_nodes.is_empty() || req.control_plane_replicas.is_some();
        let replicas = req
            .control_plane_replicas
            .filter(|r| *r > 0)
            .unwrap_or(self.control_plane_replicas);
        let pinned = req
            .control_plane_node
            .clone()
            .or_else(|| state.control_plane_node.clone());
        // A bare re-up of an assigned cluster targets the recorded set, so it stays idempotent
        // regardless of the config default; an explicit list/replica count is resolved and enforced.
        let cp_set = if assigned && !explicit_override {
            state.controllers()
        } else {
            crate::cluster_k8s::resolve_control_plane_set(
                candidates,
                &req.control_plane_nodes,
                pinned.as_deref(),
                replicas,
            )
            .map_err(Status::invalid_argument)?
        };

        // A control-plane change after roles are assigned would leave an inconsistent topology
        // (provisioning skips assigned nodes); require `spur k8s down --reset` to re-elect.
        if assigned {
            let mut current = state.controllers();
            let mut want = cp_set.clone();
            current.sort();
            want.sort();
            if current != want {
                return Err(Status::failed_precondition(format!(
                    "control plane is already assigned ({}); tear the cluster down \
                     (spur k8s down --reset) before changing it to [{}]",
                    state.controllers().join(", "),
                    cp_set.join(", "),
                )));
            }
            if scope_requested && member_nodes != state.member_nodes {
                return Err(Status::failed_precondition(
                    "cluster membership is already assigned; tear the cluster down \
                     (spur k8s down --reset) before changing the node scope",
                ));
            }
            // Neither the control plane nor the scope changed: a bare or identically-scoped re-up
            // is a true no-op, so skip writing a redundant WAL entry.
            return Ok(Response::new(ClusterUpResponse {
                accepted: true,
                message: "k0s cluster already up with this control plane and scope".to_string(),
                nodes: crate::cluster_k8s::node_statuses(&self.cluster),
            }));
        }

        let bootstrap = cp_set.first().cloned();
        self.cluster
            .set_k0s_phase(
                spur_core::k0s::K0sPhase::Provisioning,
                bootstrap,
                cp_set,
                member_nodes,
                false,
            )
            .map_err(|e| Status::internal(format!("set k0s phase: {e}")))?;
        Ok(Response::new(ClusterUpResponse {
            accepted: true,
            message: "k0s cluster provisioning requested".to_string(),
            nodes: crate::cluster_k8s::node_statuses(&self.cluster),
        }))
    }

    async fn cluster_add_nodes(
        &self,
        request: Request<ClusterAddNodesRequest>,
    ) -> Result<Response<ClusterAddNodesResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            match self.leader_proxy.get_leader_client().await {
                Ok(mut client) => {
                    let mut fwd = Request::new(request.into_inner());
                    *fwd.metadata_mut() = Self::forwarded_metadata();
                    return client.cluster_add_nodes(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward cluster_add_nodes to leader: {e}");
                    return Err(status);
                }
            }
        }
        let req = request.into_inner();
        if !is_k0s_admin(self.cluster.association_cache(), &req.caller) {
            return Err(Status::permission_denied(
                "k0s cluster add-nodes requires cluster admin",
            ));
        }

        let state = self.cluster.k0s_state();
        // Online add only makes sense on a running cluster.
        if !matches!(
            state.phase,
            spur_core::k0s::K0sPhase::Ready | spur_core::k0s::K0sPhase::Provisioning
        ) {
            return Err(Status::failed_precondition(
                "cluster is not up; use `spur k8s up` to start it",
            ));
        }
        // A whole-inventory cluster (empty member_nodes) already enrolls every registered node, so a
        // node added later is picked up automatically — narrowing to an explicit set here would drop
        // the others. Direct the operator to the mechanism that already works.
        if state.member_nodes.is_empty() {
            return Err(Status::failed_precondition(
                "cluster enrolls all nodes; a newly-registered node joins automatically \
                 (no add-nodes needed on a whole-inventory cluster)",
            ));
        }

        // Resolve the requested nodes with the same union semantics as `spur k8s up` scope flags.
        let all_nodes = self.cluster.get_nodes();
        let requested = crate::cluster_k8s::resolve_member_nodes(
            &all_nodes,
            &req.nodes,
            &req.partition,
            &req.selector,
        )
        .map_err(Status::invalid_argument)?;
        if requested.is_empty() {
            return Err(Status::invalid_argument(
                "no nodes selected; pass --nodes, --partition, or --selector",
            ));
        }
        // Adding a control-plane node would change the etcd quorum topology — out of scope here.
        let controller_set = state.controllers();
        let controllers: std::collections::HashSet<&String> = controller_set.iter().collect();
        for n in &requested {
            if controllers.contains(n) {
                return Err(Status::invalid_argument(format!(
                    "node {n} is a control plane; online control-plane changes are not supported \
                     (tear down with `spur k8s down --reset` to re-elect)"
                )));
            }
        }

        self.cluster
            .add_k0s_member_nodes(requested.clone())
            .map_err(|e| Status::internal(format!("add k0s member nodes: {e}")))?;
        Ok(Response::new(ClusterAddNodesResponse {
            accepted: true,
            message: format!("added {} node(s) to the cluster", requested.len()),
            nodes: crate::cluster_k8s::node_statuses(&self.cluster),
        }))
    }

    async fn cluster_remove_nodes(
        &self,
        request: Request<ClusterRemoveNodesRequest>,
    ) -> Result<Response<ClusterRemoveNodesResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            match self.leader_proxy.get_leader_client().await {
                Ok(mut client) => {
                    let mut fwd = Request::new(request.into_inner());
                    *fwd.metadata_mut() = Self::forwarded_metadata();
                    return client.cluster_remove_nodes(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward cluster_remove_nodes to leader: {e}");
                    return Err(status);
                }
            }
        }
        let req = request.into_inner();
        if !is_k0s_admin(self.cluster.association_cache(), &req.caller) {
            return Err(Status::permission_denied(
                "k0s cluster remove-nodes requires cluster admin",
            ));
        }

        let state = self.cluster.k0s_state();
        if !matches!(
            state.phase,
            spur_core::k0s::K0sPhase::Ready | spur_core::k0s::K0sPhase::Provisioning
        ) {
            return Err(Status::failed_precondition(
                "cluster is not up; nothing to remove",
            ));
        }
        // A whole-inventory cluster (empty member_nodes) enrolls every registered node, so a removed
        // node would just be re-enrolled by the next reconcile. Removal only makes sense for a scoped
        // cluster, where the node can actually leave the member set.
        if state.member_nodes.is_empty() {
            return Err(Status::failed_precondition(
                "cluster enrolls all nodes; remove-nodes needs a scoped cluster \
                 (use `spur node remove` to decommission a host from SPUR entirely)",
            ));
        }

        // Expand the hostlist; every name must be a registered node.
        let requested = spur_core::hostlist::expand(&req.nodes)
            .map_err(|e| Status::invalid_argument(format!("invalid --nodes hostlist: {e}")))?;
        if requested.is_empty() {
            return Err(Status::invalid_argument("no nodes selected; pass --nodes"));
        }
        // Guard against emptying the member set: an empty member_nodes means "whole inventory", so
        // removing every member would flip the scope and re-enroll the very nodes just removed.
        // Checked before the per-node loop so it fires ahead of the control-plane reject.
        let requested_set: std::collections::HashSet<&String> = requested.iter().collect();
        if state.member_nodes.iter().all(|m| requested_set.contains(m)) {
            return Err(Status::failed_precondition(
                "removing these nodes would empty the cluster; tear it down with \
                 `spur k8s down` instead of removing every member",
            ));
        }
        let registered: std::collections::HashSet<String> = self
            .cluster
            .get_nodes()
            .into_iter()
            .map(|n| n.name)
            .collect();
        let controller_set = state.controllers();
        let controllers: std::collections::HashSet<&String> = controller_set.iter().collect();
        let force = req.force.unwrap_or(false);
        for n in &requested {
            if !registered.contains(n) {
                return Err(Status::invalid_argument(format!(
                    "node {n} is not a registered node"
                )));
            }
            // Only nodes actually enrolled in this (scoped) cluster can be removed — otherwise an
            // out-of-scope registered node could be drained + `k0s reset` destructively for nothing.
            if !state.is_member(n) {
                return Err(Status::invalid_argument(format!(
                    "node {n} is not a member of this cluster"
                )));
            }
            // A control-plane node can't be removed here — that changes etcd quorum (out of scope).
            if controllers.contains(n) {
                return Err(Status::invalid_argument(format!(
                    "node {n} is a control plane; online control-plane changes are not supported \
                     (tear down with `spur k8s down --reset` to change the control plane)"
                )));
            }
            // Refuse a busy node unless forced (Slurm `scontrol delete` semantics).
            if !force && self.cluster.node_has_running_jobs(n) {
                return Err(Status::failed_precondition(format!(
                    "node {n} has running jobs; drain them or pass --force to skip this check"
                )));
            }
        }

        // Drive removal per node. Drop each node from the member set BEFORE draining it, re-adding it
        // only if removal fails: while a node is a member with no role, `provision_assignments` would
        // re-enroll and restart the very component being torn down (RECONCILE_INTERVAL 30s vs a 120s
        // drain makes that the norm, not a race — and a leader flip after the loop would strand it).
        let timeout = req.drain_timeout_secs.unwrap_or(0);
        let mut removed: Vec<String> = Vec::new();
        let mut failures: Vec<String> = Vec::new();
        for n in &requested {
            if let Err(e) = self.cluster.remove_k0s_member_nodes(vec![n.clone()]) {
                failures.push(format!("{n}: could not update membership: {e}"));
                continue;
            }
            match crate::cluster_k8s::remove_worker(&self.cluster, n, timeout, force).await {
                Ok(()) => removed.push(n.clone()),
                Err(e) => {
                    // Removal failed — put it back in scope so the reconcile loop keeps managing it
                    // rather than leaving a stranded, unmanaged node.
                    if let Err(re) = self.cluster.add_k0s_member_nodes(vec![n.clone()]) {
                        warn!(node = %n, error = %re, "failed to restore member after remove error");
                    }
                    failures.push(format!("{n}: {e}"));
                }
            }
        }

        let message = if failures.is_empty() {
            format!("removed {} node(s) from the cluster", removed.len())
        } else {
            format!(
                "removed {} node(s); {} failed: {}",
                removed.len(),
                failures.len(),
                failures.join("; ")
            )
        };
        Ok(Response::new(ClusterRemoveNodesResponse {
            accepted: failures.is_empty(),
            message,
            nodes: crate::cluster_k8s::live_node_statuses(&self.cluster).await,
        }))
    }

    async fn cluster_down(
        &self,
        request: Request<ClusterDownRequest>,
    ) -> Result<Response<ClusterDownResponse>, Status> {
        if let Err(status) = self.check_leader(&request) {
            match self.leader_proxy.get_leader_client().await {
                Ok(mut client) => {
                    let fwd = Self::forward_request(request);
                    return client.cluster_down(fwd).await;
                }
                Err(e) => {
                    warn!("failed to forward cluster_down to leader: {e}");
                    return Err(status);
                }
            }
        }
        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.caller, __identity.as_ref());
        if !is_k0s_admin(self.cluster.association_cache(), &req.caller) {
            return Err(Status::permission_denied(
                "k0s cluster down requires cluster admin",
            ));
        }
        self.cluster
            .set_k0s_phase(
                spur_core::k0s::K0sPhase::Down,
                None,
                Vec::new(),
                Vec::new(),
                req.reset,
            )
            .map_err(|e| Status::internal(format!("set k0s phase: {e}")))?;
        Ok(Response::new(ClusterDownResponse {
            accepted: true,
            message: "k0s cluster teardown requested".to_string(),
        }))
    }

    async fn cluster_status(
        &self,
        request: Request<ClusterStatusRequest>,
    ) -> Result<Response<ClusterStatusResponse>, Status> {
        if self.check_leader(&request).is_err() {
            let mut client = self.leader_proxy.get_leader_client().await?;
            let fwd = Self::forward_request(request);
            return client.cluster_status(fwd).await;
        }
        let state = self.cluster.k0s_state();
        let control_plane_nodes = state.controllers();
        Ok(Response::new(ClusterStatusResponse {
            phase: crate::cluster_k8s::phase_str(state.phase),
            control_plane_node: state.control_plane_node.unwrap_or_default(),
            control_plane_nodes,
            member_nodes: state.member_nodes,
            nodes: crate::cluster_k8s::live_node_statuses(&self.cluster).await,
        }))
    }

    async fn cluster_kubeconfig(
        &self,
        request: Request<ClusterKubeconfigRequest>,
    ) -> Result<Response<ClusterKubeconfigResponse>, Status> {
        if self.check_leader(&request).is_err() {
            let mut client = self.leader_proxy.get_leader_client().await?;
            let fwd = Self::forward_request(request);
            return client.cluster_kubeconfig(fwd).await;
        }
        let __identity = Self::verified_identity(&request).cloned();
        let mut req = request.into_inner();
        Self::authoritative_user(&mut req.caller, __identity.as_ref());
        let is_admin = is_k0s_admin(self.cluster.association_cache(), &req.caller);

        if req.admin {
            // Admin check first, opt-in second — deliberately in that order. Both must pass, so the
            // ordering costs nothing, but it keeps the "feature is disabled" detail from leaking to
            // callers who were not entitled to the credential in the first place; they simply learn
            // they are not admin.
            if !is_admin {
                return Err(Status::permission_denied(
                    "the cluster-admin kubeconfig requires cluster admin",
                ));
            }
            // Serving the cluster-admin credential is gated on an explicit opt-in, not just on the
            // admin check above: `caller` is client-supplied and unauthenticated, so without this
            // any peer that can reach the controller could ask for a cluster-admin kubeconfig by
            // claiming to be root. Off by default; get it on the control-plane node instead.
            if !self.cluster.config().cluster.allow_admin_kubeconfig {
                return Err(Status::permission_denied(
                    "serving the cluster-admin kubeconfig over RPC is disabled \
                     ([cluster] allow_admin_kubeconfig = false). Run `k0s kubeconfig admin` on the \
                     control-plane node, or enable the option if the control-plane port is already \
                     restricted to administrators.",
                ));
            }
            // Cluster-admin kubeconfig (`k0s kubeconfig admin` on the control-plane agent).
            return match crate::cluster_k8s::fetch_admin_kubeconfig(&self.cluster).await {
                Ok(kubeconfig) => Ok(Response::new(ClusterKubeconfigResponse { kubeconfig })),
                Err(e) => Err(Status::unavailable(format!(
                    "could not fetch admin kubeconfig from the control-plane agent: {e}"
                ))),
            };
        }

        // A non-admin caller may only mint their own scope; an admin may target any user. An empty
        // target means "the caller's own namespace" (an empty caller is internal/root).
        let target = if req.user.is_empty() {
            req.caller.clone()
        } else {
            req.user.clone()
        };
        if !is_admin && target != req.caller {
            return Err(Status::permission_denied(format!(
                "user '{}' may only request their own kubeconfig",
                req.caller
            )));
        }
        if target.is_empty() {
            return Err(Status::invalid_argument(
                "no target user for the scoped kubeconfig; pass --user or set a caller identity",
            ));
        }

        // Scoped kubeconfig: resolve the target's account -> its namespace + per-user ServiceAccount,
        // then have the control-plane agent mint a bound token there.
        let (namespace, sa) = resolve_user_namespace_sa(self.cluster.association_cache(), &target)?;
        match crate::cluster_k8s::fetch_user_kubeconfig(&self.cluster, &target, &namespace, &sa)
            .await
        {
            Ok(kubeconfig) => Ok(Response::new(ClusterKubeconfigResponse { kubeconfig })),
            Err(e) => Err(Status::unavailable(format!(
                "could not mint a scoped kubeconfig for user '{target}': {e}"
            ))),
        }
    }
}

pub async fn serve(
    addr: SocketAddr,
    cluster: Arc<ClusterManager>,
    raft_handle: Arc<RaftHandle>,
    rpc_stats: Arc<RpcStatsCollector>,
    sched_stats: Arc<SchedStatsCollector>,
    accounting_service: Option<crate::accounting::AccountingService>,
    control_plane_replicas: u32,
) -> anyhow::Result<()> {
    let client_addrs: BTreeMap<u64, String> = raft_handle
        .peers
        .iter()
        .map(|(id, raft_addr)| {
            let client_addr = if let Some(host) = raft_addr.rsplit_once(':').map(|(h, _)| h) {
                format!("{}:6817", host)
            } else {
                format!("{}:6817", raft_addr)
            };
            (*id, client_addr)
        })
        .collect();

    let leader_proxy = LeaderProxy::new(raft_handle.clone(), client_addrs.clone());

    let jwt_key = resolve_startup_jwt_key(&cluster.config());
    let auth_mode = cluster.config().auth.mode;
    // Unlike node admission, an unset key here must reject every credential, never fall back
    // to a forgeable constant; `required` mode refuses to start key-less (see config validation).
    let auth_verification_key = cluster.config().auth.jwt_key.clone().unwrap_or_default();

    let service = ControllerService {
        cluster,
        client_addrs,
        raft: raft_handle.clone(),
        leader_proxy,
        rpc_stats: rpc_stats.clone(),
        sched_stats: sched_stats.clone(),
        control_plane_replicas,
        jwt_key,
    };

    let stats_layer = RpcStatsLayer::new(rpc_stats, raft_handle);
    // Applied as a layer, not a per-service interceptor, so it also covers the accounting service —
    // which carries no authorization of its own yet exposes `add_user(admin_level)`.
    let auth_layer = crate::auth_middleware::AuthLayer::new(auth_mode, &auth_verification_key);

    let mut builder = tonic::transport::Server::builder()
        .layer(stats_layer)
        .layer(auth_layer);

    let mut router = builder.add_service(spur_proto::controller_server(service));
    if let Some(service) = accounting_service {
        router = router.add_service(crate::accounting::accounting_server(service));
    }

    router.serve(addr).await?;

    Ok(())
}

/// Resolve the target node for a job step. Empty = first allocated (legacy
/// default); a named node must be one the job holds, else it targets outside it.
/// Build a step `ContainerSpec` from a parent job's container config, so a
/// nested `srun` (or one that omits `--container-image`) inherits the job's
/// container. Returns `None` when the job has no container image.
fn container_spec_from_job_spec(spec: &spur_core::job::JobSpec) -> Option<ContainerSpec> {
    let image = spec.container_image.clone().filter(|s| !s.is_empty())?;
    Some(ContainerSpec {
        image,
        mounts: spec.container_mounts.clone(),
        workdir: spec.container_workdir.clone().unwrap_or_default(),
        name: spec.container_name.clone().unwrap_or_default(),
        readonly: spec.container_readonly,
        mount_home: spec.container_mount_home,
        env: spec.container_env.clone(),
        entrypoint: spec.container_entrypoint.clone().unwrap_or_default(),
        remap_root: spec.container_remap_root,
    })
}

/// Effective container for a step: the step's own `ContainerSpec` when it set one
/// (`srun --container-image`), otherwise inherit the parent job's container config
/// (`sbatch --container-image`). When the step overrides, its env is merged over the
/// job's container env (step-wins) so setting one step variable doesn't silently drop
/// the job's container env. Shared by the buffered (`run_step`) and interactive-PTY
/// (`create_job_step`) dispatch paths.
fn resolve_step_container(
    step: Option<ContainerSpec>,
    job_spec: &spur_core::job::JobSpec,
) -> Option<ContainerSpec> {
    match step.filter(|c| !c.image.is_empty()) {
        Some(mut step) => {
            if let Some(parent) = container_spec_from_job_spec(job_spec) {
                let mut env = parent.env;
                env.extend(step.env);
                step.env = env;
            }
            Some(step)
        }
        None => container_spec_from_job_spec(job_spec),
    }
}

fn select_step_node<'a>(allocated: &'a [String], requested: &str) -> Result<&'a str, String> {
    if requested.is_empty() {
        return allocated
            .first()
            .map(String::as_str)
            .ok_or_else(|| "job has no allocated nodes".to_string());
    }
    allocated
        .iter()
        .find(|n| n.as_str() == requested)
        .map(String::as_str)
        .ok_or_else(|| format!("node {requested} is not allocated to this job"))
}

// ---- Proto conversion helpers ----

// tonic::Status is 176 bytes (over clippy's 128-byte threshold); fixed upstream in tonic 0.13+
#[allow(clippy::result_large_err)]
fn proto_to_job_spec(spec: JobSpec) -> Result<spur_core::job::JobSpec, Status> {
    let mut gres = spec.gres;
    for lic in &spec.licenses {
        gres.push(format!("license:{}", lic));
    }

    let gpus = spur_core::gpu_request::GpuRequest::from_proto(spec.gpus);
    let gpus_per_node = spur_core::gpu_request::GpuRequest::from_proto(spec.gpus_per_node);
    let gpus_per_task = spur_core::gpu_request::GpuRequest::from_proto(spec.gpus_per_task);

    let job_spec = spur_core::job::JobSpec {
        name: spec.name,
        partition: if spec.partition.is_empty() {
            None
        } else {
            Some(spec.partition)
        },
        account: if spec.account.is_empty() {
            None
        } else {
            Some(spec.account)
        },
        user: spec.user,
        uid: spec.uid,
        gid: spec.gid,
        num_nodes: spec.num_nodes.max(1),
        // 0 means the caller did not set ntasks: default to one task per node
        // (Slurm's default) so it scales with num_nodes rather than collapsing.
        num_tasks: if spec.num_tasks > 0 {
            spec.num_tasks
        } else {
            spec.num_nodes.max(1)
        },
        tasks_per_node: if spec.tasks_per_node > 0 {
            Some(spec.tasks_per_node)
        } else {
            None
        },
        cpus_per_task: spec.cpus_per_task.max(1),
        memory_per_node_mb: if spec.memory_per_node_mb > 0 {
            Some(spec.memory_per_node_mb)
        } else {
            None
        },
        memory_per_cpu_mb: if spec.memory_per_cpu_mb > 0 {
            Some(spec.memory_per_cpu_mb)
        } else {
            None
        },
        gres,
        gpus,
        gpus_per_node,
        gpus_per_task,
        script: if spec.script.is_empty() {
            None
        } else {
            Some(spec.script)
        },
        argv: spec.argv,
        script_args: spec.script_args,
        work_dir: if spec.work_dir.is_empty() {
            "/tmp".into()
        } else {
            spec.work_dir
        },
        stdout_path: if spec.stdout_path.is_empty() {
            None
        } else {
            Some(spec.stdout_path)
        },
        stderr_path: if spec.stderr_path.is_empty() {
            None
        } else {
            Some(spec.stderr_path)
        },
        stdin_path: if spec.stdin_path.is_empty() {
            None
        } else {
            Some(spec.stdin_path)
        },
        environment: spec.environment,
        time_limit: spec
            .time_limit
            .map(|d| chrono::Duration::seconds(d.seconds)),
        time_min: spec.time_min.map(|d| chrono::Duration::seconds(d.seconds)),
        qos: if spec.qos.is_empty() {
            None
        } else {
            Some(spec.qos)
        },
        priority: if spec.priority > 0 {
            Some(spec.priority)
        } else {
            None
        },
        reservation: if spec.reservation.is_empty() {
            None
        } else {
            Some(spec.reservation)
        },
        dependency: spec.dependency,
        nodelist: if spec.nodelist.is_empty() {
            None
        } else {
            Some(spec.nodelist)
        },
        exclude: if spec.exclude.is_empty() {
            None
        } else {
            Some(spec.exclude)
        },
        constraint: if spec.constraint.is_empty() {
            None
        } else {
            Some(spec.constraint.clone())
        },
        mpi: if spec.mpi.is_empty() {
            None
        } else {
            Some(spec.mpi)
        },
        distribution: if spec.distribution.is_empty() {
            None
        } else {
            Some(spec.distribution)
        },
        het_group: if spec.het_group > 0 {
            Some(spec.het_group)
        } else {
            None
        },
        array_spec: if spec.array_spec.is_empty() {
            None
        } else {
            Some(spec.array_spec)
        },
        array_job_id: None,
        array_task_id: None,
        array_max_concurrent: None,
        requeue: spec.requeue,
        exclusive: spec.exclusive,
        hold: spec.hold,
        interactive: spec.interactive,
        srun_job: spec.srun_job,
        mail_type: spec.mail_type,
        mail_user: if spec.mail_user.is_empty() {
            None
        } else {
            Some(spec.mail_user)
        },
        comment: if spec.comment.is_empty() {
            None
        } else {
            Some(spec.comment)
        },
        wckey: if spec.wckey.is_empty() {
            None
        } else {
            Some(spec.wckey)
        },
        container_image: if spec.container_image.is_empty() {
            None
        } else {
            Some(spec.container_image)
        },
        container_mounts: spec.container_mounts,
        container_workdir: if spec.container_workdir.is_empty() {
            None
        } else {
            Some(spec.container_workdir)
        },
        container_name: if spec.container_name.is_empty() {
            None
        } else {
            Some(spec.container_name)
        },
        container_readonly: spec.container_readonly,
        container_mount_home: spec.container_mount_home,
        container_env: spec.container_env,
        container_entrypoint: if spec.container_entrypoint.is_empty() {
            None
        } else {
            Some(spec.container_entrypoint)
        },
        container_remap_root: spec.container_remap_root,
        burst_buffer: if spec.burst_buffer.is_empty() {
            None
        } else {
            Some(spec.burst_buffer)
        },
        begin_time: spec.begin_time.map(|ts| {
            chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                .unwrap_or_else(chrono::Utc::now)
        }),
        deadline: spec.deadline.map(|ts| {
            chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                .unwrap_or_else(chrono::Utc::now)
        }),
        spread_job: spec.spread_job,
        topology: if spec.topology.is_empty() {
            None
        } else {
            Some(spec.topology)
        },
        host_network: spec.host_network,
        privileged: spec.privileged,
        host_ipc: spec.host_ipc,
        shm_size: if spec.shm_size.is_empty() {
            None
        } else {
            Some(spec.shm_size)
        },
        extra_resources: spec.extra_resources,
        open_mode: if spec.open_mode.is_empty() {
            None
        } else {
            Some(spec.open_mode)
        },
        pty: spec.pty,
        submit_line: sanitize_submit_line(&spec.submit_line),
    };

    // GPU demand is validated in `Cluster::submit_job` after node-count
    // normalization, so `-N4 -n1 --gpus=2` (valid once reduced to one node)
    // is not rejected against the pre-normalization node count.
    Ok(job_spec)
}

fn proto_to_resource_set(r: spur_proto::proto::ResourceSet) -> spur_core::resource::ResourceSet {
    spur_core::resource::ResourceSet {
        cpus: r.cpus,
        memory_mb: r.memory_mb,
        gpus: r
            .gpus
            .into_iter()
            .map(|g| spur_core::resource::GpuResource {
                device_id: g.device_id,
                gpu_type: g.gpu_type,
                memory_mb: g.memory_mb,
                peer_gpus: g.peer_gpus,
                link_type: match g.link_type {
                    1 => spur_core::resource::GpuLinkType::XGMI,
                    2 => spur_core::resource::GpuLinkType::NVLink,
                    _ => spur_core::resource::GpuLinkType::PCIe,
                },
            })
            .collect(),
        generic: r.generic,
    }
}

/// What a caller may see of a job, once ownership/admin status and the visibility policy are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobInfoDisclosure {
    /// Full record.
    Full,
    /// Full record minus the targeting-sensitive fields (see [`redact_sensitive_job_info`]).
    Redacted,
    /// Not disclosed at all — the caller sees `NOT_FOUND`.
    Hidden,
}

/// Whether a caller may see a job's full record unconditionally: the owner, an admin, or a caller
/// with no verified identity at all. The last case preserves the pre-auth behaviour — scoping only
/// bites once callers are actually identified, so no-auth/permissive deployments and internal
/// unauthenticated consumers (the k8s operator's nodelist read) are unaffected. Pure for testing.
fn viewer_is_privileged(
    identity: Option<&spur_core::auth::Identity>,
    owner: &str,
    caller_is_admin: bool,
) -> bool {
    match identity {
        None => true,
        Some(id) => id.user == owner || caller_is_admin,
    }
}

/// The user an assoc-mgr read is scoped to. A privileged caller — an admin, or
/// an unauthenticated one under `permissive`/`disabled`, the same treatment
/// `viewer_is_privileged` gives — reads whichever user the request names, or
/// every user when it names none. A non-admin authenticated caller is pinned to
/// their own identity, so they can neither read another tenant's usage nor
/// enumerate the cluster-wide scope inventory. Pure so the policy is testable
/// without a live service.
fn assoc_mgr_scope_user(
    identity: Option<&spur_core::auth::Identity>,
    requested: &str,
    caller_is_admin: bool,
) -> Option<String> {
    if identity.is_none() || caller_is_admin {
        return (!requested.is_empty()).then(|| requested.to_string());
    }
    identity.map(|id| id.user.clone())
}

/// Resolve the disclosure level for a job-info read. The owner and admins (`privileged`) always get
/// the full record; everyone else is governed by the configured policy. Pure so the policy matrix is
/// unit-testable without a live service.
fn job_info_disclosure(
    privileged: bool,
    visibility: spur_core::config::JobInfoVisibility,
) -> JobInfoDisclosure {
    use spur_core::config::JobInfoVisibility;
    if privileged {
        return JobInfoDisclosure::Full;
    }
    match visibility {
        JobInfoVisibility::Full => JobInfoDisclosure::Full,
        JobInfoVisibility::Redacted => JobInfoDisclosure::Redacted,
        JobInfoVisibility::OwnerOnly => JobInfoDisclosure::Hidden,
    }
}

/// Blank what a non-owner should not see, keeping identity/state/timing/account
/// so the Slurm-standard queue view still works.
fn redact_sensitive_job_info(info: &mut JobInfo) {
    info.work_dir = String::new();
    info.command = String::new();
    info.stdout_path = String::new();
    info.stderr_path = String::new();
    info.stdin_path = String::new();
    info.comment = String::new();
    info.nodelist = String::new();
    info.resources = None;
    // The submit line re-exposes most of the above, and the requested lists are
    // the same targeting oracle as the allocated one.
    info.submit_line = String::new();
    info.req_nodelist = String::new();
    info.exc_nodelist = String::new();
    info.sched_nodelist = String::new();
    // The requested shape restates the resource detail blanked above.
    info.req_tres = String::new();
    info.features = String::new();
    info.min_cpus_node = 0;
    info.min_memory_node_mb = 0;
    info.min_memory_is_per_cpu = false;
}

fn job_to_proto(job: &spur_core::job::Job) -> JobInfo {
    use spur_core::hostlist;

    JobInfo {
        job_id: job.job_id,
        name: job.spec.name.clone(),
        user: job.spec.user.clone(),
        uid: job.spec.uid,
        partition: job.spec.partition.clone().unwrap_or_default(),
        account: job.spec.account.clone().unwrap_or_default(),
        state: job.state.to_proto_i32(),
        state_reason: job.state_reason(),
        submit_time: Some(datetime_to_proto(job.submit_time)),
        start_time: job.start_time.map(datetime_to_proto),
        end_time: job.end_time.map(datetime_to_proto),
        time_limit: job.spec.time_limit.map(|d| prost_types::Duration {
            seconds: d.num_seconds(),
            nanos: 0,
        }),
        run_time: job.run_time().map(|d| prost_types::Duration {
            seconds: d.num_seconds(),
            nanos: 0,
        }),
        num_nodes: job.spec.num_nodes,
        num_tasks: job.spec.num_tasks,
        cpus_per_task: job.spec.cpus_per_task,
        nodelist: if job.allocated_nodes.is_empty() {
            String::new()
        } else {
            hostlist::compress(&job.allocated_nodes)
        },
        work_dir: job.spec.work_dir.clone(),
        command: job
            .spec
            .script
            .as_deref()
            .map(|s| {
                s.lines()
                    .find(|l| !l.starts_with('#') && !l.trim().is_empty())
                    .unwrap_or("")
                    .to_string()
            })
            .unwrap_or_default(),
        exit_code: job.exit_code.unwrap_or(0),
        exit_signal: job.exit_signal,
        derived_exit_code: job.derived_exit_code,
        stdout_path: job
            .actual_stdout_path
            .clone()
            .unwrap_or_else(|| job.resolved_stdout()),
        stderr_path: job
            .actual_stderr_path
            .clone()
            .unwrap_or_else(|| job.resolved_stderr()),
        stdin_path: job.resolved_stdin().unwrap_or_default(),
        resources: job.allocated_resources.as_ref().map(allocations_to_proto),
        priority: job.priority,
        qos: job.spec.qos.clone().unwrap_or_default(),
        array_job_id: job.spec.array_job_id.unwrap_or(0),
        array_task_id: job.spec.array_task_id.unwrap_or(0),
        reservation: job.spec.reservation.clone().unwrap_or_default(),
        comment: job.spec.comment.clone().unwrap_or_default(),
        srun_step_dispatch: job.srun_step_dispatch,
        req_gpus: spur_core::job::effective_gpus(&job.spec, job.spec.num_nodes) as u32,
        req_gpus_detail: requested_gpus_detail(&job.spec),
        preempted_by: job.preempted_by.unwrap_or(0),
        preempt_mode: job.preempt_mode.clone().unwrap_or_default(),
        preempt_qos: job.preempt_qos.clone().unwrap_or_default(),
        req_nodelist: job.spec.nodelist.clone().unwrap_or_default(),
        exc_nodelist: job.spec.exclude.clone().unwrap_or_default(),
        features: job.spec.constraint.clone().unwrap_or_default(),
        dependency: job.spec.dependency.clone(),
        submit_line: job.spec.submit_line.clone().unwrap_or_default(),
        req_tres: requested_tres(&job.spec),
        min_cpus_node: min_cpus_per_node(&job.spec).try_into().unwrap_or(u32::MAX),
        min_memory_node_mb: min_memory_requested_mb(&job.spec),
        min_memory_is_per_cpu: job.spec.memory_per_node_mb.is_none()
            && job.spec.memory_per_cpu_mb.is_some(),
        eligible_time: Some(datetime_to_proto(job.eligible_time())),
        accrue_time: Some(datetime_to_proto(job.accrue_time())),
        last_sched_eval: job.last_sched_eval.map(datetime_to_proto),
        deadline: job.spec.deadline.map(datetime_to_proto),
        time_min: job.spec.time_min.map(|d| prost_types::Duration {
            seconds: d.num_seconds(),
            nanos: 0,
        }),
        requeue: job.spec.requeue,
        restarts: job
            .requeue_count
            .saturating_add(job.preempt_requeue_count)
            .saturating_add(job.user_requeue_count),
        // srun and salloc synthesize a script too, so its presence alone does not
        // mean the batch submission Slurm's flag denotes.
        batch_flag: job.spec.script.is_some() && !job.spec.srun_job && !job.spec.interactive,
        exclusive: job.spec.exclusive,
        // Filled by annotate_jobs_with_planned_reservations: the scheduler's
        // plan lives on the cluster, not the job record.
        planned_start_time: None,
        sched_nodelist: String::new(),
    }
}

/// Client-asserted, replicated in every Raft entry, and printed to a terminal,
/// so it is bounded in bytes and stripped of terminal-controlling characters.
const MAX_SUBMIT_LINE_BYTES: usize = 1024;

/// Neutralize characters that would rewrite an operator's terminal or reorder
/// the rendered text, and bound the length. Controls become spaces rather than
/// vanishing, so `--wrap $'a\nb'` cannot silently render as one token.
fn sanitize_submit_line(raw: &str) -> Option<String> {
    let mut cleaned = String::new();
    for c in raw.chars() {
        let c = if c.is_control() || is_format_char(c) {
            ' '
        } else {
            c
        };
        if cleaned.len() + c.len_utf8() > MAX_SUBMIT_LINE_BYTES {
            cleaned.push('…');
            break;
        }
        cleaned.push(c);
    }
    let trimmed = cleaned.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Unicode `Cf`: bidi overrides and zero-width joiners render deceptively but
/// are not `char::is_control`.
fn is_format_char(c: char) -> bool {
    matches!(c, '\u{00ad}' | '\u{061c}' | '\u{180e}' | '\u{200b}'..='\u{200f}'
        | '\u{202a}'..='\u{202e}' | '\u{2060}'..='\u{2064}' | '\u{2066}'..='\u{206f}'
        | '\u{feff}' | '\u{fff9}'..='\u{fffb}' | '\u{110bd}' | '\u{110cd}'
        | '\u{13430}'..='\u{1343f}' | '\u{1bca0}'..='\u{1bca3}'
        | '\u{1d173}'..='\u{1d17a}' | '\u{e0000}'..='\u{e007f}')
}

/// Tasks landing on one node, for the per-node minima Slurm reports. Nothing
/// bounds the inputs at submit, so this family saturates rather than overflows.
fn tasks_per_node(spec: &spur_core::job::JobSpec) -> u64 {
    spec.tasks_per_node.map(u64::from).unwrap_or_else(|| {
        let nodes = u64::from(spec.num_nodes.max(1));
        u64::from(spec.num_tasks.max(1)).div_ceil(nodes)
    })
}

fn min_cpus_per_node(spec: &spur_core::job::JobSpec) -> u64 {
    tasks_per_node(spec).saturating_mul(u64::from(spec.cpus_per_task.max(1)))
}

/// The memory floor as requested. A `--mem-per-cpu` job reports the per-CPU
/// figure verbatim; Slurm labels that MinMemoryCPU rather than converting it.
fn min_memory_requested_mb(spec: &spur_core::job::JobSpec) -> u64 {
    spec.memory_per_node_mb
        .or(spec.memory_per_cpu_mb)
        .unwrap_or(0)
}

/// Total per-node memory the job implies, for the TRES roll-up.
fn min_memory_per_node_mb(spec: &spur_core::job::JobSpec) -> u64 {
    if let Some(mb) = spec.memory_per_node_mb {
        return mb;
    }
    spec.memory_per_cpu_mb
        .map(|per_cpu| per_cpu.saturating_mul(min_cpus_per_node(spec)))
        .unwrap_or(0)
}

/// Slurm-style requested TRES summary. `billing` is omitted: Spur has no TRES
/// billing weights, and emitting a made-up value would misreport usage.
fn requested_tres(spec: &spur_core::job::JobSpec) -> String {
    let nodes = u64::from(spec.num_nodes.max(1));
    // The CPU total is the job's real ask, not the per-node minimum scaled up:
    // an uneven split (-N2 -n7) would otherwise round the total upward.
    let cpu_total =
        u64::from(spec.num_tasks.max(1)).saturating_mul(u64::from(spec.cpus_per_task.max(1)));
    let mut parts = vec![format!("cpu={cpu_total}"), format!("node={nodes}")];

    let mem_mb = min_memory_per_node_mb(spec).saturating_mul(nodes);
    if mem_mb > 0 {
        parts.insert(1, format!("mem={mem_mb}M"));
    }

    let gpus = spur_core::job::effective_gpus(spec, nodes as u32);
    if gpus > 0 {
        parts.push(format!("gres/gpu={gpus}"));
    }

    parts.join(",")
}

/// Human-readable summary of a job's GPU request for display.
fn requested_gpus_detail(spec: &spur_core::job::JobSpec) -> String {
    let ty = |t: Option<&str>| t.map(|t| format!("{t}:")).unwrap_or_default();

    // Per-task form: render from the raw spec field (before resolution into PerNode).
    if let Some(ref r) = spec.gpus_per_task {
        return format!("gpu:{}{}/task", ty(r.gpu_type.as_deref()), r.count);
    }

    use spur_core::gpu_request::GpuDemand;
    let Ok(demand) = spur_core::gpu_request::resolve_gpu_demand(spec) else {
        return String::new();
    };
    match demand {
        GpuDemand::None => String::new(),
        GpuDemand::Total { count, gpu_type } => format!("gpu:{}{}", ty(gpu_type.as_deref()), count),
        GpuDemand::PerNode { counts, gpu_type } => {
            let first = counts.first().copied().unwrap_or(0);
            if counts.iter().all(|&c| c == first) {
                format!("gpu:{}{}/node", ty(gpu_type.as_deref()), first)
            } else {
                let list = counts
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                format!("gpu:{}[{}]/node", ty(gpu_type.as_deref()), list)
            }
        }
    }
}

/// Which side of the hierarchy a usage record came from. A QOS names its
/// per-user caps `MaxJobsPU`/`MaxTRESPU`, an association names its own
/// `MaxJobs`/`MaxSubmitJobs`, so the same breach reads differently.
#[derive(Clone, Copy)]
enum AssocMgrScope {
    Qos,
    Association,
}

impl AssocMgrScope {
    fn cap_name(self, cap: spur_core::accounting::Cap) -> &'static str {
        use spur_core::accounting::Cap;
        match (self, cap) {
            (Self::Qos, Cap::MaxJobs) => "MaxJobsPU",
            (Self::Qos, Cap::MaxSubmitJobs) => "MaxSubmitJobsPU",
            (Self::Qos, Cap::MaxTres) => "MaxTRESPU",
            (Self::Association, Cap::MaxJobs) => "MaxJobs",
            (Self::Association, Cap::MaxSubmitJobs) => "MaxSubmitJobs",
            (Self::Association, Cap::MaxTres) => "MaxTRES",
            (_, Cap::GrpTres) => "GrpTRES",
            (_, Cap::GrpSubmitJobs) => "GrpSubmitJobs",
        }
    }
}

/// One scope record onto the wire, users nested inside it. An unset cap carries
/// the `INFINITE` sentinel, keeping it distinguishable from a literal `0` cap as
/// `sacctmgr` already does, and TRES are `<name>=<value>` strings (e.g.
/// `cpu=36,node=9`), empty when nothing is held or capped. Caps and consumption
/// stay separate fields: composing Slurm's `Limit(Consumed)` display is the
/// client's job, and a machine consumer should not have to take it apart again.
/// Caps already exceeded are named here rather than in the client, so every
/// consumer reads the same verdict.
fn assoc_mgr_to_proto(
    usage: &spur_core::accounting::ScopeLimitUsage,
    scope: AssocMgrScope,
) -> AssocMgrRecord {
    AssocMgrRecord {
        scope: usage.scope.clone(),
        grp_running_jobs: usage.grp_running_jobs,
        grp_submitted_jobs: usage.grp_submitted_jobs,
        grp_running_tres: usage.grp_running_tres.format(),
        grp_tres: opt_tres(&usage.grp_tres),
        grp_submit_jobs: opt_cap(usage.grp_submit_jobs),
        max_wall_minutes: opt_cap(usage.max_wall_minutes),
        scope_caps: usage.user_caps.as_ref().map(|caps| AssocMgrCaps {
            max_jobs: opt_cap(caps.max_jobs),
            max_submit_jobs: opt_cap(caps.max_submit_jobs),
            max_tres: opt_tres(&caps.max_tres),
        }),
        users: usage
            .users
            .iter()
            .map(|user| AssocMgrUserRecord {
                user: user.user.clone(),
                running_jobs: user.running_jobs,
                submitted_jobs: user.submitted_jobs,
                running_tres: user.running_tres.format(),
                max_jobs: opt_cap(user.caps.max_jobs),
                max_submit_jobs: opt_cap(user.caps.max_submit_jobs),
                max_tres: opt_tres(&user.caps.max_tres),
                over_limit: cap_names(user.exceeded_caps(), scope),
            })
            .collect(),
        over_limit: cap_names(usage.exceeded_caps(), scope),
    }
}

fn opt_cap(v: Option<u32>) -> u32 {
    v.unwrap_or(spur_core::accounting::INFINITE)
}

fn opt_tres(t: &Option<spur_core::accounting::TresRecord>) -> String {
    t.as_ref().map(|t| t.format()).unwrap_or_default()
}

fn cap_names(caps: Vec<spur_core::accounting::Cap>, scope: AssocMgrScope) -> Vec<String> {
    caps.into_iter()
        .map(|c| scope.cap_name(c).to_string())
        .collect()
}

fn node_to_proto(node: &spur_core::node::Node) -> NodeInfo {
    NodeInfo {
        name: node.name.clone(),
        state: node.state.to_proto_i32(),
        state_reason: node.state_reason.clone().unwrap_or_default(),
        partitions: node.partitions.clone(),
        total_resources: Some(resource_to_proto(&node.total_resources)),
        alloc_resources: Some(allocations_to_proto(&node.alloc_resources)),
        arch: node.arch.clone(),
        os: node.os.clone(),
        cpu_load: node.cpu_load,
        free_memory_mb: node.free_memory_mb,
        boot_time: node.boot_time.map(datetime_to_proto),
        last_busy: node.last_busy.map(datetime_to_proto),
        slurmd_start_time: node.agent_start_time.map(datetime_to_proto),
        switch_name: node.switch_name.clone().unwrap_or_default(),
        active_reservation: String::new(),
        labels: node.labels.clone(),
        reservation_maint: false,
        features: node.features.clone(),
        planned_job_id: 0,
        planned_start: None,
        reason_uid: node.reason_uid,
        reason_time: node.reason_time.map(datetime_to_proto),
    }
}

fn partition_to_proto(part: &spur_core::partition::Partition) -> PartitionInfo {
    PartitionInfo {
        name: part.name.clone(),
        state: part.state.display().to_string(),
        is_default: part.is_default,
        total_nodes: 0,
        total_cpus: 0,
        nodes: part.nodes.clone(),
        max_time: part.max_time_minutes.map(|m| prost_types::Duration {
            seconds: m as i64 * 60,
            nanos: 0,
        }),
        default_time: part.default_time_minutes.map(|m| prost_types::Duration {
            seconds: m as i64 * 60,
            nanos: 0,
        }),
        max_nodes: part.max_nodes.unwrap_or(0),
        min_nodes: part.min_nodes,
        allow_root: part.allow_root,
        exclusive_user: part.exclusive_user,
        allow_accounts: part.allow_accounts.join(","),
        allow_groups: part.allow_groups.join(","),
        allow_qos: part.allow_qos.join(","),
        deny_accounts: part.deny_accounts.join(","),
        deny_qos: part.deny_qos.join(","),
        preempt_mode: format!("{:?}", part.preempt_mode),
        priority_tier: part.priority_tier,
        preempt_exempt_time: part.preempt_exempt_time,
    }
}

pub(crate) fn allocations_to_proto(
    r: &spur_core::resource::ResourceAllocations,
) -> spur_proto::proto::ResourceAllocations {
    use std::collections::HashMap;
    spur_proto::proto::ResourceAllocations {
        cpus: r.cpus,
        memory_mb: r.memory_mb,
        devices: r
            .devices
            .iter()
            .map(|(name, devs)| {
                (
                    name.clone(),
                    spur_proto::proto::DeviceAllocations {
                        devices: devs
                            .iter()
                            .map(|d| spur_proto::proto::AllocatedDevice {
                                device_id: d.device_id,
                                count: d.count,
                            })
                            .collect(),
                    },
                )
            })
            .collect::<HashMap<_, _>>(),
    }
}

#[allow(dead_code)]
pub(crate) fn proto_to_allocations(
    r: spur_proto::proto::ResourceAllocations,
) -> spur_core::resource::ResourceAllocations {
    use std::collections::HashMap;
    spur_core::resource::ResourceAllocations {
        cpus: r.cpus,
        memory_mb: r.memory_mb,
        devices: r
            .devices
            .into_iter()
            .map(|(name, devs)| {
                (
                    name,
                    devs.devices
                        .into_iter()
                        .map(|d| spur_core::resource::AllocatedDevice {
                            device_id: d.device_id,
                            count: d.count,
                        })
                        .collect(),
                )
            })
            .collect::<HashMap<_, _>>(),
    }
}

pub(crate) fn resource_to_proto(
    r: &spur_core::resource::ResourceSet,
) -> spur_proto::proto::ResourceSet {
    spur_proto::proto::ResourceSet {
        cpus: r.cpus,
        memory_mb: r.memory_mb,
        gpus: r
            .gpus
            .iter()
            .map(|g| spur_proto::proto::GpuResource {
                device_id: g.device_id,
                gpu_type: g.gpu_type.clone(),
                memory_mb: g.memory_mb,
                peer_gpus: g.peer_gpus.clone(),
                link_type: match g.link_type {
                    spur_core::resource::GpuLinkType::XGMI => {
                        spur_proto::proto::GpuLinkType::GpuLinkXgmi as i32
                    }
                    spur_core::resource::GpuLinkType::NVLink => {
                        spur_proto::proto::GpuLinkType::GpuLinkNvlink as i32
                    }
                    spur_core::resource::GpuLinkType::PCIe => {
                        spur_proto::proto::GpuLinkType::GpuLinkPcie as i32
                    }
                },
            })
            .collect(),
        generic: r.generic.clone(),
    }
}

pub(crate) fn datetime_to_proto(dt: chrono::DateTime<chrono::Utc>) -> prost_types::Timestamp {
    prost_types::Timestamp {
        seconds: dt.timestamp(),
        nanos: dt.timestamp_subsec_nanos() as i32,
    }
}

/// Whether a node passes a GetNodes request's name/partition filters. A `None`
/// name set matches every name; an empty `partition` matches every partition.
/// `allowed_names` is the caller-expanded Slurm hostlist. `partition` accepts a
/// Slurm-style comma-separated list, matching if any token names a node partition.
fn node_matches_filter(
    node: &spur_core::node::Node,
    allowed_names: Option<&HashSet<String>>,
    partition: &str,
) -> bool {
    if let Some(allowed) = allowed_names {
        if !allowed.contains(&node.name) {
            return false;
        }
    }
    let mut tokens = partition
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty());
    if partition.trim().is_empty() {
        return true;
    }
    tokens.any(|req| node.partitions.iter().any(|p| p == req))
}

fn annotate_nodes_with_reservations(
    nodes: &mut [NodeInfo],
    reservations: &[Reservation],
    now: DateTime<Utc>,
) {
    for node_info in nodes.iter_mut() {
        node_info.reservation_maint = false;
        node_info.active_reservation.clear();
        for res in reservations {
            if res.is_active(now) && res.covers_node(&node_info.name) {
                if node_info.active_reservation.is_empty() {
                    node_info.active_reservation = res.name.clone();
                }
                if res.flags.maint {
                    node_info.reservation_maint = true;
                }
            }
        }
    }
}

/// Surface the future slot the scheduler is holding for a pending job, so
/// `StartTime` answers "when" instead of staying N/A while the job waits.
fn annotate_jobs_with_planned_reservations(jobs: &mut [JobInfo], cluster: &ClusterManager) {
    if jobs.is_empty() {
        return;
    }
    cluster.with_planned_job_starts(|by_job| apply_planned_reservations(jobs, by_job));
}

fn apply_planned_reservations(
    jobs: &mut [JobInfo],
    by_job: &spur_sched::backfill::PlannedJobStarts,
) {
    let pending = spur_core::job::JobState::Pending.to_proto_i32();
    for job in jobs.iter_mut().filter(|j| j.state == pending) {
        let Some((nodes, start)) = by_job.get(&job.job_id) else {
            continue;
        };
        job.planned_start_time = Some(datetime_to_proto(*start));
        job.sched_nodelist = spur_core::hostlist::compress(nodes);
    }
}

fn annotate_nodes_with_planned_reservations(nodes: &mut [NodeInfo], cluster: &ClusterManager) {
    for node_info in nodes.iter_mut() {
        // Only meaningful while idle, matching how sinfo displays "plnd" —
        // a node that has since gone Allocated/Down must not report a
        // planned reservation left over from an earlier, staler read.
        if node_info.state != spur_core::node::NodeState::Idle.to_proto_i32() {
            continue;
        }
        if let Some((job_id, start)) = cluster.planned_reservation(&node_info.name) {
            node_info.planned_job_id = job_id;
            node_info.planned_start = Some(datetime_to_proto(start));
        }
    }
}

fn reservation_rpc_status(err: ReservationError) -> Status {
    match err {
        ReservationError::InvalidArgument(m) => Status::invalid_argument(m),
        ReservationError::NotFound(m) => Status::not_found(m),
        ReservationError::AlreadyExists(m) => Status::already_exists(m),
        ReservationError::Raft(m) => Status::internal(m),
    }
}

fn submit_rpc_status(err: crate::cluster::SubmitError) -> Status {
    match err {
        crate::cluster::SubmitError::InvalidArgument(m) => Status::invalid_argument(m),
        crate::cluster::SubmitError::Unavailable(m) => Status::unavailable(m),
        crate::cluster::SubmitError::Internal(m) => Status::internal(m),
    }
}

fn register_node_rpc_status(err: crate::cluster::RegisterNodeError) -> Status {
    match err {
        crate::cluster::RegisterNodeError::PermissionDenied(m) => Status::permission_denied(m),
        crate::cluster::RegisterNodeError::Internal(m) => Status::internal(m),
    }
}

fn partition_rpc_status(err: PartitionError) -> Status {
    match err {
        PartitionError::InvalidArgument(m) => Status::invalid_argument(m),
        PartitionError::NotFound(m) => Status::not_found(m),
        PartitionError::AlreadyExists(m) => Status::already_exists(m),
        PartitionError::Raft(m) => Status::internal(m),
    }
}

/// Resolve an `UpdatePartitionRequest`'s max-nodes intent into the
/// `(max_nodes, clear)` pair `ClusterManager::update_partition` expects.
///
/// `clear_max_nodes` and a literal `max_nodes_value == 0` both mean "no limit"
/// (0 is documented as "clear limit" in the proto); neither can express a real
/// 0-node cap. The two inputs must be collapsed into a single `clear` bool that
/// is passed through — forwarding the raw request flag would drop a `--max-nodes
/// 0` clear, since that arrives as `Some(0)` with the flag unset.
fn resolve_max_nodes_update(max_nodes_value: Option<u32>, clear_flag: bool) -> (Option<u32>, bool) {
    let clear = clear_flag || max_nodes_value == Some(0);
    let max_nodes = if clear { None } else { max_nodes_value };
    (max_nodes, clear)
}

fn cluster_err_to_status(err: anyhow::Error) -> Status {
    if err.downcast_ref::<spur_core::auth::AuthError>().is_some() {
        return Status::permission_denied(err.to_string());
    }
    Status::internal(err.to_string())
}

fn cluster_err_to_precondition_status(err: anyhow::Error) -> Status {
    if err.downcast_ref::<spur_core::auth::AuthError>().is_some() {
        return Status::permission_denied(err.to_string());
    }
    Status::failed_precondition(err.to_string())
}

fn node_complete_to_status(err: NodeCompleteError) -> Status {
    let message = err.to_string();
    let code = match err {
        NodeCompleteError::JobNotFound { .. } => Code::NotFound,
        NodeCompleteError::NodeNotAllocated { .. } => Code::InvalidArgument,
        NodeCompleteError::RaftPropose { .. } => Code::Unavailable,
    };
    Status::new(code, message)
}

#[allow(clippy::result_large_err)]
fn validate_completion_report_state_for_rpc(
    state: spur_core::job::JobState,
    exit_code: i32,
) -> Result<(), Status> {
    spur_core::job::JobState::validate_completion_report_state(state, exit_code)
        .map_err(|e| Status::invalid_argument(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use spur_core::job::{JobState, NodeCompleteError};
    use spur_core::reservation::ReservationFlags;
    use tonic::Code;

    fn job_state(cluster: &crate::cluster::ClusterManager, job_id: u32) -> Option<JobState> {
        cluster.get_job(job_id).map(|j| j.state)
    }

    #[test]
    fn resolve_registration_comm_addr_normalizes_explicit_ip() {
        assert_eq!(
            resolve_registration_comm_addr("10.0.0.2", "", false).unwrap(),
            "10.0.0.2"
        );
    }

    #[test]
    fn resolve_registration_comm_addr_falls_back_to_peer_ip() {
        assert_eq!(
            resolve_registration_comm_addr("", "10.0.0.9", false).unwrap(),
            "10.0.0.9"
        );
    }

    #[test]
    fn resolve_registration_comm_addr_prefers_peer_over_loopback_advertised() {
        assert_eq!(
            resolve_registration_comm_addr("127.0.0.1", "10.245.159.30", false).unwrap(),
            "10.245.159.30"
        );
    }

    #[test]
    fn resolve_registration_comm_addr_rejects_loopback_when_configured() {
        assert!(resolve_registration_comm_addr("127.0.0.1", "", true).is_err());
    }

    #[test]
    fn resolve_registration_comm_addr_prefers_peer_when_rejecting_loopback() {
        assert_eq!(
            resolve_registration_comm_addr("127.0.0.1", "10.245.159.30", true).unwrap(),
            "10.245.159.30"
        );
    }

    #[test]
    fn node_comm_http_url_brackets_ipv6() {
        let mut node =
            spur_core::node::Node::new("n1".into(), spur_core::resource::ResourceSet::default());
        node.address = Some("2001:db8::1".into());
        node.port = 6818;
        assert_eq!(
            node_comm_http_url(&node, "n1").unwrap(),
            "http://[2001:db8::1]:6818"
        );
    }

    #[test]
    fn resolve_user_namespace_sa_fails_closed_on_cold_cache() {
        // A not-yet-loaded cache resolves fail-open, so the scoped-kubeconfig path
        // must reject rather than mint an unscoped (cluster-wide) token.
        let cache = crate::association_cache::AssociationCache::new();
        let err = resolve_user_namespace_sa(&cache, "alice").unwrap_err();
        assert_eq!(err.code(), Code::Unavailable);
    }

    #[test]
    fn resolve_user_namespace_sa_rejects_unassociated_user() {
        let cache = crate::association_cache::AssociationCache::new();
        cache.set_loaded_without_associations();
        let err = resolve_user_namespace_sa(&cache, "alice").unwrap_err();
        assert_eq!(err.code(), Code::NotFound);
    }

    #[test]
    fn resolve_user_namespace_sa_derives_namespace_and_sa() {
        let cache = crate::association_cache::AssociationCache::new();
        cache.insert_default_account("alice", "physics");
        let (namespace, sa) = resolve_user_namespace_sa(&cache, "alice").unwrap();
        assert_eq!(namespace, "spur-acct-physics");
        assert_eq!(sa, "spur-user-alice");
    }

    #[test]
    fn forwarded_metadata_carries_the_callers_credential_to_the_leader() {
        // The leader authorizes the request, so a forwarded hop must arrive as the ORIGINAL caller.
        // Building fresh metadata (the old behaviour) dropped the credential, which would make every
        // forwarded call anonymous the moment HA + `auth.mode = required` were both on.
        let mut orig = tonic::metadata::MetadataMap::new();
        orig.insert("authorization", "Bearer tok123".parse().unwrap());

        let fwd = ControllerService::forwarded_metadata_preserving(&orig);
        assert_eq!(
            fwd.get("authorization").map(|v| v.to_str().unwrap()),
            Some("Bearer tok123"),
            "the caller's credential must survive forwarding"
        );
        assert!(
            fwd.get(FORWARDED_HEADER).is_some(),
            "the loop-breaker header is still set"
        );
    }

    #[test]
    fn forwarding_an_unauthenticated_request_adds_no_credential() {
        let fwd =
            ControllerService::forwarded_metadata_preserving(&tonic::metadata::MetadataMap::new());
        assert!(fwd.get("authorization").is_none());
        assert!(fwd.get(FORWARDED_HEADER).is_some());
    }

    #[test]
    fn is_k0s_admin_rejects_empty_caller_but_still_allows_root() {
        let cache = crate::association_cache::AssociationCache::new();
        // An omitted caller must NOT be admin: the field is client-supplied, so treating "unset" as
        // admin made the check bypassable by simply leaving it out.
        assert!(!is_k0s_admin(&cache, ""));
        // `root` is still accepted so accounting-less clusters can manage k0s at all.
        assert!(is_k0s_admin(&cache, "root"));
    }

    #[test]
    fn is_k0s_admin_named_user_denied_when_accounting_off() {
        // Cold cache = accounting disabled/not loaded: only root/internal are admin.
        let cache = crate::association_cache::AssociationCache::new();
        assert!(!is_k0s_admin(&cache, "alice"));
    }

    #[test]
    fn clamp_priority_caps_a_non_admin_above_the_ceiling() {
        let mut p = Some(u32::MAX);
        let warning = ControllerService::clamp_priority(&mut p, false, 1000);
        assert_eq!(p, Some(1000), "over-ceiling request is clamped down");
        assert!(warning.is_some(), "a clamp returns a warning");
    }

    #[test]
    fn clamp_priority_leaves_a_non_admin_at_or_below_the_ceiling() {
        for req in [None, Some(0), Some(500), Some(1000)] {
            let mut p = req;
            let warning = ControllerService::clamp_priority(&mut p, false, 1000);
            assert_eq!(p, req, "request within the ceiling is untouched");
            assert!(warning.is_none(), "no warning when nothing is clamped");
        }
    }

    #[test]
    fn clamp_priority_never_touches_a_privileged_caller() {
        // Privileged = admin or unauthenticated (auth not enforced); see caller_is_privileged.
        let mut p = Some(u32::MAX);
        let warning = ControllerService::clamp_priority(&mut p, true, 1000);
        assert_eq!(
            p,
            Some(u32::MAX),
            "a privileged caller may set any priority"
        );
        assert!(warning.is_none());
    }

    #[test]
    fn job_info_disclosure_matrix() {
        use spur_core::config::JobInfoVisibility::*;
        // The owner/admin (privileged) always sees the full record, whatever the policy.
        for v in [Redacted, OwnerOnly, Full] {
            assert_eq!(job_info_disclosure(true, v), JobInfoDisclosure::Full);
        }
        // A non-owner is governed by the policy.
        assert_eq!(
            job_info_disclosure(false, Redacted),
            JobInfoDisclosure::Redacted
        );
        assert_eq!(
            job_info_disclosure(false, OwnerOnly),
            JobInfoDisclosure::Hidden
        );
        assert_eq!(job_info_disclosure(false, Full), JobInfoDisclosure::Full);
    }

    #[test]
    fn viewer_privilege_owner_admin_and_anonymous() {
        use spur_core::auth::Identity;
        let bob = Identity {
            user: "bob".into(),
            uid: 1000,
            gid: 1000,
            is_admin: false,
        };
        let alice = Identity {
            user: "alice".into(),
            uid: 1001,
            gid: 1001,
            is_admin: false,
        };
        // Owner and admin are privileged.
        assert!(viewer_is_privileged(Some(&bob), "bob", false));
        assert!(viewer_is_privileged(Some(&alice), "bob", true));
        // A different, non-admin identified user is not.
        assert!(!viewer_is_privileged(Some(&alice), "bob", false));
        // No verified identity (auth disabled, or permissive with no credential, or an internal
        // consumer like the k8s operator) is privileged: scoping only applies to identified callers.
        assert!(viewer_is_privileged(None, "bob", false));
    }

    #[test]
    fn redact_blanks_sensitive_fields_and_keeps_the_rest() {
        let mut info = JobInfo {
            job_id: 42,
            name: "train".into(),
            user: "bob".into(),
            account: "team-b".into(),
            state_reason: "Running".into(),
            // sensitive
            work_dir: "/home/bob/run".into(),
            command: "python train.py --secret".into(),
            stdout_path: "/home/bob/out".into(),
            stderr_path: "/home/bob/err".into(),
            stdin_path: "/home/bob/in".into(),
            comment: "internal".into(),
            nodelist: "gpu-b-[01-04]".into(),
            submit_line: "sbatch --wrap 'python train.py --secret'".into(),
            req_nodelist: "gpu-b-[01-04]".into(),
            exc_nodelist: "gpu-b-05".into(),
            ..Default::default()
        };
        redact_sensitive_job_info(&mut info);
        // Sensitive fields are gone — the nodelist in particular (the co-residency targeting oracle).
        assert!(info.work_dir.is_empty());
        assert!(info.command.is_empty());
        assert!(info.stdout_path.is_empty());
        assert!(info.stderr_path.is_empty());
        assert!(info.stdin_path.is_empty());
        assert!(info.comment.is_empty());
        assert!(info.nodelist.is_empty());
        assert!(info.resources.is_none());
        assert!(info.submit_line.is_empty());
        assert!(info.req_nodelist.is_empty());
        assert!(info.exc_nodelist.is_empty());
        // Non-sensitive identity/state fields survive so the queue view still works.
        assert_eq!(info.job_id, 42);
        assert_eq!(info.name, "train");
        assert_eq!(info.user, "bob");
        assert_eq!(info.account, "team-b");
        assert_eq!(info.state_reason, "Running");
    }

    /// Identity for a viewer in the get_job/get_job_steps handler tests. No NSS resolution happens
    /// on the read path (only ownership/admin comparison), so any username works.
    fn viewer(user: &str, is_admin: bool) -> spur_core::auth::Identity {
        spur_core::auth::Identity {
            user: user.to_string(),
            uid: 1000,
            gid: 1000,
            is_admin,
        }
    }

    fn get_job_req(job_id: u32, id: Option<spur_core::auth::Identity>) -> Request<GetJobRequest> {
        let mut req = Request::new(GetJobRequest { job_id });
        if let Some(id) = id {
            req.extensions_mut().insert(id);
        }
        req
    }

    /// Wrap a request body with a verified admin identity, as the auth layer would for an admin
    /// token. Used to exercise the admin-gated control-plane mutation handlers.
    fn admin_request<T>(inner: T) -> Request<T> {
        let mut req = Request::new(inner);
        req.extensions_mut().insert(viewer("root", true));
        req
    }

    fn owned_job(user: &str, work_dir: &str) -> spur_core::job::JobSpec {
        spur_core::job::JobSpec {
            name: "j".into(),
            user: user.to_string(),
            work_dir: work_dir.to_string(),
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            ..Default::default()
        }
    }

    #[test]
    fn resolve_step_container_prefers_the_steps_own_image() {
        let mut job = owned_job("u", "/w");
        job.container_image = Some("parent.sqsh".into());
        let step = ContainerSpec {
            image: "step.sqsh".into(),
            ..Default::default()
        };
        let eff = resolve_step_container(Some(step), &job).expect("effective container");
        assert_eq!(eff.image, "step.sqsh");
    }

    #[test]
    fn resolve_step_container_inherits_parent_when_step_omits_image() {
        // A bare `srun --pty` inside a containerized salloc/sbatch: the step has
        // no image of its own and must inherit the allocation's.
        let mut job = owned_job("u", "/w");
        job.container_image = Some("parent.sqsh".into());
        assert_eq!(
            resolve_step_container(Some(ContainerSpec::default()), &job)
                .expect("inherit on empty image")
                .image,
            "parent.sqsh"
        );
        assert_eq!(
            resolve_step_container(None, &job)
                .expect("inherit on no container")
                .image,
            "parent.sqsh"
        );
    }

    #[test]
    fn resolve_step_container_merges_parent_env_under_step_env() {
        let mut job = owned_job("u", "/w");
        job.container_image = Some("parent.sqsh".into());
        job.container_env = std::collections::HashMap::from([
            ("A".to_string(), "parent".to_string()),
            ("B".to_string(), "parent".to_string()),
        ]);
        let step = ContainerSpec {
            image: "step.sqsh".into(),
            env: std::collections::HashMap::from([("B".to_string(), "step".to_string())]),
            ..Default::default()
        };
        let eff = resolve_step_container(Some(step), &job).expect("effective container");
        assert_eq!(eff.env.get("A").map(String::as_str), Some("parent"));
        assert_eq!(eff.env.get("B").map(String::as_str), Some("step"));
    }

    #[test]
    fn resolve_step_container_is_none_without_any_image() {
        let job = owned_job("u", "/w");
        assert!(resolve_step_container(None, &job).is_none());
        assert!(resolve_step_container(Some(ContainerSpec::default()), &job).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_job_scopes_by_identity_under_default_redacted_policy() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = svc
            .cluster
            .submit_job(owned_job("bob", "/home/bob"))
            .unwrap()
            .job_id;

        // Owner, admin, and unauthenticated callers all get the full record.
        for id in [
            Some(viewer("bob", false)),
            Some(viewer("carol", true)),
            None,
        ] {
            let info = svc
                .get_job(get_job_req(job_id, id))
                .await
                .unwrap()
                .into_inner();
            assert_eq!(info.work_dir, "/home/bob");
        }

        // An identified non-owner is redacted: the work_dir (and other sensitive fields) are blanked,
        // but the non-sensitive identity fields survive.
        let other = svc
            .get_job(get_job_req(job_id, Some(viewer("alice", false))))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(other.work_dir, "", "a non-owner must not see the work_dir");
        assert_eq!(other.user, "bob", "non-sensitive fields survive redaction");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_job_owner_only_hides_the_job_from_a_non_owner() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut config = step_test_config();
        config.controller.job_info_visibility = spur_core::config::JobInfoVisibility::OwnerOnly;
        let svc = test_service_with(&dir, config).await;
        let job_id = svc
            .cluster
            .submit_job(owned_job("bob", "/home/bob"))
            .unwrap()
            .job_id;

        // Owner still sees the job; a non-owner gets NOT_FOUND rather than a redacted record.
        assert!(svc
            .get_job(get_job_req(job_id, Some(viewer("bob", false))))
            .await
            .is_ok());
        let err = svc
            .get_job(get_job_req(job_id, Some(viewer("alice", false))))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_job_steps_returns_not_found_for_an_unknown_job() {
        // Regression guard: a job absent from the display cache must be NOT_FOUND, not a response
        // whose owner is mis-derived as "" (which treated the true owner as unprivileged).
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let err = svc
            .get_job_steps(Request::new(GetJobStepsRequest { job_id: 999_999 }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[test]
    fn is_k0s_admin_honors_accounting_admin_level() {
        let cache = crate::association_cache::AssociationCache::new();
        cache.insert_admin_level("carol", "Admin");
        cache.insert_admin_level("dave", "Operator");
        assert!(is_k0s_admin(&cache, "carol"));
        // Operator is not full admin; a plain member isn't either.
        assert!(!is_k0s_admin(&cache, "dave"));
        assert!(!is_k0s_admin(&cache, "erin"));
    }

    #[test]
    fn read_forwards_only_when_follower_and_not_already_forwarded() {
        // Leader serves reads locally; never forwards.
        assert!(!read_forwarding_policy(true, false));
        assert!(!read_forwarding_policy(true, true));
        // Follower forwards a fresh read to the leader.
        assert!(read_forwarding_policy(false, false));
        // An already-forwarded read is served locally to avoid forward loops.
        assert!(!read_forwarding_policy(false, true));
    }

    fn test_slurm_config() -> spur_core::config::SlurmConfig {
        serde_json::from_str(r#"{"cluster_name":"test"}"#).unwrap()
    }

    /// A `ControllerService` on a node that can never elect a leader: three
    /// unreachable peers mean no quorum, so `current_leader` stays `None`.
    async fn no_leader_service(
        cluster: Arc<crate::cluster::ClusterManager>,
        dir: &std::path::Path,
    ) -> ControllerService {
        use crate::rpc_stats::RpcStatsCollector;
        use crate::sched_stats::SchedStatsCollector;

        let handle = crate::raft::start_raft(
            1,
            &[
                "[::1]:0".to_string(),
                "[::1]:0".to_string(),
                "[::1]:0".to_string(),
            ],
            dir,
            cluster.clone(),
        )
        .await
        .unwrap();
        assert!(!handle.is_leader());
        assert_eq!(handle.current_leader(), None);

        let raft = Arc::new(handle);
        let client_addrs: BTreeMap<u64, String> = BTreeMap::new();
        ControllerService {
            cluster,
            raft: raft.clone(),
            leader_proxy: LeaderProxy::new(raft.clone(), client_addrs.clone()),
            client_addrs,
            rpc_stats: Arc::new(RpcStatsCollector::new()),
            sched_stats: Arc::new(SchedStatsCollector::new("sched/backfill")),
            control_plane_replicas: 1,
            jwt_key: String::new(),
        }
    }

    /// A non-leader that can't reach a leader must still answer reads
    /// from local applied state, not fail with "no leader elected yet".
    #[tokio::test]
    async fn get_jobs_serves_locally_when_no_leader_elected() {
        use crate::raft::StateMachineApply;
        use spur_core::job::JobSpec;
        use spur_core::wal::WalOperation;

        let dir = tempfile::TempDir::new().unwrap();
        let cluster =
            Arc::new(crate::cluster::ClusterManager::new(test_slurm_config(), dir.path()).unwrap());

        // Mirrors a follower applying a committed entry, without a leader.
        let spec = JobSpec {
            name: "job-a".into(),
            user: "alice".into(),
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            work_dir: "/tmp".into(),
            ..Default::default()
        };
        <crate::cluster::ClusterManager as StateMachineApply>::apply_operation(
            cluster.as_ref(),
            &WalOperation::JobSubmit {
                job_id: 1,
                spec: Box::new(spec),
            },
        );

        let service = no_leader_service(cluster, dir.path()).await;

        let resp = service
            .get_jobs(Request::new(GetJobsRequest::default()))
            .await;
        let jobs = resp
            .expect("get_jobs must serve local state, not fail with 'no leader elected yet'")
            .into_inner()
            .jobs;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].job_id, 1);
    }

    /// `scontrol show job` and `squeue` both read through `get_jobs`, so this — not the
    /// `get_job` redaction — is what stops one user reading another's submit line and
    /// requested nodes. Spoofing the wire `user` field, and naming the job id outright,
    /// must both come back empty.
    #[tokio::test]
    async fn get_jobs_never_returns_another_users_job_to_an_identified_caller() {
        use crate::raft::StateMachineApply;
        use spur_core::job::JobSpec;
        use spur_core::wal::WalOperation;

        let dir = tempfile::TempDir::new().unwrap();
        let cluster =
            Arc::new(crate::cluster::ClusterManager::new(test_slurm_config(), dir.path()).unwrap());
        <crate::cluster::ClusterManager as StateMachineApply>::apply_operation(
            cluster.as_ref(),
            &WalOperation::JobSubmit {
                job_id: 1,
                spec: Box::new(JobSpec {
                    name: "alice-job".into(),
                    user: "alice".into(),
                    num_nodes: 1,
                    num_tasks: 1,
                    cpus_per_task: 1,
                    work_dir: "/tmp".into(),
                    submit_line: Some("sbatch --nodelist=secret-node run.sh".into()),
                    ..Default::default()
                }),
            },
        );
        let service = no_leader_service(cluster, dir.path()).await;

        for (label, req_fields) in [
            (
                "a spoofed --user",
                GetJobsRequest {
                    user: "alice".into(),
                    ..Default::default()
                },
            ),
            (
                "an explicit job id",
                GetJobsRequest {
                    job_ids: vec![1],
                    ..Default::default()
                },
            ),
        ] {
            let mut req = Request::new(req_fields);
            req.extensions_mut().insert(viewer("bob", false));
            let jobs = service.get_jobs(req).await.unwrap().into_inner().jobs;
            assert!(
                jobs.is_empty(),
                "{label} let bob read alice's job: {jobs:?}"
            );
        }
    }

    /// The write side of the contract: with no leader, writes must fail rather
    /// than fall back to local state — guards against a refactor extending the
    /// read fallback to writes.
    #[tokio::test]
    async fn submit_job_fails_when_no_leader_elected() {
        let dir = tempfile::TempDir::new().unwrap();
        let cluster =
            Arc::new(crate::cluster::ClusterManager::new(test_slurm_config(), dir.path()).unwrap());
        let service = no_leader_service(cluster, dir.path()).await;

        let status = service
            .submit_job(Request::new(SubmitJobRequest::default()))
            .await
            .expect_err("submit_job must not serve locally when there is no leader");
        assert_eq!(status.code(), Code::Unavailable);
        assert_eq!(status.message(), "not the Raft leader");
    }

    #[test]
    fn submit_rpc_status_maps_invalid_argument() {
        use crate::cluster::SubmitError;

        let status = submit_rpc_status(SubmitError::invalid("partition 'gpu' not found"));
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(status.message(), "partition 'gpu' not found");
    }

    #[test]
    fn submit_rpc_status_maps_unavailable_as_retryable() {
        use crate::cluster::SubmitError;

        // A cold association cache is transient; it must surface as Unavailable so
        // clients (e.g. the k8s operator) retry instead of failing the job.
        let status = submit_rpc_status(SubmitError::unavailable("associations unavailable"));
        assert_eq!(status.code(), Code::Unavailable);
        assert!(spur_proto::controller_rpc_retryable(&status));
    }

    /// Only a controller-terminal job is stale; Pending (mid-dispatch), Running,
    /// and unknown ids are all spared.
    #[tokio::test]
    async fn stale_reported_jobs_selects_only_terminal_jobs() {
        use crate::raft::StateMachineApply;
        use spur_core::job::{JobSpec, JobState};
        use spur_core::wal::WalOperation;

        let dir = tempfile::TempDir::new().unwrap();
        let cluster =
            Arc::new(crate::cluster::ClusterManager::new(test_slurm_config(), dir.path()).unwrap());

        let submit = |job_id: u32, name: &str| WalOperation::JobSubmit {
            job_id,
            spec: Box::new(JobSpec {
                name: name.into(),
                user: "alice".into(),
                num_nodes: 1,
                num_tasks: 1,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            }),
        };
        let apply = |op: &WalOperation| {
            <crate::cluster::ClusterManager as StateMachineApply>::apply_operation(
                cluster.as_ref(),
                op,
            );
        };

        // 10 Pending, 11 Running, 12 terminal, and the non-Running active states
        // that must also be spared: 13 Completing, 14 Suspended, 15 Preempted.
        apply(&submit(10, "pending"));
        for id in [11, 12, 13, 14, 15] {
            apply(&submit(id, "job"));
        }

        let res = spur_core::resource::ResourceAllocations {
            cpus: 1,
            memory_mb: 0,
            devices: std::collections::HashMap::new(),
        };
        let mut per_node = std::collections::HashMap::new();
        per_node.insert("n1".to_string(), res.clone());
        for id in [11, 12, 13, 14, 15] {
            apply(&WalOperation::job_start(
                id,
                vec!["n1".into()],
                res.clone(),
                per_node.clone(),
            ));
            apply(&WalOperation::job_state_change(
                id,
                JobState::Pending,
                JobState::Running,
            ));
        }
        apply(&WalOperation::job_state_change(
            12,
            JobState::Running,
            JobState::Cancelled,
        ));
        apply(&WalOperation::job_state_change(
            13,
            JobState::Running,
            JobState::Completing,
        ));
        apply(&WalOperation::job_state_change(
            14,
            JobState::Running,
            JobState::Suspended,
        ));
        apply(&WalOperation::job_state_change(
            15,
            JobState::Running,
            JobState::Preempted,
        ));

        assert_eq!(job_state(&cluster, 10), Some(JobState::Pending));
        assert_eq!(job_state(&cluster, 11), Some(JobState::Running));
        assert!(job_state(&cluster, 12).unwrap().is_terminal());
        assert_eq!(job_state(&cluster, 13), Some(JobState::Completing));
        assert_eq!(job_state(&cluster, 14), Some(JobState::Suspended));
        assert_eq!(job_state(&cluster, 15), Some(JobState::Preempted));

        let reported: Vec<RunningJobStatus> = [10, 11, 12, 13, 14, 15, 999]
            .into_iter()
            .map(|job_id| RunningJobStatus {
                job_id,
                ..Default::default()
            })
            .collect();

        let stale = stale_reported_jobs(&cluster, "n1", &reported);
        assert_eq!(
            stale,
            vec![12],
            "terminal is reclaimed; Pending/Running/Completing/Suspended/Preempted spared, and 999 was never issued"
        );
    }

    /// GATE: a terminal job aged out of the job map must stay reclaimable, or an
    /// agent still holding it keeps that allocation forever.
    #[tokio::test]
    async fn stale_reported_jobs_reclaims_job_evicted_from_memory() {
        use crate::raft::StateMachineApply;
        use spur_core::job::{JobSpec, JobState};
        use spur_core::wal::WalOperation;

        let dir = tempfile::TempDir::new().unwrap();
        let cluster =
            Arc::new(crate::cluster::ClusterManager::new(test_slurm_config(), dir.path()).unwrap());
        let apply = |op: &WalOperation| {
            <crate::cluster::ClusterManager as StateMachineApply>::apply_operation(
                cluster.as_ref(),
                op,
            );
        };

        apply(&WalOperation::JobSubmit {
            job_id: 20,
            spec: Box::new(JobSpec {
                name: "evicted".into(),
                user: "alice".into(),
                num_nodes: 1,
                num_tasks: 1,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            }),
        });
        apply(&WalOperation::job_state_change(
            20,
            JobState::Pending,
            JobState::Cancelled,
        ));
        apply(&WalOperation::EvictTerminalJobs { job_ids: vec![20] });

        assert_eq!(job_state(&cluster, 20), None, "eviction drops the record");
        let issued = cluster.peek_next_job_id();
        assert!(20 < issued, "id 20 was issued by this controller");

        let reported: Vec<RunningJobStatus> = [20, issued, issued + 5]
            .into_iter()
            .map(|job_id| RunningJobStatus {
                job_id,
                ..Default::default()
            })
            .collect();

        assert_eq!(
            stale_reported_jobs(&cluster, "n1", &reported),
            vec![20],
            "the evicted id is reclaimed; ids at or above next_job_id were never issued here"
        );
    }

    /// Pins a known false positive: an array parent id is consumed but never
    /// stored, so it reads as reclaimable. Agents only report dispatched tasks.
    #[tokio::test]
    async fn reclaimable_reports_true_for_a_consumed_but_unstored_id() {
        use crate::raft::StateMachineApply;
        use spur_core::job::JobSpec;
        use spur_core::wal::WalOperation;

        let dir = tempfile::TempDir::new().unwrap();
        let cluster =
            Arc::new(crate::cluster::ClusterManager::new(test_slurm_config(), dir.path()).unwrap());

        for task_id in [31, 32] {
            <crate::cluster::ClusterManager as StateMachineApply>::apply_operation(
                cluster.as_ref(),
                &WalOperation::JobSubmit {
                    job_id: task_id,
                    spec: Box::new(JobSpec {
                        name: "array-task".into(),
                        user: "alice".into(),
                        num_nodes: 1,
                        num_tasks: 1,
                        cpus_per_task: 1,
                        work_dir: "/tmp".into(),
                        array_job_id: Some(30),
                        ..Default::default()
                    }),
                },
            );
        }

        assert_eq!(job_state(&cluster, 30), None, "parent id is never stored");
        assert!(30 < cluster.peek_next_job_id());
        assert!(
            is_reclaimable(&cluster, "n1", 30),
            "an unstored id below the watermark is reclaimable — reachable only if an agent reports it"
        );
    }

    /// A node evicted mid-run keeps the job and its devices; the controller
    /// restarts it elsewhere and must let the old node release.
    #[tokio::test]
    async fn stale_reported_jobs_reclaims_a_job_restarted_on_another_node() {
        use crate::raft::StateMachineApply;
        use spur_core::job::{JobSpec, JobState};
        use spur_core::wal::WalOperation;

        let dir = tempfile::TempDir::new().unwrap();
        let cluster =
            Arc::new(crate::cluster::ClusterManager::new(test_slurm_config(), dir.path()).unwrap());
        let apply = |op: &WalOperation| {
            <crate::cluster::ClusterManager as StateMachineApply>::apply_operation(
                cluster.as_ref(),
                op,
            );
        };

        apply(&WalOperation::JobSubmit {
            job_id: 40,
            spec: Box::new(JobSpec {
                name: "moved".into(),
                user: "alice".into(),
                num_nodes: 1,
                num_tasks: 1,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                ..Default::default()
            }),
        });

        let res = spur_core::resource::ResourceAllocations {
            cpus: 1,
            memory_mb: 0,
            devices: std::collections::HashMap::new(),
        };
        let mut per_node = std::collections::HashMap::new();
        per_node.insert("n2".to_string(), res.clone());
        apply(&WalOperation::job_start(
            40,
            vec!["n2".into()],
            res,
            per_node,
        ));
        apply(&WalOperation::job_state_change(
            40,
            JobState::Pending,
            JobState::Running,
        ));

        let reported = vec![RunningJobStatus {
            job_id: 40,
            ..Default::default()
        }];
        assert_eq!(
            stale_reported_jobs(&cluster, "n1", &reported),
            vec![40],
            "the node it no longer runs on may release it"
        );
        assert!(
            stale_reported_jobs(&cluster, "n2", &reported).is_empty(),
            "the node actually running it must never be told to release"
        );
    }

    /// A job dispatched but not yet started has no nodelist, so the node it is
    /// being launched on must not be told to release it.
    #[tokio::test]
    async fn stale_reported_jobs_spares_a_job_mid_dispatch() {
        use crate::raft::StateMachineApply;
        use spur_core::job::JobSpec;
        use spur_core::wal::WalOperation;

        let dir = tempfile::TempDir::new().unwrap();
        let cluster =
            Arc::new(crate::cluster::ClusterManager::new(test_slurm_config(), dir.path()).unwrap());

        <crate::cluster::ClusterManager as StateMachineApply>::apply_operation(
            cluster.as_ref(),
            &WalOperation::JobSubmit {
                job_id: 50,
                spec: Box::new(JobSpec {
                    name: "launching".into(),
                    user: "alice".into(),
                    num_nodes: 1,
                    num_tasks: 1,
                    cpus_per_task: 1,
                    work_dir: "/tmp".into(),
                    ..Default::default()
                }),
            },
        );

        let reported = vec![RunningJobStatus {
            job_id: 50,
            ..Default::default()
        }];
        assert!(
            stale_reported_jobs(&cluster, "n1", &reported).is_empty(),
            "a Pending job the agent already holds is mid-launch, not stale"
        );
    }

    /// A job's Running state and its nodelist commit as two separate WAL
    /// entries; a heartbeat landing between them must not read the empty
    /// nodelist as authoritative and tell the launching node to release.
    #[tokio::test]
    async fn stale_reported_jobs_spares_running_job_before_job_start_lands() {
        use crate::raft::StateMachineApply;
        use spur_core::job::{JobSpec, JobState};
        use spur_core::wal::WalOperation;

        let dir = tempfile::TempDir::new().unwrap();
        let cluster =
            Arc::new(crate::cluster::ClusterManager::new(test_slurm_config(), dir.path()).unwrap());

        <crate::cluster::ClusterManager as StateMachineApply>::apply_operation(
            cluster.as_ref(),
            &WalOperation::JobSubmit {
                job_id: 60,
                spec: Box::new(JobSpec {
                    name: "starting".into(),
                    user: "alice".into(),
                    num_nodes: 1,
                    num_tasks: 1,
                    cpus_per_task: 1,
                    work_dir: "/tmp".into(),
                    ..Default::default()
                }),
            },
        );
        <crate::cluster::ClusterManager as StateMachineApply>::apply_operation(
            cluster.as_ref(),
            &WalOperation::job_state_change(60, JobState::Pending, JobState::Running),
        );

        assert_eq!(job_state(&cluster, 60), Some(JobState::Running));
        let reported = vec![RunningJobStatus {
            job_id: 60,
            ..Default::default()
        }];
        assert!(
            stale_reported_jobs(&cluster, "n1", &reported).is_empty(),
            "Running with no nodelist yet is not authoritative — the node accepting the launch must be spared"
        );
    }

    /// GATE: a job requeued (Timeout -> Pending) between the reclaim snapshot
    /// and the spawned loop's send must fail the re-check, not just the snapshot.
    #[tokio::test]
    async fn is_reclaimable_false_after_requeue_race() {
        use crate::raft::StateMachineApply;
        use spur_core::job::{JobSpec, JobState};
        use spur_core::wal::WalOperation;

        let dir = tempfile::TempDir::new().unwrap();
        let cluster =
            Arc::new(crate::cluster::ClusterManager::new(test_slurm_config(), dir.path()).unwrap());
        let apply = |op: &WalOperation| {
            <crate::cluster::ClusterManager as StateMachineApply>::apply_operation(
                cluster.as_ref(),
                op,
            );
        };

        apply(&WalOperation::JobSubmit {
            job_id: 77,
            spec: Box::new(JobSpec {
                name: "interactive".into(),
                user: "alice".into(),
                num_nodes: 1,
                num_tasks: 1,
                cpus_per_task: 1,
                work_dir: "/tmp".into(),
                requeue: true,
                ..Default::default()
            }),
        });
        let res = spur_core::resource::ResourceAllocations {
            cpus: 1,
            memory_mb: 0,
            devices: std::collections::HashMap::new(),
        };
        let mut per_node = std::collections::HashMap::new();
        per_node.insert("n1".to_string(), res.clone());
        apply(&WalOperation::job_start(
            77,
            vec!["n1".into()],
            res,
            per_node,
        ));
        apply(&WalOperation::job_state_change(
            77,
            JobState::Pending,
            JobState::Running,
        ));
        apply(&WalOperation::job_state_change(
            77,
            JobState::Running,
            JobState::Timeout,
        ));
        assert!(is_reclaimable(&cluster, "n1", 77), "snapshot sees Timeout");

        // Concurrent requeue lands before the reclaim loop's re-check.
        apply(&WalOperation::job_state_change(
            77,
            JobState::Timeout,
            JobState::Pending,
        ));

        assert!(
            !is_reclaimable(&cluster, "n1", 77),
            "re-check must skip a job requeued since the snapshot"
        );
    }

    /// GATE: `auth.jwt_key` is captured at startup and must NOT change on
    /// `reconfigure`. Swapping it live would instantly invalidate every
    /// outstanding node token. This drives the real capture path
    /// (`resolve_startup_jwt_key`) and the real `reconfigure`, then proves the
    /// running controller still verifies a token minted with the startup key.
    #[tokio::test]
    async fn reconfigure_does_not_adopt_new_jwt_key() {
        use spur_core::admission::{generate_node_token, verify_node_token};

        let dir = tempfile::TempDir::new().unwrap();
        let conf_path = dir.path().join("spur.conf");
        std::fs::write(
            &conf_path,
            "cluster_name = \"test\"\n[auth]\nplugin = \"jwt\"\njwt_key = \"old-secret\"\n",
        )
        .unwrap();

        let config = spur_core::config::SlurmConfig::load_from_file(&conf_path).unwrap();
        let cluster = Arc::new(
            ClusterManager::new_with_config_path(config, dir.path(), Some(conf_path.clone()))
                .unwrap(),
        );
        let handle = crate::raft::start_raft(1, &["[::1]:0".into()], dir.path(), cluster.clone())
            .await
            .unwrap();
        handle
            .raft
            .wait(Some(std::time::Duration::from_secs(5)))
            .metrics(|m| m.current_leader == Some(1), "leader elected")
            .await
            .unwrap();
        cluster.set_raft(handle.raft);

        // The controller captures the signing key exactly here, at startup.
        let startup_key = resolve_startup_jwt_key(&cluster.config());
        assert_eq!(startup_key, "old-secret");
        let token = generate_node_token("node-1", startup_key.as_bytes()).unwrap();

        // Operator edits jwt_key and reconfigures.
        std::fs::write(
            &conf_path,
            "cluster_name = \"test\"\n[auth]\nplugin = \"jwt\"\njwt_key = \"new-secret\"\n",
        )
        .unwrap();
        cluster.reconfigure().unwrap();

        // The live config reflects the new key (proving reconfigure did swap
        // config)...
        assert_eq!(
            cluster.config().auth.jwt_key.as_deref(),
            Some("new-secret"),
            "reconfigure must swap the live config"
        );
        // ...but the controller's captured key is unchanged, so tokens minted
        // with the startup key still verify. A live-reloaded key would reject
        // this token.
        assert_eq!(startup_key, "old-secret", "captured key must not change");
        assert!(
            verify_node_token(&token, startup_key.as_bytes()).is_ok(),
            "outstanding node token must still verify against the startup key"
        );
        assert!(
            verify_node_token(&token, b"new-secret").is_err(),
            "sanity: the token would fail under the new key (proving the key matters)"
        );
    }

    #[test]
    fn submit_rpc_status_maps_internal() {
        use crate::cluster::SubmitError;

        let status = submit_rpc_status(SubmitError::internal("raft propose failed"));
        assert_eq!(status.code(), Code::Internal);
    }

    fn step_test_config() -> spur_core::config::SlurmConfig {
        spur_core::config::SlurmConfig::load_from_str(
            "cluster_name = \"test\"\n\
             [controller]\nfirst_job_id = 1\n\
             [[partitions]]\nname = \"default\"\ndefault = true\nnodes = \"ALL\"\n",
        )
        .unwrap()
    }

    async fn test_service(dir: &tempfile::TempDir) -> ControllerService {
        test_service_with(dir, step_test_config()).await
    }

    /// `test_service` with an explicit config, so a test can exercise a config-gated code path
    /// (e.g. `[cluster] allow_admin_kubeconfig`).
    async fn test_service_with(
        dir: &tempfile::TempDir,
        config: spur_core::config::SlurmConfig,
    ) -> ControllerService {
        use crate::cluster::ClusterManager;
        let cluster = std::sync::Arc::new(ClusterManager::new(config, dir.path()).unwrap());
        let handle = crate::raft::start_raft(1, &["[::1]:0".into()], dir.path(), cluster.clone())
            .await
            .unwrap();
        handle
            .raft
            .wait(Some(std::time::Duration::from_secs(5)))
            .metrics(|m| m.current_leader == Some(1), "leader elected")
            .await
            .unwrap();
        cluster.set_raft(handle.raft.clone());
        let raft = std::sync::Arc::new(handle);
        ControllerService {
            cluster,
            raft: raft.clone(),
            leader_proxy: LeaderProxy::new(raft, BTreeMap::new()),
            client_addrs: BTreeMap::new(),
            rpc_stats: std::sync::Arc::new(RpcStatsCollector::new()),
            sched_stats: std::sync::Arc::new(SchedStatsCollector::new("backfill")),
            control_plane_replicas: 1,
            jwt_key: String::new(),
        }
    }

    // Ping carries the configured cluster name so `scontrol show config` can
    // report it without reading the login node's local config.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ping_reports_configured_cluster_name() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let resp = svc.ping(Request::new(())).await.unwrap().into_inner();
        assert_eq!(resp.cluster_name, "test");
    }

    // Exercises the requeue RPC boundary: gRPC status mapping (auth vs. state
    // precondition) and the requeued-count response the CLI relies on.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requeue_job_rpc_reports_count_and_maps_errors() {
        use spur_core::resource::{ResourceAllocations, ResourceSet};
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        svc.cluster
            .register_node(
                "n1".into(),
                "n1".into(),
                ResourceSet {
                    cpus: 8,
                    memory_mb: 16000,
                    ..Default::default()
                },
                "127.0.0.1".into(),
                6818,
                String::new(),
                String::new(),
                spur_core::node::NodeSource::NativeHost,
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_node("n1").is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        let spec = spur_core::job::JobSpec {
            name: "rq".into(),
            user: "alice".into(),
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            work_dir: "/tmp".into(),
            ..Default::default()
        };
        let job_id = svc.cluster.submit_job(spec).unwrap().job_id;
        for _ in 0..200 {
            if svc.cluster.get_job(job_id).is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        // Unknown job -> FailedPrecondition (non-auth cluster error).
        let err = svc
            .requeue_job(Request::new(RequeueJobRequest {
                job_id: 999_999,
                user: "alice".into(),
                hold: false,
            }))
            .await
            .expect_err("unknown job must error");
        assert_eq!(err.code(), Code::FailedPrecondition);

        // Wrong user -> PermissionDenied (auth precedes the state checks).
        let err = svc
            .requeue_job(Request::new(RequeueJobRequest {
                job_id,
                user: "mallory".into(),
                hold: false,
            }))
            .await
            .expect_err("non-owner must be denied");
        assert_eq!(err.code(), Code::PermissionDenied);

        // Drive to terminal, then a valid owner requeue succeeds and reports the count.
        let res = ResourceAllocations::with_scalar(1, 1000);
        let per_node: std::collections::HashMap<_, _> =
            [("n1".to_string(), res.clone())].into_iter().collect();
        svc.cluster
            .start_job(job_id, vec!["n1".into()], res, per_node)
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_job(job_id).map(|j| j.state) == Some(JobState::Running) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        svc.cluster
            .complete_job(job_id, 0, JobState::Completed)
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_job(job_id).map(|j| j.state) == Some(JobState::Completed) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        let resp = svc
            .requeue_job(Request::new(RequeueJobRequest {
                job_id,
                user: "alice".into(),
                hold: false,
            }))
            .await
            .expect("owner requeue must succeed")
            .into_inner();
        assert_eq!(resp.requeued, 1);
        assert!(resp.skipped.is_empty());
    }

    // An authenticated non-admin cannot requeue another user's job by setting user="root" on
    // the wire. authoritative_user overwrites the field before the owner check.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requeue_job_rejects_spoofed_root_from_authenticated_non_admin() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        let mut req = Request::new(RequeueJobRequest {
            job_id,
            user: "root".into(), // spoof admin on the wire
            hold: false,
        });
        req.extensions_mut().insert(viewer("mallory", false));

        let err = svc
            .requeue_job(req)
            .await
            .expect_err("a spoofed root claim must not bypass the ownership check");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    // A verified admin (whose username is not literally "root") can requeue any job.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requeue_job_admin_override_uses_verified_identity_not_wire_root() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        // A verified admin (name != "root") may requeue another user's job.
        let mut admin_req = Request::new(RequeueJobRequest {
            job_id,
            hold: false,
            ..Default::default()
        });
        admin_req.extensions_mut().insert(viewer("carol", true));
        assert!(
            svc.requeue_job(admin_req).await.is_ok(),
            "a verified admin (non-root) must be able to requeue any job"
        );

        // An unauthenticated caller cannot claim admin by putting user="root" on the wire.
        let err = svc
            .requeue_job(Request::new(RequeueJobRequest {
                job_id,
                user: "root".into(),
                hold: false,
            }))
            .await
            .expect_err("claiming root without a credential must not bypass ownership");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    // A step must NOT be created when the target node is allocated but unregistered: address
    // resolution runs first and returns a retryable Unavailable, so the client's retries can't
    // each leak a step.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_job_step_unregistered_target_creates_no_step() {
        use spur_core::resource::{ResourceAllocations, ResourceSet};
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;

        svc.cluster
            .register_node(
                "n1".into(),
                "n1".into(),
                ResourceSet {
                    cpus: 8,
                    memory_mb: 16000,
                    ..Default::default()
                },
                "127.0.0.1".into(),
                6818,
                String::new(),
                String::new(),
                spur_core::node::NodeSource::NativeHost,
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_node("n1").is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        let mut spec = spur_core::job::JobSpec {
            name: "leak".into(),
            user: "u".into(),
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            work_dir: "/tmp".into(),
            ..Default::default()
        };
        spec.srun_job = true;
        let job_id = svc.cluster.submit_job(spec).unwrap().job_id;
        // Allocate to a real node plus a ghost node that is never registered.
        let res = ResourceAllocations::with_scalar(2, 4000);
        let per_node: std::collections::HashMap<_, _> = [
            ("n1".to_string(), res.clone()),
            ("ghost".to_string(), res.clone()),
        ]
        .into_iter()
        .collect();
        svc.cluster
            .start_job(job_id, vec!["n1".into(), "ghost".into()], res, per_node)
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_job(job_id).map(|j| j.state) == Some(JobState::Running) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        let steps_before = svc.cluster.get_steps(job_id).len();
        // Simulate the client's retry loop: each attempt returns retryable Unavailable and must
        // leave the step count unchanged (the pre-fix bug created one step per attempt).
        for _ in 0..5 {
            let err = svc
                .create_job_step(Request::new(CreateJobStepRequest {
                    job_id,
                    command: vec!["hostname".into()],
                    num_tasks: 1,
                    cpus_per_task: 1,
                    overlap: true,
                    pty: true,
                    winsize: None,
                    node: "ghost".into(),
                    user: "u".into(),
                    ..Default::default()
                }))
                .await
                .expect_err("unregistered target must fail");
            assert_eq!(err.code(), Code::Unavailable);
        }
        assert_eq!(
            svc.cluster.get_steps(job_id).len(),
            steps_before,
            "no step should be created when the target node is unregistered"
        );
    }

    // exec_in_job and create_job_step are what keep job_entry (backing
    // exec_in_job and interactive_session/attach) and the step-dispatch path
    // from ever reaching a node before its LaunchJob is confirmed: both
    // reject here, before any node is even resolved, for a job that hasn't
    // reached Running. Pin that both gates actually exist and fire, since
    // confirm_dispatch_on_nodes closing the race for those RPCs depends on
    // this check never being bypassed or accidentally dropped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exec_in_job_rejects_when_job_not_running() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;

        let spec = spur_core::job::JobSpec {
            name: "pending-exec".into(),
            user: "u".into(),
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            work_dir: "/tmp".into(),
            ..Default::default()
        };
        let job_id = svc.cluster.submit_job(spec).unwrap().job_id;

        let err = svc
            .exec_in_job(Request::new(ExecInJobRequest {
                job_id,
                command: vec!["hostname".into()],
                user: "u".into(),
            }))
            .await
            .expect_err("a still-Pending job must not be exec'd into");
        assert_eq!(err.code(), Code::FailedPrecondition);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_job_step_rejects_when_job_not_running() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;

        let mut spec = spur_core::job::JobSpec {
            name: "pending-step".into(),
            user: "u".into(),
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            work_dir: "/tmp".into(),
            ..Default::default()
        };
        spec.srun_job = true;
        let job_id = svc.cluster.submit_job(spec).unwrap().job_id;

        let err = svc
            .create_job_step(Request::new(CreateJobStepRequest {
                job_id,
                command: vec!["hostname".into()],
                user: "u".into(),
                num_tasks: 1,
                cpus_per_task: 1,
                overlap: true,
                pty: false,
                winsize: None,
                node: String::new(),
                ..Default::default()
            }))
            .await
            .expect_err("a still-Pending job must not accept a new step");
        assert_eq!(err.code(), Code::FailedPrecondition);
    }

    /// Submit and start a single-node job owned by `owner`, returning its id.
    async fn running_job_owned_by(svc: &ControllerService, owner: &str) -> u32 {
        running_job_owned_by_inner(svc, owner, false).await
    }

    /// Like `running_job_owned_by`, but flags the job as interactive so the
    /// keepalive handler treats it as reapable (and thus records last-seen).
    async fn running_interactive_job_owned_by(svc: &ControllerService, owner: &str) -> u32 {
        running_job_owned_by_inner(svc, owner, true).await
    }

    async fn running_job_owned_by_inner(
        svc: &ControllerService,
        owner: &str,
        interactive: bool,
    ) -> u32 {
        use spur_core::resource::{ResourceAllocations, ResourceSet};

        svc.cluster
            .register_node(
                "n1".into(),
                "n1".into(),
                ResourceSet {
                    cpus: 8,
                    memory_mb: 16000,
                    ..Default::default()
                },
                "127.0.0.1".into(),
                6818,
                String::new(),
                String::new(),
                spur_core::node::NodeSource::NativeHost,
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_node("n1").is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        let spec = spur_core::job::JobSpec {
            name: "owned".into(),
            user: owner.into(),
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            work_dir: "/tmp".into(),
            interactive,
            ..Default::default()
        };
        let job_id = svc.cluster.submit_job(spec).unwrap().job_id;

        let res = ResourceAllocations::with_scalar(1, 1000);
        let per_node: std::collections::HashMap<_, _> =
            [("n1".to_string(), res.clone())].into_iter().collect();
        svc.cluster
            .start_job(job_id, vec!["n1".into()], res, per_node)
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_job(job_id).map(|j| j.state) == Some(JobState::Running) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        job_id
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exec_in_job_denies_non_owner() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        let err = svc
            .exec_in_job(Request::new(ExecInJobRequest {
                job_id,
                command: vec!["whoami".into()],
                user: "rsikande".into(),
            }))
            .await
            .expect_err("a non-owner must not exec inside another user's job");

        assert_eq!(err.code(), Code::PermissionDenied);
    }

    /// A REST submission omitting `user` records no owner. Such a job runs as
    /// root, so non-root users must be denied to prevent privilege escalation.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exec_in_job_denies_non_root_on_empty_owner_job() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "").await;

        let err = svc
            .exec_in_job(Request::new(ExecInJobRequest {
                job_id,
                command: vec!["whoami".into()],
                user: "alice".into(),
            }))
            .await
            .expect_err("empty-owner jobs run as root; non-root must be denied");

        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_keepalive_rejects_unknown_job() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;

        let err = svc
            .job_keepalive(Request::new(JobKeepaliveRequest {
                job_id: 999_999,
                user: "ubuntu".into(),
            }))
            .await
            .expect_err("keepalive for a nonexistent job must fail");

        assert_eq!(err.code(), Code::NotFound);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_keepalive_denies_non_owner() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        let err = svc
            .job_keepalive(Request::new(JobKeepaliveRequest {
                job_id,
                user: "rsikande".into(),
            }))
            .await
            .expect_err("a non-owner must not keep another user's allocation alive");

        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_keepalive_rejects_empty_user() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        let err = svc
            .job_keepalive(Request::new(JobKeepaliveRequest {
                job_id,
                user: String::new(),
            }))
            .await
            .expect_err("an empty user must be rejected, not treated as authorized");

        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_keepalive_records_interactive() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_interactive_job_owned_by(&svc, "ubuntu").await;

        assert!(
            svc.cluster.keepalive_last_seen(job_id).is_none(),
            "no keepalive recorded before the RPC"
        );

        svc.job_keepalive(Request::new(JobKeepaliveRequest {
            job_id,
            user: "ubuntu".into(),
        }))
        .await
        .expect("owner keepalive on an interactive job must succeed");

        assert!(
            svc.cluster.keepalive_last_seen(job_id).is_some(),
            "a successful keepalive must record last-seen for the reaper"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_keepalive_skips_non_interactive() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        svc.job_keepalive(Request::new(JobKeepaliveRequest {
            job_id,
            user: "ubuntu".into(),
        }))
        .await
        .expect("keepalive on a non-interactive job must not error");

        assert!(
            svc.cluster.keepalive_last_seen(job_id).is_none(),
            "a non-interactive job must not be tracked for reaping"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_job_step_denies_non_owner() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        let steps_before = svc.cluster.get_steps(job_id).len();
        let err = svc
            .create_job_step(Request::new(CreateJobStepRequest {
                job_id,
                command: vec!["bash".into()],
                num_tasks: 1,
                cpus_per_task: 1,
                overlap: true,
                pty: true,
                winsize: None,
                node: String::new(),
                user: "rsikande".into(),
                ..Default::default()
            }))
            .await
            .expect_err("a non-owner must not attach to another user's job");

        assert_eq!(err.code(), Code::PermissionDenied);
        assert_eq!(
            svc.cluster.get_steps(job_id).len(),
            steps_before,
            "a denied attach must not leave a step behind"
        );
    }

    /// Create a PTY-style step and return its id, so completion tests have a
    /// real step to act on rather than a synthesized id.
    async fn interactive_step_on(svc: &ControllerService, job_id: u32, owner: &str) -> u32 {
        svc.create_job_step(Request::new(CreateJobStepRequest {
            job_id,
            command: vec!["bash".into()],
            num_tasks: 1,
            cpus_per_task: 1,
            overlap: true,
            pty: true,
            winsize: None,
            node: String::new(),
            user: owner.into(),
            ..Default::default()
        }))
        .await
        .expect("owner may create a step")
        .into_inner()
        .step_id
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn complete_job_step_denies_non_owner() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;
        let step_id = interactive_step_on(&svc, job_id, "ubuntu").await;

        let err = svc
            .complete_job_step(Request::new(CompleteJobStepRequest {
                job_id,
                step_id,
                exit_code: 42,
                user: "rsikande".into(),
                ..Default::default()
            }))
            .await
            .expect_err("a non-owner must not complete another user's step");

        assert_eq!(err.code(), Code::PermissionDenied);
        let step = svc
            .cluster
            .get_steps(job_id)
            .into_iter()
            .find(|s| s.step_id == step_id)
            .expect("step still present");
        assert!(
            !step.state.is_terminal(),
            "denied call must not close the step"
        );
        assert_eq!(step.exit_code, None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn complete_job_step_records_owner_exit_code() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;
        let step_id = interactive_step_on(&svc, job_id, "ubuntu").await;

        svc.complete_job_step(Request::new(CompleteJobStepRequest {
            job_id,
            step_id,
            exit_code: 7,
            user: "ubuntu".into(),
            ..Default::default()
        }))
        .await
        .expect("the owner may complete their own step");

        let step = svc
            .cluster
            .get_steps(job_id)
            .into_iter()
            .find(|s| s.step_id == step_id)
            .expect("step still present");
        assert_eq!(step.exit_code, Some(7));
        assert!(step.state.is_terminal());
        let job = svc.cluster.get_job(job_id).expect("job present");
        assert_eq!(
            job.derived_exit_code, 7,
            "step exit must reach DerivedExitCode"
        );
    }

    /// Owning the job must not let a client address the batch step or invent an id.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn complete_job_step_rejects_reserved_and_unknown_step_ids() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        let reserved = svc
            .complete_job_step(Request::new(CompleteJobStepRequest {
                job_id,
                step_id: spur_core::step::STEP_BATCH,
                exit_code: 99,
                user: "ubuntu".into(),
                ..Default::default()
            }))
            .await
            .expect_err("reserved step ids are not client-addressable");
        assert_eq!(reserved.code(), Code::InvalidArgument);

        let unknown = svc
            .complete_job_step(Request::new(CompleteJobStepRequest {
                job_id,
                step_id: 4242,
                exit_code: 99,
                user: "ubuntu".into(),
                ..Default::default()
            }))
            .await
            .expect_err("an unknown step id must not be recorded");
        assert_eq!(unknown.code(), Code::NotFound);

        let job = svc.cluster.get_job(job_id).expect("job present");
        assert_eq!(
            job.derived_exit_code, 0,
            "a rejected completion must not move DerivedExitCode"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn complete_job_step_keeps_the_first_verdict() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;
        let step_id = interactive_step_on(&svc, job_id, "ubuntu").await;

        for exit_code in [3, 0] {
            svc.complete_job_step(Request::new(CompleteJobStepRequest {
                job_id,
                step_id,
                exit_code,
                user: "ubuntu".into(),
                ..Default::default()
            }))
            .await
            .expect("a repeat report is accepted as a no-op");
        }

        let step = svc
            .cluster
            .get_steps(job_id)
            .into_iter()
            .find(|s| s.step_id == step_id)
            .expect("step still present");
        assert_eq!(
            step.exit_code,
            Some(3),
            "a late report must not overwrite a recorded verdict"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_job_step_allows_owner() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        let resp = svc
            .create_job_step(Request::new(CreateJobStepRequest {
                job_id,
                command: vec!["bash".into()],
                num_tasks: 1,
                cpus_per_task: 1,
                overlap: true,
                pty: true,
                winsize: None,
                node: String::new(),
                user: "ubuntu".into(),
                ..Default::default()
            }))
            .await
            .expect("the owner must be allowed to attach");

        assert_eq!(resp.into_inner().node_addr, "127.0.0.1:6818");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_job_step_allows_uid_match_despite_username_mismatch() {
        use spur_core::job::JobState;
        use spur_core::resource::{ResourceAllocations, ResourceSet};

        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;

        svc.cluster
            .register_node(
                "n1".into(),
                "n1".into(),
                ResourceSet {
                    cpus: 8,
                    memory_mb: 16000,
                    ..Default::default()
                },
                "127.0.0.1".into(),
                6818,
                String::new(),
                String::new(),
                spur_core::node::NodeSource::NativeHost,
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_node("n1").is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        let owner_uid = 1000;
        let spec = spur_core::job::JobSpec {
            name: "uid-owned".into(),
            user: "jwt-subject".into(),
            uid: owner_uid,
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            work_dir: "/tmp".into(),
            ..Default::default()
        };
        let job_id = svc.cluster.submit_job(spec).unwrap().job_id;
        let res = ResourceAllocations::with_scalar(1, 1000);
        let per_node: std::collections::HashMap<_, _> =
            [("n1".to_string(), res.clone())].into_iter().collect();
        svc.cluster
            .start_job(job_id, vec!["n1".into()], res, per_node)
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_job(job_id).map(|j| j.state) == Some(JobState::Running) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        let resp = svc
            .create_job_step(Request::new(CreateJobStepRequest {
                job_id,
                command: vec!["bash".into()],
                num_tasks: 1,
                cpus_per_task: 1,
                overlap: true,
                pty: false,
                winsize: None,
                node: String::new(),
                user: "local-wire-name".into(),
                uid: owner_uid,
                container: None,
            }))
            .await
            .expect("uid match must authorize step creation despite username mismatch");

        assert_eq!(resp.into_inner().node_addr, "127.0.0.1:6818");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_job_denies_non_owner() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        let err = svc
            .update_job(Request::new(UpdateJobRequest {
                job_id,
                comment: Some("owned".into()),
                user: "rsikande".into(),
                ..Default::default()
            }))
            .await
            .expect_err("a non-owner must not modify another user's job");

        assert_eq!(err.code(), Code::PermissionDenied);
        assert_eq!(
            svc.cluster.get_job(job_id).unwrap().spec.comment,
            None,
            "a denied update must not mutate the job"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_job_allows_owner() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        svc.update_job(Request::new(UpdateJobRequest {
            job_id,
            comment: Some("reviewed".into()),
            user: "ubuntu".into(),
            ..Default::default()
        }))
        .await
        .expect("the owner must be allowed to modify their job");

        assert_eq!(
            svc.cluster.get_job(job_id).unwrap().spec.comment,
            Some("reviewed".into())
        );
    }

    /// The owner check must key off the *authenticated* identity, not the
    /// wire-asserted `user`: a caller cannot spoof the owner's name to slip past
    /// the gate. `authoritative_user` overwrites the field before the check.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_job_rejects_spoofed_owner_when_authenticated() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        let mut req = Request::new(UpdateJobRequest {
            job_id,
            comment: Some("owned".into()),
            user: "ubuntu".into(), // spoof the owner's name on the wire
            ..Default::default()
        });
        req.extensions_mut().insert(spur_core::auth::Identity {
            user: "rsikande".into(),
            uid: 1001,
            gid: 1001,
            is_admin: false,
        });

        let err = svc
            .update_job(req)
            .await
            .expect_err("a spoofed owner name must not bypass the ownership check");

        assert_eq!(err.code(), Code::PermissionDenied);
        assert_eq!(
            svc.cluster.get_job(job_id).unwrap().spec.comment,
            None,
            "a denied update must not mutate the job"
        );
    }

    // `is_internal` is derived from the verified identity (an admin), not the wire `user` string, so
    // (a) an admin whose username is not literally "root" is still treated as internal, and (b) an
    // unauthenticated caller cannot bypass ownership by claiming user = "root".
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_job_admin_override_uses_verified_identity_not_wire_root() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        // A verified admin whose name is not "root" may modify another user's job.
        let mut admin_req = Request::new(UpdateJobRequest {
            job_id,
            comment: Some("by-admin".into()),
            ..Default::default()
        });
        admin_req.extensions_mut().insert(viewer("carol", true));
        assert!(
            svc.update_job(admin_req).await.is_ok(),
            "a verified admin (non-root) must be able to modify any job"
        );

        // An unauthenticated caller cannot spoof admin by putting user = "root" on the wire.
        let err = svc
            .update_job(Request::new(UpdateJobRequest {
                job_id,
                comment: Some("spoof".into()),
                user: "root".into(),
                ..Default::default()
            }))
            .await
            .expect_err("claiming root without a credential must not bypass ownership");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_step_denies_non_owner() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let job_id = running_job_owned_by(&svc, "ubuntu").await;

        let err = svc
            .run_step(Request::new(RunStepRequest {
                job_id,
                command: vec!["id".into()],
                step_id: 0,
                user: "rsikande".into(),
                ..Default::default()
            }))
            .await
            .expect_err("a non-owner must not run a step in another user's allocation");

        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_step_allows_uid_match_despite_username_mismatch() {
        use spur_core::job::JobState;
        use spur_core::resource::{ResourceAllocations, ResourceSet};
        use spur_core::step::{JobStep, StepState, TaskDistribution};

        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;

        svc.cluster
            .register_node(
                "n1".into(),
                "n1".into(),
                ResourceSet {
                    cpus: 8,
                    memory_mb: 16000,
                    ..Default::default()
                },
                "127.0.0.1".into(),
                6818,
                String::new(),
                String::new(),
                spur_core::node::NodeSource::NativeHost,
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_node("n1").is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        let owner_uid = 1000;
        let spec = spur_core::job::JobSpec {
            name: "uid-owned".into(),
            user: "jwt-subject".into(),
            uid: owner_uid,
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            work_dir: "/tmp".into(),
            ..Default::default()
        };
        let job_id = svc.cluster.submit_job(spec).unwrap().job_id;
        let res = ResourceAllocations::with_scalar(1, 1000);
        let per_node: std::collections::HashMap<_, _> =
            [("n1".to_string(), res.clone())].into_iter().collect();
        svc.cluster
            .start_job(job_id, vec!["n1".into()], res, per_node)
            .unwrap();
        for _ in 0..200 {
            if svc.cluster.get_job(job_id).map(|j| j.state) == Some(JobState::Running) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        let allocated_nodes = svc.cluster.get_job(job_id).unwrap().allocated_nodes.clone();
        svc.cluster
            .create_step(JobStep {
                job_id,
                step_id: 0,
                name: "id".into(),
                state: StepState::Running,
                num_tasks: 1,
                cpus_per_task: 1,
                resources: ResourceAllocations::default(),
                nodes: allocated_nodes,
                distribution: TaskDistribution::Block,
                start_time: Some(chrono::Utc::now()),
                end_time: None,
                exit_code: None,
            })
            .expect("test setup: seed step for run_step");

        svc.run_step(Request::new(RunStepRequest {
            job_id,
            command: vec!["id".into()],
            uid: owner_uid,
            step_id: 0,
            user: "local-wire-name".into(),
            ..Default::default()
        }))
        .await
        .expect("uid match must authorize step dispatch despite username mismatch");
    }

    async fn assign_ha_control_plane(svc: &ControllerService) {
        use spur_core::k0s::{K0sPhase, K0sRole};
        use spur_core::node::NodeSource;
        for (i, name) in ["cp-a", "cp-b", "cp-c"].iter().enumerate() {
            svc.cluster
                .register_node(
                    (*name).into(),
                    (*name).into(),
                    spur_core::resource::ResourceSet {
                        cpus: 4,
                        memory_mb: 8000,
                        ..Default::default()
                    },
                    "127.0.0.1".into(),
                    6818 + i as u16,
                    String::new(),
                    String::new(),
                    NodeSource::NativeHost,
                    std::collections::HashMap::new(),
                    true,
                )
                .unwrap();
            svc.cluster
                .assign_node_k0s(
                    name,
                    K0sRole::Controller,
                    &format!("10.44.0.{}", i + 1),
                    "10.42.0.0/24",
                )
                .unwrap();
        }
        svc.cluster
            .set_k0s_phase(
                K0sPhase::Ready,
                Some("cp-a".into()),
                vec!["cp-a".into(), "cp-b".into(), "cp-c".into()],
                Vec::new(),
                false,
            )
            .unwrap();
    }

    // A bare `spur k8s up` (no flags) on an assigned 3-CP cluster is idempotent: it must target the
    // recorded set, not re-resolve against the replicas=1 config default and spuriously reject.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_bare_reup_on_ha_cluster_is_idempotent() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        assign_ha_control_plane(&svc).await;

        let resp = svc
            .cluster_up(Request::new(ClusterUpRequest {
                // explicit admin caller: an empty caller is no longer treated as admin
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect("bare re-up of an assigned HA cluster must be accepted")
            .into_inner();
        assert!(resp.accepted);
    }

    // An explicit replica count that doesn't match the assigned set is still rejected.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_mismatched_replicas_on_ha_cluster_is_rejected() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        assign_ha_control_plane(&svc).await;

        let err = svc
            .cluster_up(Request::new(ClusterUpRequest {
                // explicit admin caller: an empty caller is no longer treated as admin
                caller: "root".into(),
                control_plane_replicas: Some(1),
                ..Default::default()
            }))
            .await
            .expect_err("shrinking an assigned 3-CP cluster must be rejected");
        assert_eq!(err.code(), Code::FailedPrecondition);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_denied_for_non_admin_caller() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        svc.cluster
            .association_cache()
            .insert_association("alice", "physics");
        let err = svc
            .cluster_up(Request::new(ClusterUpRequest {
                caller: "alice".into(),
                ..Default::default()
            }))
            .await
            .expect_err("a non-admin caller must not bring the cluster up");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    // Real RPC path, not just the raw cache method: a named caller must not become admin just
    // because the association cache hasn't loaded yet (fail closed on a cold cache).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_denied_for_non_root_caller_on_cold_cache_with_accounting_enabled() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut config = step_test_config();
        config.accounting.database_url = "postgresql://unused-in-test".into();
        let svc = test_service_with(&dir, config).await;
        assert!(!svc.cluster.association_cache().is_loaded());

        let err = svc
            .cluster_up(Request::new(ClusterUpRequest {
                caller: "alice".into(),
                ..Default::default()
            }))
            .await
            .expect_err("a cold association cache must not grant admin");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_down_denied_for_non_admin_caller() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        svc.cluster
            .association_cache()
            .insert_association("alice", "physics");
        let err = svc
            .cluster_down(Request::new(ClusterDownRequest {
                caller: "alice".into(),
                ..Default::default()
            }))
            .await
            .expect_err("a non-admin caller must not tear the cluster down");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_allowed_for_admin_level_caller() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        svc.cluster
            .register_node(
                "cp-a".into(),
                "cp-a".into(),
                spur_core::resource::ResourceSet {
                    cpus: 4,
                    memory_mb: 8000,
                    ..Default::default()
                },
                "127.0.0.1".into(),
                6818,
                String::new(),
                String::new(),
                spur_core::node::NodeSource::NativeHost,
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();
        svc.cluster
            .association_cache()
            .insert_admin_level("carol", "Admin");
        let resp = svc
            .cluster_up(Request::new(ClusterUpRequest {
                caller: "carol".into(),
                control_plane_replicas: Some(1),
                ..Default::default()
            }))
            .await
            .expect("an Admin-level caller may bring the cluster up")
            .into_inner();
        assert!(resp.accepted);
    }

    async fn register_plain_node(svc: &ControllerService, name: &str, port: u16) {
        svc.cluster
            .register_node(
                name.into(),
                name.into(),
                spur_core::resource::ResourceSet {
                    cpus: 4,
                    memory_mb: 8000,
                    ..Default::default()
                },
                "127.0.0.1".into(),
                port,
                String::new(),
                String::new(),
                spur_core::node::NodeSource::NativeHost,
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_scopes_membership_to_selected_nodes() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        for (i, n) in ["node-a", "node-b", "node-c"].iter().enumerate() {
            register_plain_node(&svc, n, 6818 + i as u16).await;
        }
        let resp = svc
            .cluster_up(Request::new(ClusterUpRequest {
                // explicit admin caller: an empty caller is no longer treated as admin
                caller: "root".into(),
                nodes: "node-a,node-b".into(),
                ..Default::default()
            }))
            .await
            .expect("scoped up accepted")
            .into_inner();
        assert!(resp.accepted);
        assert_eq!(
            svc.cluster.k0s_state().member_nodes,
            vec!["node-a", "node-b"]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_rejects_control_plane_outside_scope() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        for (i, n) in ["node-a", "node-b", "node-c"].iter().enumerate() {
            register_plain_node(&svc, n, 6818 + i as u16).await;
        }
        let err = svc
            .cluster_up(Request::new(ClusterUpRequest {
                // explicit admin caller: an empty caller is no longer treated as admin
                caller: "root".into(),
                nodes: "node-a,node-b".into(),
                control_plane_nodes: vec!["node-c".into()],
                ..Default::default()
            }))
            .await
            .expect_err("a control plane outside the node scope must be rejected");
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    async fn scoped_assigned_cluster(svc: &ControllerService) {
        for (i, n) in ["node-a", "node-b", "node-c"].iter().enumerate() {
            register_plain_node(svc, n, 6818 + i as u16).await;
        }
        svc.cluster
            .set_k0s_phase(
                spur_core::k0s::K0sPhase::Provisioning,
                Some("node-a".into()),
                vec!["node-a".into()],
                vec!["node-a".into(), "node-b".into()],
                false,
            )
            .unwrap();
        svc.cluster
            .assign_node_k0s(
                "node-a",
                spur_core::k0s::K0sRole::Single,
                "10.44.0.1",
                "10.42.0.0/24",
            )
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_add_nodes_unions_into_scoped_member_set() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        // Scoped to {node-a, node-b}; node-c is registered but out of scope.
        scoped_assigned_cluster(&svc).await;
        let resp = svc
            .cluster_add_nodes(Request::new(ClusterAddNodesRequest {
                nodes: "node-c".into(),
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect("add-nodes accepted")
            .into_inner();
        assert!(resp.accepted);
        assert_eq!(
            svc.cluster.k0s_state().member_nodes,
            vec!["node-a", "node-b", "node-c"]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_add_nodes_rejected_on_whole_inventory_cluster() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        for (i, n) in ["node-a", "node-b"].iter().enumerate() {
            register_plain_node(&svc, n, 6818 + i as u16).await;
        }
        // Whole-inventory up (empty member_nodes) — new nodes auto-enroll, so add-nodes is rejected.
        svc.cluster
            .set_k0s_phase(
                spur_core::k0s::K0sPhase::Ready,
                Some("node-a".into()),
                vec!["node-a".into()],
                Vec::new(),
                false,
            )
            .unwrap();
        let err = svc
            .cluster_add_nodes(Request::new(ClusterAddNodesRequest {
                nodes: "node-b".into(),
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect_err("add-nodes on a whole-inventory cluster must be rejected");
        assert_eq!(err.code(), Code::FailedPrecondition);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_add_nodes_rejects_control_plane_node() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        scoped_assigned_cluster(&svc).await; // control plane = node-a
        let err = svc
            .cluster_add_nodes(Request::new(ClusterAddNodesRequest {
                nodes: "node-a".into(),
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect_err("adding a control-plane node must be rejected");
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_remove_nodes_rejects_control_plane_node() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        scoped_assigned_cluster(&svc).await; // control plane = node-a
        let err = svc
            .cluster_remove_nodes(Request::new(ClusterRemoveNodesRequest {
                nodes: "node-a".into(),
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect_err("removing a control-plane node must be rejected");
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_remove_nodes_rejects_unregistered_node() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        scoped_assigned_cluster(&svc).await;
        let err = svc
            .cluster_remove_nodes(Request::new(ClusterRemoveNodesRequest {
                nodes: "ghost".into(),
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect_err("removing an unregistered node must be rejected");
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_remove_nodes_rejected_when_down() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        for (i, n) in ["node-a", "node-b"].iter().enumerate() {
            register_plain_node(&svc, n, 6818 + i as u16).await;
        }
        // Cluster never brought up (phase Down) — nothing to remove.
        let err = svc
            .cluster_remove_nodes(Request::new(ClusterRemoveNodesRequest {
                nodes: "node-b".into(),
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect_err("remove on a down cluster must be rejected");
        assert_eq!(err.code(), Code::FailedPrecondition);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_remove_nodes_denied_for_non_admin_caller() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        scoped_assigned_cluster(&svc).await;
        let err = svc
            .cluster_remove_nodes(Request::new(ClusterRemoveNodesRequest {
                nodes: "node-b".into(),
                caller: "mallory".into(),
                ..Default::default()
            }))
            .await
            .expect_err("non-admin must not remove nodes");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_remove_nodes_rejected_on_whole_inventory_cluster() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        for (i, n) in ["node-a", "node-b"].iter().enumerate() {
            register_plain_node(&svc, n, 6818 + i as u16).await;
        }
        // Whole-inventory up (empty member_nodes) — a removed node would just re-enroll, so reject.
        svc.cluster
            .set_k0s_phase(
                spur_core::k0s::K0sPhase::Ready,
                Some("node-a".into()),
                vec!["node-a".into()],
                Vec::new(),
                false,
            )
            .unwrap();
        let err = svc
            .cluster_remove_nodes(Request::new(ClusterRemoveNodesRequest {
                nodes: "node-b".into(),
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect_err("remove on a whole-inventory cluster must be rejected");
        assert_eq!(err.code(), Code::FailedPrecondition);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_remove_nodes_rejects_emptying_the_member_set() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        // Scoped to exactly {node-a (CP), node-b}; removing node-b alone is fine, but requesting the
        // whole member set must be refused rather than flip the cluster to whole-inventory.
        scoped_assigned_cluster(&svc).await;
        let err = svc
            .cluster_remove_nodes(Request::new(ClusterRemoveNodesRequest {
                nodes: "node-a,node-b".into(),
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect_err("removing every member must be rejected");
        assert_eq!(err.code(), Code::FailedPrecondition);
        assert!(
            err.message().contains("would empty the cluster"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_bare_reup_preserves_recorded_scope() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        scoped_assigned_cluster(&svc).await;
        let resp = svc
            .cluster_up(Request::new(ClusterUpRequest {
                // explicit admin caller: an empty caller is no longer treated as admin
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect("bare re-up accepted")
            .into_inner();
        assert!(resp.accepted);
        assert_eq!(
            svc.cluster.k0s_state().member_nodes,
            vec!["node-a", "node-b"]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_with_identical_scope_is_a_noop() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        scoped_assigned_cluster(&svc).await;
        // Same scope expressed again (not a bare re-up) must not rewrite the recorded state.
        let resp = svc
            .cluster_up(Request::new(ClusterUpRequest {
                // explicit admin caller: an empty caller is no longer treated as admin
                caller: "root".into(),
                nodes: "node-a,node-b".into(),
                ..Default::default()
            }))
            .await
            .expect("re-up with identical scope must succeed")
            .into_inner();
        assert!(resp.accepted);
        assert!(resp.message.contains("already up"), "got: {}", resp.message);
        assert_eq!(
            svc.cluster.k0s_state().member_nodes,
            vec!["node-a", "node-b"]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_rejects_scope_change_on_assigned_cluster() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        scoped_assigned_cluster(&svc).await;
        let err = svc
            .cluster_up(Request::new(ClusterUpRequest {
                // explicit admin caller: an empty caller is no longer treated as admin
                caller: "root".into(),
                nodes: "node-a,node-c".into(),
                ..Default::default()
            }))
            .await
            .expect_err("changing scope on an assigned cluster must be rejected");
        assert_eq!(err.code(), Code::FailedPrecondition);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_rejected_while_teardown_drains_roles() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        scoped_assigned_cluster(&svc).await;
        // down clears the recorded scope/CP immediately; node-a's role drains on a later tick. A
        // re-up in that window must be rejected, not silently widen to the whole inventory.
        svc.cluster
            .set_k0s_phase(
                spur_core::k0s::K0sPhase::Down,
                None,
                Vec::new(),
                Vec::new(),
                false,
            )
            .unwrap();
        let err = svc
            .cluster_up(Request::new(ClusterUpRequest {
                // explicit admin caller: an empty caller is no longer treated as admin
                caller: "root".into(),
                ..Default::default()
            }))
            .await
            .expect_err("re-up during teardown must be rejected");
        assert_eq!(err.code(), Code::FailedPrecondition);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_kubeconfig_admin_flag_denied_for_non_admin() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        svc.cluster
            .association_cache()
            .insert_association("alice", "physics");
        let err = svc
            .cluster_kubeconfig(Request::new(ClusterKubeconfigRequest {
                caller: "alice".into(),
                admin: true,
                ..Default::default()
            }))
            .await
            .expect_err("a non-admin must not get the cluster-admin kubeconfig");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_kubeconfig_other_user_denied_for_non_admin() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        svc.cluster
            .association_cache()
            .insert_default_account("alice", "physics");
        let err = svc
            .cluster_kubeconfig(Request::new(ClusterKubeconfigRequest {
                caller: "alice".into(),
                user: "bob".into(),
                ..Default::default()
            }))
            .await
            .expect_err("a non-admin must not mint another user's kubeconfig");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_kubeconfig_bare_self_scope_clears_authz_for_non_admin() {
        // A non-admin bare request (own scope) clears authz; the absent control-plane agent then
        // yields Unavailable, never PermissionDenied/InvalidArgument.
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        svc.cluster
            .association_cache()
            .insert_default_account("alice", "physics");
        let err = svc
            .cluster_kubeconfig(Request::new(ClusterKubeconfigRequest {
                caller: "alice".into(),
                ..Default::default()
            }))
            .await
            .expect_err("no control-plane agent is running in this test");
        assert_eq!(err.code(), Code::Unavailable);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_kubeconfig_admin_may_target_other_user() {
        // An Admin caller clears authz for another user's scope; failure is the absent agent
        // (Unavailable), not PermissionDenied.
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let cache = svc.cluster.association_cache();
        cache.insert_admin_level("carol", "Admin");
        cache.insert_default_account("bob", "chem");
        let err = svc
            .cluster_kubeconfig(Request::new(ClusterKubeconfigRequest {
                caller: "carol".into(),
                user: "bob".into(),
                ..Default::default()
            }))
            .await
            .expect_err("no control-plane agent is running in this test");
        assert_eq!(err.code(), Code::Unavailable);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_kubeconfig_admin_flag_denied_when_not_explicitly_enabled() {
        // Default posture: serving the cluster-admin credential over RPC is off, so even a caller
        // the association cache calls Admin is refused. `caller` is unauthenticated, so the opt-in —
        // not the admin check — is what protects this credential.
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        svc.cluster
            .association_cache()
            .insert_admin_level("carol", "Admin");
        let err = svc
            .cluster_kubeconfig(Request::new(ClusterKubeconfigRequest {
                caller: "carol".into(),
                admin: true,
                ..Default::default()
            }))
            .await
            .expect_err("admin kubeconfig is disabled by default");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_kubeconfig_admin_flag_allowed_for_admin_level_caller_when_enabled() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut config = step_test_config();
        config.cluster.allow_admin_kubeconfig = true;
        let svc = test_service_with(&dir, config).await;
        svc.cluster
            .association_cache()
            .insert_admin_level("carol", "Admin");
        let err = svc
            .cluster_kubeconfig(Request::new(ClusterKubeconfigRequest {
                caller: "carol".into(),
                admin: true,
                ..Default::default()
            }))
            .await
            .expect_err("no control-plane agent is running in this test");
        // Unavailable (not PermissionDenied) proves it passed both the opt-in and the admin check.
        assert_eq!(err.code(), Code::Unavailable);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_kubeconfig_admin_flag_denied_for_non_admin_even_when_enabled() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut config = step_test_config();
        config.cluster.allow_admin_kubeconfig = true;
        let svc = test_service_with(&dir, config).await;
        let err = svc
            .cluster_kubeconfig(Request::new(ClusterKubeconfigRequest {
                caller: "mallory".into(),
                admin: true,
                ..Default::default()
            }))
            .await
            .expect_err("non-admin must not receive the cluster-admin kubeconfig");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cluster_up_denied_for_named_caller_when_accounting_off() {
        // Cold cache = accounting disabled: a named caller is not admin, so up fails closed.
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        let err = svc
            .cluster_up(Request::new(ClusterUpRequest {
                caller: "alice".into(),
                ..Default::default()
            }))
            .await
            .expect_err("a named caller must be denied when accounting is off");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[test]
    fn select_step_node_empty_request_uses_first_allocated() {
        let allocated = vec!["node001".to_string(), "node002".to_string()];
        assert_eq!(select_step_node(&allocated, ""), Ok("node001"));
    }

    #[test]
    fn select_step_node_named_request_targets_that_node() {
        let allocated = vec!["node001".to_string(), "node002".to_string()];
        assert_eq!(select_step_node(&allocated, "node002"), Ok("node002"));
    }

    #[test]
    fn select_step_node_rejects_node_outside_allocation() {
        let allocated = vec!["node001".to_string(), "node002".to_string()];
        let err = select_step_node(&allocated, "node999").unwrap_err();
        assert!(err.contains("node999"));
        assert!(err.contains("not allocated"));
    }

    #[test]
    fn select_step_node_empty_request_no_nodes_errors() {
        let allocated: Vec<String> = Vec::new();
        assert!(select_step_node(&allocated, "").is_err());
    }

    #[test]
    fn select_step_node_rejects_comma_joined_request() {
        let allocated = vec!["node001".to_string(), "node002".to_string()];
        assert!(select_step_node(&allocated, "node001,node002").is_err());
    }

    #[test]
    fn proto_to_job_spec_sanitizes_the_submit_line_it_stores() {
        let spec = |submit_line: &str| spur_proto::proto::JobSpec {
            name: "j".into(),
            submit_line: submit_line.into(),
            ..Default::default()
        };

        let stored = |line: &str| {
            proto_to_job_spec(spec(line))
                .expect("valid spec")
                .submit_line
        };

        assert_eq!(stored(""), None);
        assert_eq!(stored("sbatch job.sh"), Some("sbatch job.sh".to_string()));
        // The escape must be neutralized on the way in, not just by the printer.
        assert_eq!(
            stored("sbatch \u{1b}[2Kjob.sh"),
            Some("sbatch  [2Kjob.sh".to_string())
        );
    }

    #[test]
    fn sanitize_submit_line_neutralizes_terminal_control_characters() {
        // A client-asserted field printed straight to an operator's terminal.
        assert_eq!(
            sanitize_submit_line("sbatch \u{1b}[2Kjob.sh"),
            Some("sbatch  [2Kjob.sh".to_string())
        );
        // Bidi override renders deceptively but is not char::is_control.
        assert_eq!(
            sanitize_submit_line("sbatch \u{202e}job.sh"),
            Some("sbatch  job.sh".to_string())
        );
        assert_eq!(sanitize_submit_line("\u{1b}\u{7}"), None);
    }

    #[test]
    fn sanitize_submit_line_replaces_controls_rather_than_joining_tokens() {
        // Deleting the newline would render `a b` as the single token `ab`.
        assert_eq!(
            sanitize_submit_line("sbatch --wrap 'a\nb'"),
            Some("sbatch --wrap 'a b'".to_string())
        );
    }

    #[test]
    fn sanitize_submit_line_bounds_bytes_not_chars() {
        let ascii = "x".repeat(MAX_SUBMIT_LINE_BYTES * 2);
        let out = sanitize_submit_line(&ascii).unwrap();
        assert!(out.len() <= MAX_SUBMIT_LINE_BYTES + '\u{2026}'.len_utf8());
        assert!(out.ends_with('\u{2026}'), "truncation must be visible");

        // Multibyte input is where a char-based bound silently stores ~4x the
        // budget into every Raft entry.
        let wide = "\u{4e16}".repeat(MAX_SUBMIT_LINE_BYTES);
        let out = sanitize_submit_line(&wide).unwrap();
        assert!(
            out.len() <= MAX_SUBMIT_LINE_BYTES + '\u{2026}'.len_utf8(),
            "bounded in bytes, got {}",
            out.len()
        );
    }

    #[test]
    fn annotate_jobs_fills_only_pending_jobs_from_the_planned_map() {
        use std::collections::HashMap;

        let start = chrono::Utc::now() + chrono::Duration::minutes(5);
        let by_job = HashMap::from([
            (7u32, (vec!["n1".to_string(), "n2".to_string()], start)),
            (9u32, (vec!["n3".to_string()], start)),
        ]);

        let mut jobs = vec![
            JobInfo {
                job_id: 7,
                state: spur_core::job::JobState::Pending.to_proto_i32(),
                ..Default::default()
            },
            JobInfo {
                job_id: 9,
                state: spur_core::job::JobState::Running.to_proto_i32(),
                ..Default::default()
            },
        ];
        apply_planned_reservations(&mut jobs, &by_job);

        assert!(jobs[0].planned_start_time.is_some());
        assert_eq!(jobs[0].sched_nodelist, "n[1-2]");
        // A running job holds no future slot; a leftover entry must not surface.
        assert!(jobs[1].planned_start_time.is_none());
        assert!(jobs[1].sched_nodelist.is_empty());
    }

    #[test]
    fn redaction_strips_the_planned_nodelist() {
        // Ordering guard: annotation runs before redaction, so the planned list
        // must not reappear on a redacted record.
        let mut info = JobInfo {
            sched_nodelist: "gpu-b-[01-04]".into(),
            req_tres: "cpu=16,mem=64G,gres/gpu=8".into(),
            features: "mi300x".into(),
            min_cpus_node: 16,
            min_memory_node_mb: 65536,
            min_memory_is_per_cpu: true,
            ..Default::default()
        };
        redact_sensitive_job_info(&mut info);
        assert!(info.sched_nodelist.is_empty());
        // The requested shape restates the allocated detail, so it goes too.
        assert!(info.req_tres.is_empty());
        assert!(info.features.is_empty());
        assert_eq!(info.min_cpus_node, 0);
        assert_eq!(info.min_memory_node_mb, 0);
        assert!(!info.min_memory_is_per_cpu);
    }

    #[test]
    fn batch_flag_marks_only_batch_submissions() {
        use spur_core::job::{Job, JobSpec};

        let with = |spec: JobSpec| job_to_proto(&Job::new(1, spec)).batch_flag;
        let script = || Some("#!/bin/bash\nhostname\n".to_string());

        assert!(with(JobSpec {
            script: script(),
            ..Default::default()
        }));
        // srun and salloc synthesize a script, so it cannot be the sole signal.
        assert!(!with(JobSpec {
            script: script(),
            srun_job: true,
            ..Default::default()
        }));
        assert!(!with(JobSpec {
            script: script(),
            interactive: true,
            ..Default::default()
        }));
        assert!(!with(JobSpec::default()));
    }

    #[test]
    fn job_to_proto_reports_requested_placement_and_submit_line() {
        use spur_core::job::{Job, JobSpec};

        let job = Job::new(
            7,
            JobSpec {
                nodelist: Some("node[1-2]".into()),
                exclude: Some("node9".into()),
                constraint: Some("mi300x".into()),
                dependency: vec!["afterok:5".into()],
                submit_line: Some("sbatch -w node[1-2] job.sh".into()),
                ..Default::default()
            },
        );

        let info = job_to_proto(&job);
        assert_eq!(info.req_nodelist, "node[1-2]");
        assert_eq!(info.exc_nodelist, "node9");
        assert_eq!(info.features, "mi300x");
        assert_eq!(info.dependency, vec!["afterok:5".to_string()]);
        assert_eq!(info.submit_line, "sbatch -w node[1-2] job.sh");
        // The allocated list stays empty while pending: this is exactly the
        // distinction that made a pinned pending job undiagnosable.
        assert!(info.nodelist.is_empty());
    }

    #[test]
    fn job_to_proto_leaves_last_sched_eval_unset_until_evaluated() {
        use spur_core::job::{Job, JobSpec};

        let mut job = Job::new(7, JobSpec::default());
        assert!(job_to_proto(&job).last_sched_eval.is_none());

        job.last_sched_eval = Some(job.submit_time);
        assert!(job_to_proto(&job).last_sched_eval.is_some());
    }

    #[test]
    fn requested_tres_summarizes_cpu_memory_nodes_and_gpus() {
        use spur_core::job::JobSpec;

        let spec = JobSpec {
            num_nodes: 2,
            num_tasks: 2,
            cpus_per_task: 8,
            memory_per_node_mb: Some(16000),
            gres: vec!["gpu:4".into()],
            ..Default::default()
        };
        assert_eq!(requested_tres(&spec), "cpu=16,mem=32000M,node=2,gres/gpu=8");
        assert_eq!(min_cpus_per_node(&spec), 8);
        assert_eq!(min_memory_per_node_mb(&spec), 16000);
    }

    #[test]
    fn requested_tres_cpu_total_is_the_real_ask_not_the_rounded_per_node_minimum() {
        use spur_core::job::JobSpec;

        // -N2 -n7 -c1: the per-node minimum rounds up to 4, but the job asked
        // for 7 CPUs, and that is what ReqTRES must report.
        let spec = JobSpec {
            num_nodes: 2,
            num_tasks: 7,
            cpus_per_task: 1,
            ..Default::default()
        };
        assert_eq!(requested_tres(&spec), "cpu=7,node=2");
        assert_eq!(min_cpus_per_node(&spec), 4);
    }

    #[test]
    fn requested_tres_saturates_instead_of_overflowing() {
        use spur_core::job::JobSpec;

        // Nothing bounds these at submit, and they feed every GetJobs.
        let spec = JobSpec {
            num_nodes: 1,
            num_tasks: u32::MAX,
            cpus_per_task: u32::MAX,
            memory_per_cpu_mb: Some(u64::MAX),
            ..Default::default()
        };
        assert_eq!(
            requested_tres(&spec),
            format!("cpu=18446744065119617025,mem={}M,node=1", u64::MAX)
        );
        assert_eq!(min_memory_per_node_mb(&spec), u64::MAX);
    }

    #[test]
    fn requested_tres_omits_memory_when_none_requested() {
        use spur_core::job::JobSpec;

        let spec = JobSpec {
            num_nodes: 1,
            num_tasks: 1,
            cpus_per_task: 1,
            ..Default::default()
        };
        assert_eq!(requested_tres(&spec), "cpu=1,node=1");
        assert_eq!(min_memory_per_node_mb(&spec), 0);
    }

    #[test]
    fn mem_per_cpu_is_reported_verbatim_under_the_slurm_cpu_label() {
        use spur_core::job::{Job, JobSpec};

        // Slurm prints MinMemoryCPU=1000M here; converting to a per-node
        // figure would report a request the user never made.
        let job = Job::new(
            1,
            JobSpec {
                num_nodes: 1,
                num_tasks: 1,
                cpus_per_task: 4,
                memory_per_cpu_mb: Some(1000),
                ..Default::default()
            },
        );
        let info = job_to_proto(&job);
        assert_eq!(info.min_memory_node_mb, 1000);
        assert!(info.min_memory_is_per_cpu);

        // A per-node request keeps the node label and its own value.
        let job = Job::new(
            2,
            JobSpec {
                memory_per_node_mb: Some(16000),
                ..Default::default()
            },
        );
        let info = job_to_proto(&job);
        assert_eq!(info.min_memory_node_mb, 16000);
        assert!(!info.min_memory_is_per_cpu);
    }

    #[test]
    fn min_memory_per_node_scales_a_mem_per_cpu_request() {
        use spur_core::job::JobSpec;

        let spec = JobSpec {
            num_nodes: 1,
            num_tasks: 4,
            cpus_per_task: 2,
            memory_per_cpu_mb: Some(1000),
            ..Default::default()
        };
        assert_eq!(min_cpus_per_node(&spec), 8);
        assert_eq!(min_memory_per_node_mb(&spec), 8000);
    }

    #[test]
    fn job_to_proto_output_path_prefers_actual_else_absolute_computed() {
        use spur_core::job::{Job, JobSpec};

        let mut job = Job::new(
            42,
            JobSpec {
                work_dir: "/home/alice".into(),
                ..Default::default()
            },
        );

        // Unset: absolute computed fallback.
        let info = job_to_proto(&job);
        assert_eq!(info.stdout_path, "/home/alice/spur-42.out");
        assert_eq!(info.stderr_path, "/home/alice/spur-42.out");

        // Set: reported path wins.
        job.actual_stdout_path = Some("/tmp/spur-42.out".into());
        job.actual_stderr_path = Some("/tmp/spur-42.out".into());
        let info = job_to_proto(&job);
        assert_eq!(info.stdout_path, "/tmp/spur-42.out");
        assert_eq!(info.stderr_path, "/tmp/spur-42.out");
    }

    #[test]
    fn resolve_max_nodes_update_maps_intents() {
        // `--max-nodes 0` (Some(0), flag unset) must resolve to a clear, not a
        // silent no-op: cluster.rs only clears when the passed bool is true.
        assert_eq!(resolve_max_nodes_update(Some(0), false), (None, true));
        // Explicit clear flag, regardless of value.
        assert_eq!(resolve_max_nodes_update(None, true), (None, true));
        assert_eq!(resolve_max_nodes_update(Some(4), true), (None, true));
        // A real positive cap is preserved and does not clear.
        assert_eq!(resolve_max_nodes_update(Some(4), false), (Some(4), false));
        // No value and no flag means "leave unchanged".
        assert_eq!(resolve_max_nodes_update(None, false), (None, false));
    }

    fn make_node_info(name: &str) -> NodeInfo {
        NodeInfo {
            name: name.into(),
            ..Default::default()
        }
    }

    fn make_reservation(
        name: &str,
        nodes: &[&str],
        start_offset_hours: i64,
        end_offset_hours: i64,
    ) -> Reservation {
        let now = Utc::now();
        Reservation {
            name: name.into(),
            start_time: now + Duration::hours(start_offset_hours),
            end_time: now + Duration::hours(end_offset_hours),
            nodes: nodes.iter().map(|s| s.to_string()).collect(),
            accounts: Vec::new(),
            users: Vec::new(),
            flags: Default::default(),
            owner: String::new(),
        }
    }

    fn core_node(name: &str, partition: &str) -> spur_core::node::Node {
        let mut n =
            spur_core::node::Node::new(name.into(), spur_core::resource::ResourceSet::default());
        n.partitions = vec![partition.into()];
        n
    }

    #[test]
    fn node_info_includes_available_features() {
        let mut node = core_node("gpu-node1", "gpu");
        node.features = vec!["mi350x".into(), "atl".into()];

        let info = node_to_proto(&node);

        assert_eq!(info.features, ["mi350x", "atl"]);
    }

    fn names(pattern: &str) -> Option<HashSet<String>> {
        let pattern = pattern.trim();
        (!pattern.is_empty()).then(|| {
            spur_sched::node_match::expand_hostlist_or_split(pattern)
                .into_iter()
                .collect()
        })
    }

    #[test]
    fn test_node_filter_empty_matches_all() {
        let n = core_node("gpu01", "batch");
        assert!(node_matches_filter(&n, names("").as_ref(), ""));
        assert!(node_matches_filter(&n, names("  ").as_ref(), "  "));
    }

    #[test]
    fn test_node_filter_by_name() {
        let n = core_node("gpu01", "batch");
        assert!(node_matches_filter(&n, names("gpu01").as_ref(), ""));
        assert!(!node_matches_filter(&n, names("gpu02").as_ref(), ""));
    }

    #[test]
    fn test_node_filter_hostlist_bracket() {
        let n = core_node("gpu03", "batch");
        assert!(node_matches_filter(&n, names("gpu[01-04]").as_ref(), ""));
        assert!(!node_matches_filter(&n, names("gpu[05-08]").as_ref(), ""));
    }

    #[test]
    fn test_node_filter_by_partition() {
        let n = core_node("gpu01", "batch");
        assert!(node_matches_filter(&n, names("").as_ref(), "batch"));
        assert!(!node_matches_filter(&n, names("").as_ref(), "debug"));
    }

    #[test]
    fn test_node_filter_name_and_partition_both_apply() {
        let n = core_node("gpu01", "batch");
        // Right name, wrong partition -> excluded.
        assert!(!node_matches_filter(&n, names("gpu01").as_ref(), "debug"));
        assert!(node_matches_filter(&n, names("gpu01").as_ref(), "batch"));
    }

    #[test]
    fn test_node_filter_partition_comma_list() {
        let n = core_node("gpu01", "gpu");
        assert!(node_matches_filter(&n, names("").as_ref(), "gpu,cpu"));
        assert!(!node_matches_filter(&n, names("").as_ref(), "cpu,fpga"));
    }

    #[test]
    fn test_annotate_no_reservations() {
        let mut nodes = vec![make_node_info("n1"), make_node_info("n2")];
        annotate_nodes_with_reservations(&mut nodes, &[], Utc::now());
        assert!(nodes[0].active_reservation.is_empty());
        assert!(nodes[1].active_reservation.is_empty());
    }

    #[test]
    fn test_annotate_active_reservation() {
        let mut nodes = vec![make_node_info("n1"), make_node_info("n2")];
        let reservations = vec![make_reservation("maint", &["n1"], -1, 1)];
        annotate_nodes_with_reservations(&mut nodes, &reservations, Utc::now());
        assert_eq!(nodes[0].active_reservation, "maint");
        assert!(nodes[1].active_reservation.is_empty());
    }

    #[test]
    fn test_annotate_expired_reservation() {
        let mut nodes = vec![make_node_info("n1")];
        let reservations = vec![make_reservation("old", &["n1"], -3, -1)];
        annotate_nodes_with_reservations(&mut nodes, &reservations, Utc::now());
        assert!(nodes[0].active_reservation.is_empty());
    }

    #[test]
    fn test_annotate_future_reservation() {
        let mut nodes = vec![make_node_info("n1")];
        let reservations = vec![make_reservation("future", &["n1"], 1, 3)];
        annotate_nodes_with_reservations(&mut nodes, &reservations, Utc::now());
        assert!(nodes[0].active_reservation.is_empty());
    }

    #[test]
    fn test_annotate_partial_coverage() {
        let mut nodes = vec![
            make_node_info("n1"),
            make_node_info("n2"),
            make_node_info("n3"),
        ];
        let reservations = vec![make_reservation("gpu-resv", &["n1", "n3"], -1, 1)];
        annotate_nodes_with_reservations(&mut nodes, &reservations, Utc::now());
        assert_eq!(nodes[0].active_reservation, "gpu-resv");
        assert!(nodes[1].active_reservation.is_empty());
        assert_eq!(nodes[2].active_reservation, "gpu-resv");
    }

    #[test]
    fn test_annotate_multiple_reservations_first_wins() {
        let mut nodes = vec![make_node_info("n1")];
        let reservations = vec![
            make_reservation("first", &["n1"], -1, 1),
            make_reservation("second", &["n1"], -1, 1),
        ];
        annotate_nodes_with_reservations(&mut nodes, &reservations, Utc::now());
        assert_eq!(nodes[0].active_reservation, "first");
    }

    #[test]
    fn test_annotate_maint_flag_with_overlapping_reservations() {
        let mut nodes = vec![make_node_info("n1")];
        let now = Utc::now();
        let plain = Reservation {
            name: "plain".into(),
            start_time: now - Duration::hours(1),
            end_time: now + Duration::hours(1),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: Vec::new(),
            flags: ReservationFlags {
                overlap: true,
                ..Default::default()
            },
            owner: String::new(),
        };
        let maint = Reservation {
            name: "maint".into(),
            start_time: now - Duration::hours(1),
            end_time: now + Duration::hours(1),
            nodes: vec!["n1".into()],
            accounts: Vec::new(),
            users: Vec::new(),
            flags: ReservationFlags {
                maint: true,
                overlap: true,
                ..Default::default()
            },
            owner: String::new(),
        };
        let reservations = vec![plain, maint];
        annotate_nodes_with_reservations(&mut nodes, &reservations, Utc::now());
        assert_eq!(nodes[0].active_reservation, "plain");
        assert!(nodes[0].reservation_maint);
    }

    #[test]
    fn node_complete_error_status_mapping_covers_all_variants() {
        let cases: Vec<(NodeCompleteError, Code, bool)> = vec![
            (
                NodeCompleteError::JobNotFound { job_id: 1 },
                Code::NotFound,
                false,
            ),
            (
                NodeCompleteError::NodeNotAllocated {
                    job_id: 1,
                    node: "n1".into(),
                },
                Code::InvalidArgument,
                false,
            ),
            (
                NodeCompleteError::RaftPropose {
                    source: anyhow::anyhow!("test"),
                },
                Code::Unavailable,
                true,
            ),
        ];

        for (err, want_code, want_retry) in cases {
            assert_eq!(err.retryable(), want_retry, "{err:?}");
            let retry = err.retryable();
            let status = node_complete_to_status(err);
            assert_eq!(status.code(), want_code);
            let agent_retryable = spur_proto::controller_rpc_retryable(&status);
            assert_eq!(retry, agent_retryable, "{status:?}");
        }
    }

    #[test]
    fn completion_report_state_accepts_completed_zero() {
        assert!(validate_completion_report_state_for_rpc(JobState::Completed, 0).is_ok());
    }

    #[test]
    fn completion_report_state_accepts_failed_nonzero() {
        assert!(validate_completion_report_state_for_rpc(JobState::Failed, 42).is_ok());
    }

    // A signaled job is reported as (Completed, exit_code=0); the validator must
    // accept it (controller rederives Failed from the signal). See agent_server.rs.
    #[test]
    fn completion_report_state_accepts_signaled_completed_zero() {
        assert!(validate_completion_report_state_for_rpc(JobState::Completed, 0).is_ok());
    }

    #[test]
    fn completion_report_state_rejects_completed_nonzero() {
        let err = validate_completion_report_state_for_rpc(JobState::Completed, 1).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("does not match exit_code"));
    }

    #[test]
    fn completion_report_state_rejects_cancelled() {
        let err = validate_completion_report_state_for_rpc(JobState::Cancelled, 0).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("invalid completion state"));
    }

    #[test]
    fn completion_report_state_rejects_completing() {
        let err = validate_completion_report_state_for_rpc(JobState::Completing, 0).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("invalid completion state"));
    }

    #[test]
    fn completion_report_state_rejects_running() {
        let err = validate_completion_report_state_for_rpc(JobState::Running, 0).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("invalid completion state"));
    }

    #[test]
    fn requested_gpus_detail_per_task() {
        use spur_core::gpu_request::GpuRequest;
        use spur_core::job::JobSpec;

        let spec = JobSpec {
            num_nodes: 2,
            num_tasks: 4,
            cpus_per_task: 1,
            gpus_per_task: Some(GpuRequest::new(2, None)),
            ..Default::default()
        };
        assert_eq!(requested_gpus_detail(&spec), "gpu:2/task");
    }

    #[test]
    fn requested_gpus_detail_per_task_typed() {
        use spur_core::gpu_request::GpuRequest;
        use spur_core::job::JobSpec;

        let spec = JobSpec {
            num_nodes: 2,
            num_tasks: 4,
            cpus_per_task: 1,
            gpus_per_task: Some(GpuRequest::new(4, Some("mi300x".into()))),
            ..Default::default()
        };
        assert_eq!(requested_gpus_detail(&spec), "gpu:mi300x:4/task");
    }

    #[test]
    fn requested_gpus_detail_total() {
        use spur_core::gpu_request::GpuRequest;
        use spur_core::job::JobSpec;

        let spec = JobSpec {
            num_nodes: 2,
            num_tasks: 2,
            cpus_per_task: 1,
            gpus: Some(GpuRequest::new(8, None)),
            ..Default::default()
        };
        assert_eq!(requested_gpus_detail(&spec), "gpu:8");
    }

    #[test]
    fn requested_gpus_detail_per_node() {
        use spur_core::gpu_request::GpuRequest;
        use spur_core::job::JobSpec;

        let spec = JobSpec {
            num_nodes: 2,
            num_tasks: 2,
            cpus_per_task: 1,
            gpus_per_node: Some(GpuRequest::new(4, None)),
            ..Default::default()
        };
        assert_eq!(requested_gpus_detail(&spec), "gpu:4/node");
    }

    #[test]
    fn proto_to_job_spec_defaults_absent_ntasks_to_nodes() {
        // C1: num_tasks=0 (unset over the wire) defaults to one task per node,
        // not 1, so the node count is not silently collapsed.
        let spec = spur_proto::proto::JobSpec {
            num_nodes: 4,
            num_tasks: 0,
            ..Default::default()
        };
        let core = proto_to_job_spec(spec).unwrap();
        assert_eq!(core.num_tasks, 4);
        assert_eq!(core.effective_num_nodes(), 4);
    }

    #[test]
    fn proto_to_job_spec_respects_explicit_single_task() {
        // An explicit --ntasks=1 is preserved and reduces the job to one node.
        let spec = spur_proto::proto::JobSpec {
            num_nodes: 4,
            num_tasks: 1,
            ..Default::default()
        };
        let core = proto_to_job_spec(spec).unwrap();
        assert_eq!(core.num_tasks, 1);
        assert_eq!(core.effective_num_nodes(), 1);
    }

    #[test]
    fn proto_to_job_spec_defers_gpu_validation() {
        // C3: proto_to_job_spec no longer rejects -N4 -n1 --gpus=2. GPU demand
        // is validated in submit_job against the normalized (1) node count.
        let spec = spur_proto::proto::JobSpec {
            num_nodes: 4,
            num_tasks: 1,
            gpus: Some(spur_proto::proto::GpuRequest {
                count: 2,
                gpu_type: String::new(),
            }),
            ..Default::default()
        };
        assert!(proto_to_job_spec(spec).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn list_reservations_filters_by_name() {
        use spur_core::resource::ResourceSet;
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;

        for n in ["n1", "n2"] {
            svc.cluster
                .register_node(
                    n.into(),
                    n.into(),
                    ResourceSet {
                        cpus: 4,
                        memory_mb: 8000,
                        ..Default::default()
                    },
                    "127.0.0.1".into(),
                    6818,
                    String::new(),
                    String::new(),
                    spur_core::node::NodeSource::NativeHost,
                    std::collections::HashMap::new(),
                    true,
                )
                .unwrap();
        }
        for _ in 0..200 {
            if svc.cluster.get_node("n1").is_some() && svc.cluster.get_node("n2").is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        for (name, node) in [("rocm_patch", "n1"), ("other_resv", "n2")] {
            svc.create_reservation(admin_request(CreateReservationRequest {
                name: name.into(),
                start_time: "now".into(),
                duration_minutes: 60,
                nodes: vec![node.into()],
                accounts: Vec::new(),
                users: Vec::new(),
                flags: Vec::new(),
                user: String::new(),
            }))
            .await
            .unwrap();
        }

        let one = svc
            .list_reservations(Request::new(ListReservationsRequest {
                name: "rocm_patch".into(),
            }))
            .await
            .unwrap()
            .into_inner()
            .reservations;
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].name, "rocm_patch");

        let all = svc
            .list_reservations(Request::new(ListReservationsRequest {
                name: String::new(),
            }))
            .await
            .unwrap()
            .into_inner()
            .reservations;
        assert_eq!(all.len(), 2);

        // A whitespace-only name is trimmed to empty and treated as no filter.
        let blank = svc
            .list_reservations(Request::new(ListReservationsRequest { name: "   ".into() }))
            .await
            .unwrap()
            .into_inner()
            .reservations;
        assert_eq!(blank.len(), 2);

        let none = svc
            .list_reservations(Request::new(ListReservationsRequest {
                name: "does_not_exist".into(),
            }))
            .await
            .unwrap()
            .into_inner()
            .reservations;
        assert!(none.is_empty());
    }

    // --- control-plane mutation authorization ---
    //
    // Every RPC that defines cluster tenancy (partitions, node placement/labels, admission tokens,
    // reservations) must reject an identified non-admin. An anonymous caller (no identity —
    // `disabled`, or `permissive` with no credential) passes the gate, matching the auth model's
    // non-enforcing modes; the gate binds real users in `required` mode. Enumerated as a table so the
    // next handler added cannot quietly reopen the boundary to an identified non-admin.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn control_plane_mutations_deny_identified_non_admin() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;

        macro_rules! assert_admin_gated {
            ($method:ident, $req:expr) => {{
                // A verified but non-admin identity is rejected.
                let mut r = Request::new($req);
                r.extensions_mut().insert(viewer("mallory", false));
                let err = svc.$method(r).await.expect_err(concat!(
                    stringify!($method),
                    " must reject a non-admin caller"
                ));
                assert_eq!(
                    err.code(),
                    Code::PermissionDenied,
                    concat!(stringify!($method), " (non-admin) must be PermissionDenied")
                );

                // No verified identity (disabled, or permissive with no credential) passes the gate,
                // keeping the non-enforcing modes working; the call then proceeds and may fail for
                // unrelated reasons, but must never be rejected as PermissionDenied by the gate.
                if let Err(e) = svc.$method(Request::new($req)).await {
                    assert_ne!(
                        e.code(),
                        Code::PermissionDenied,
                        concat!(stringify!($method), " (anonymous) must not be gate-rejected")
                    );
                }
            }};
        }

        assert_admin_gated!(
            create_partition,
            CreatePartitionRequest {
                name: "pwn".into(),
                nodes: "ALL".into(),
                ..Default::default()
            }
        );
        assert_admin_gated!(
            update_partition,
            UpdatePartitionRequest {
                name: "tenant-b".into(),
                ..Default::default()
            }
        );
        assert_admin_gated!(
            delete_partition,
            DeletePartitionRequest {
                name: "tenant-b".into(),
            }
        );
        assert_admin_gated!(
            update_node,
            UpdateNodeRequest {
                name: "victim-node".into(),
                ..Default::default()
            }
        );
        assert_admin_gated!(reconfigure, ());
        assert_admin_gated!(create_token, CreateTokenRequest::default());
        assert_admin_gated!(list_tokens, ListTokensRequest::default());
        assert_admin_gated!(
            revoke_token,
            RevokeTokenRequest {
                token_id: "t".into(),
            }
        );
        // Reservations use the stricter `require_reservation_manager` instead — see
        // `reservation_mutations_deny_non_operators`.
    }

    /// Reservations also reject an anonymous caller, unlike the generic gate: the asserted name is
    /// still checkable, and a client skipping the CLI is the gap being closed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reservation_mutations_deny_non_operators() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;

        // A name no host resolves, so the outcome cannot turn on the runner's user database.
        const UNKNOWN: &str = "no_such_user_in_nss_7f3a";

        macro_rules! assert_operator_gated {
            ($method:ident, $req:expr) => {{
                // Verified non-admin, anonymous asserting a name, and anonymous asserting nothing —
                // the last being what the retired ownership rule took for a trusted internal call.
                for (identity, user) in [
                    (Some(viewer(UNKNOWN, false)), UNKNOWN),
                    (None, UNKNOWN),
                    (None, ""),
                ] {
                    let mut msg = $req;
                    msg.user = user.to_string();
                    let mut r = Request::new(msg);
                    if let Some(id) = identity {
                        r.extensions_mut().insert(id);
                    }
                    let err = svc.$method(r).await.expect_err(concat!(
                        stringify!($method),
                        " must reject a non-operator caller"
                    ));
                    assert_eq!(
                        err.code(),
                        Code::PermissionDenied,
                        "{} must be PermissionDenied for user {user:?}",
                        stringify!($method)
                    );
                }
            }};
        }

        assert_operator_gated!(
            create_reservation,
            CreateReservationRequest {
                name: "r".into(),
                ..Default::default()
            }
        );
        assert_operator_gated!(
            update_reservation,
            UpdateReservationRequest {
                name: "r".into(),
                ..Default::default()
            }
        );
        assert_operator_gated!(
            delete_reservation,
            DeleteReservationRequest {
                name: "r".into(),
                ..Default::default()
            }
        );

        assert!(
            svc.cluster.get_reservations().is_empty(),
            "a denied create must not reach the state machine"
        );
    }

    /// An admin credential must work on a controller that shares no user directory with the login
    /// nodes. The name here resolves nowhere, so only the pre-lookup admin return can allow it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admin_manages_reservations_without_a_resolvable_name() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        register_plain_node(&svc, "node-a", 6822).await;

        let mut req = Request::new(CreateReservationRequest {
            name: "maint".into(),
            start_time: "now".into(),
            duration_minutes: 60,
            nodes: vec!["node-a".into()],
            ..Default::default()
        });
        req.extensions_mut()
            .insert(viewer("no_such_user_in_nss_7f3a", true));

        svc.create_reservation(req)
            .await
            .expect("an admin credential must not depend on an NSS lookup");
        assert_eq!(svc.cluster.get_reservations().len(), 1);
    }

    #[test]
    fn privileged_user_manages_reservations() {
        assert!(reservation_manager_ruling("alice", Ok(true)).is_ok());
    }

    #[test]
    fn unprivileged_user_may_not_manage_reservations() {
        let err = reservation_manager_ruling("alice", Ok(false))
            .expect_err("an unprivileged user must be denied");
        assert_eq!(err.code(), Code::PermissionDenied);
        assert!(
            err.message().contains("alice") && err.message().contains("sudo"),
            "denial must name the user and the requirement: {}",
            err.message()
        );
    }

    /// An unresolvable name must deny rather than fall through to an allow.
    #[test]
    fn unresolvable_user_is_denied() {
        let err = reservation_manager_ruling(
            "ghost",
            Err(spur_core::auth::AuthError::UnknownUser("ghost".into())),
        )
        .expect_err("an unresolvable user must be denied");
        assert_eq!(err.code(), Code::PermissionDenied);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn control_plane_mutation_allows_admin() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        register_plain_node(&svc, "node-a", 6820).await;

        // An admin identity passes the gate and the partition is actually created.
        svc.create_partition(admin_request(CreatePartitionRequest {
            name: "team-a".into(),
            nodes: "node-a".into(),
            ..Default::default()
        }))
        .await
        .expect("an admin caller may create a partition");

        assert!(
            svc.cluster
                .get_partitions()
                .iter()
                .any(|p| p.name == "team-a"),
            "the admin's partition must exist after the gated call succeeds"
        );

        // A structurally different RPC (token issuance) also passes the admin gate.
        let token_resp = svc
            .create_token(admin_request(CreateTokenRequest::default()))
            .await
            .expect("an admin caller may create a token");
        assert!(
            !token_resp.into_inner().token_id.is_empty(),
            "the admin's token must be issued"
        );
    }

    // --- register_agent relabeling gate ---
    //
    // A node re-registering with different labels reaches the same NodeLabelsUpdate WAL op as the
    // gated `update_node`, so it must be admin-gated too; first registration and an unchanged
    // re-registration must not be, so a node agent can still self-register with no identity.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn register_agent_first_registration_needs_no_admin() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;

        svc.register_agent(Request::new(RegisterAgentRequest {
            hostname: "fresh-node".into(),
            address: "127.0.0.1".into(),
            labels: [("pool".to_string(), "a".to_string())].into(),
            ..Default::default()
        }))
        .await
        .expect("first registration must not require an admin identity");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn register_agent_relabel_of_existing_node_is_admin_gated() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        register_plain_node(&svc, "victim-node", 6821).await;

        let relabel = || RegisterAgentRequest {
            hostname: "victim-node".into(),
            address: "127.0.0.1".into(),
            labels: [("pool".to_string(), "stolen".to_string())].into(),
            ..Default::default()
        };

        let mut r = Request::new(relabel());
        r.extensions_mut().insert(viewer("mallory", false));
        let err = svc
            .register_agent(r)
            .await
            .expect_err("a non-admin caller must not be able to relabel an existing node");
        assert_eq!(err.code(), Code::PermissionDenied);
        assert!(
            svc.cluster
                .get_node("victim-node")
                .unwrap()
                .labels
                .is_empty(),
            "the rejected relabel must not have applied"
        );

        svc.register_agent(Request::new(RegisterAgentRequest {
            hostname: "victim-node".into(),
            address: "127.0.0.1".into(),
            labels: std::collections::HashMap::new(),
            ..Default::default()
        }))
        .await
        .expect("an unchanged re-registration must not be admin-gated");

        svc.register_agent(admin_request(relabel()))
            .await
            .expect("an admin caller may relabel an existing node");
        assert_eq!(
            svc.cluster.get_node("victim-node").unwrap().labels["pool"],
            "stolen"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn register_agent_unchanged_non_empty_labels_needs_no_admin() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        register_plain_node(&svc, "steady-node", 6822).await;

        let labels: std::collections::HashMap<String, String> =
            [("pool".to_string(), "prod".to_string())].into();
        svc.register_agent(admin_request(RegisterAgentRequest {
            hostname: "steady-node".into(),
            address: "127.0.0.1".into(),
            labels: labels.clone(),
            ..Default::default()
        }))
        .await
        .expect("an admin may set the initial labels");

        svc.register_agent(Request::new(RegisterAgentRequest {
            hostname: "steady-node".into(),
            address: "127.0.0.1".into(),
            labels,
            ..Default::default()
        }))
        .await
        .expect("re-registering with the same non-empty labels must not be admin-gated");
    }

    // The diff a non-admin's re-registration is checked against is read at write time, not decided
    // ahead of it — so a non-admin can't revert a relabel that lands between their read and write.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn register_agent_non_admin_cannot_revert_a_privileged_relabel() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = test_service(&dir).await;
        register_plain_node(&svc, "contended-node", 6823).await;

        svc.register_agent(admin_request(RegisterAgentRequest {
            hostname: "contended-node".into(),
            address: "127.0.0.1".into(),
            labels: [("pool".to_string(), "prod".to_string())].into(),
            ..Default::default()
        }))
        .await
        .expect("an admin may relabel the node");

        let mut r = Request::new(RegisterAgentRequest {
            hostname: "contended-node".into(),
            address: "127.0.0.1".into(),
            labels: std::collections::HashMap::new(),
            ..Default::default()
        });
        r.extensions_mut().insert(viewer("mallory", false));
        let err = svc
            .register_agent(r)
            .await
            .expect_err("a non-admin re-registration must not revert the privileged relabel");
        assert_eq!(err.code(), Code::PermissionDenied);
        assert_eq!(
            svc.cluster.get_node("contended-node").unwrap().labels["pool"],
            "prod",
            "the privileged relabel must survive the non-admin's attempt"
        );
    }

    // Runs the admin and non-admin relabel concurrently (not sequentially) so the gate's decision
    // is genuinely raced against the write, not just checked against an already-settled state.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn register_agent_concurrent_non_admin_relabel_never_wins() {
        let dir = tempfile::TempDir::new().unwrap();
        let svc = std::sync::Arc::new(test_service(&dir).await);
        register_plain_node(&svc, "racy-node", 6824).await;

        let admin_svc = svc.clone();
        let admin_task = tokio::spawn(async move {
            admin_svc
                .register_agent(admin_request(RegisterAgentRequest {
                    hostname: "racy-node".into(),
                    address: "127.0.0.1".into(),
                    labels: [("pool".to_string(), "prod".to_string())].into(),
                    ..Default::default()
                }))
                .await
        });

        let non_admin_svc = svc.clone();
        let non_admin_task = tokio::spawn(async move {
            let mut r = Request::new(RegisterAgentRequest {
                hostname: "racy-node".into(),
                address: "127.0.0.1".into(),
                labels: [("pool".to_string(), "hacked".to_string())].into(),
                ..Default::default()
            });
            r.extensions_mut().insert(viewer("mallory", false));
            non_admin_svc.register_agent(r).await
        });

        let (admin_result, non_admin_result) = tokio::join!(admin_task, non_admin_task);
        admin_result
            .unwrap()
            .expect("the admin's relabel must succeed regardless of scheduling order");
        let non_admin_err = non_admin_result
            .unwrap()
            .expect_err("the non-admin's concurrent relabel must never win the race");
        assert_eq!(non_admin_err.code(), Code::PermissionDenied);
        assert_eq!(
            svc.cluster.get_node("racy-node").unwrap().labels["pool"],
            "prod",
            "only the admin's labels may land, whichever task the scheduler ran first"
        );
    }

    // --- authoritative_user ---

    #[test]
    fn authoritative_user_leaves_field_alone_when_no_identity() {
        let mut user = "alice".to_string();
        ControllerService::authoritative_user(&mut user, None);
        assert_eq!(user, "alice");
    }

    #[test]
    fn authoritative_user_overwrites_with_authenticated_user() {
        let mut user = "alice".to_string();
        let id = spur_core::auth::Identity {
            user: "bob".to_string(),
            uid: 1001,
            gid: 1001,
            is_admin: false,
        };
        ControllerService::authoritative_user(&mut user, Some(&id));
        assert_eq!(user, "bob");
    }

    #[test]
    fn authoritative_user_fills_an_empty_field() {
        let mut user = String::new();
        let id = spur_core::auth::Identity {
            user: "carol".to_string(),
            uid: 1002,
            gid: 1002,
            is_admin: false,
        };
        ControllerService::authoritative_user(&mut user, Some(&id));
        assert_eq!(user, "carol");
    }

    // --- assoc_mgr_scope_user (assoc-mgr read authorization) ---

    fn ident(user: &str) -> spur_core::auth::Identity {
        spur_core::auth::Identity {
            user: user.to_string(),
            uid: 1001,
            gid: 1001,
            is_admin: false,
        }
    }

    #[test]
    fn assoc_mgr_scope_user_unauthenticated_honors_the_request() {
        // permissive/disabled: no identity is privileged, so the request stands —
        // empty means every user, a name means that user.
        assert_eq!(assoc_mgr_scope_user(None, "", false), None);
        assert_eq!(
            assoc_mgr_scope_user(None, "alice", false).as_deref(),
            Some("alice")
        );
    }

    #[test]
    fn assoc_mgr_scope_user_pins_a_non_admin_to_itself() {
        // A non-admin cannot widen (empty request) or retarget (another user) the
        // view: both collapse to their own identity.
        let id = ident("bob");
        assert_eq!(
            assoc_mgr_scope_user(Some(&id), "", false).as_deref(),
            Some("bob")
        );
        assert_eq!(
            assoc_mgr_scope_user(Some(&id), "alice", false).as_deref(),
            Some("bob")
        );
    }

    #[test]
    fn assoc_mgr_scope_user_lets_an_admin_target_anyone() {
        // An admin keeps the request: every user when empty, an arbitrary user
        // when named.
        let id = ident("root");
        assert_eq!(assoc_mgr_scope_user(Some(&id), "", true), None);
        assert_eq!(
            assoc_mgr_scope_user(Some(&id), "alice", true).as_deref(),
            Some("alice")
        );
    }

    // --- build_reservation_txn (audit attribution) ---

    #[test]
    fn build_reservation_txn_records_verified_identity_and_large_uid() {
        let id = spur_core::auth::Identity {
            user: "alice".to_string(),
            uid: 4_000_000_000, // > i32::MAX: must survive as i64, not wrap negative
            gid: 0,
            is_admin: false,
        };
        let rec = ControllerService::build_reservation_txn(
            TxnAction::Create,
            "resv1",
            "alice",
            Some(&id),
            serde_json::json!({}),
            &Ok(()),
        );
        assert!(rec.verified);
        assert_eq!(rec.actor_uid, Some(4_000_000_000));
        assert_eq!(rec.actor, "alice");
        assert_eq!(rec.source, TxnSource::Api);
        assert_eq!(rec.entity_type, TxnEntity::Reservation);
        assert_eq!(rec.outcome, crate::accounting::TxnOutcome::Success);
    }

    #[test]
    fn build_reservation_txn_anonymous_is_unverified_and_captures_error() {
        let rec = ControllerService::build_reservation_txn(
            TxnAction::Delete,
            "resv1",
            "bob",
            None,
            serde_json::json!({}),
            &Err(Status::permission_denied(
                "user 'bob' cannot delete reservation",
            )),
        );
        assert!(!rec.verified);
        assert_eq!(rec.actor_uid, None);
        assert_eq!(rec.outcome, crate::accounting::TxnOutcome::Denied);
        assert!(rec.details.contains("cannot delete"));
    }

    // --- bind_spec_to_identity ---

    fn spec_for(user: &str, uid: u32, gid: u32) -> spur_core::job::JobSpec {
        spur_core::job::JobSpec {
            user: user.to_string(),
            uid,
            gid,
            ..Default::default()
        }
    }

    #[test]
    fn bind_spec_leaves_spec_alone_when_no_identity() {
        let mut spec = spec_for("alice", 9999, 9999);
        ControllerService::bind_spec_to_identity(&mut spec, None).unwrap();
        assert_eq!(spec.user, "alice");
        assert_eq!(spec.uid, 9999);
        assert_eq!(spec.gid, 9999);
    }

    #[test]
    fn bind_spec_overwrites_user_uid_gid_from_nss() {
        // Uses "root" — always present in /etc/passwd so NSS resolves it everywhere.
        let mut spec = spec_for("impersonated", 9999, 9999);
        let id = spur_core::auth::Identity {
            user: "root".to_string(),
            uid: 9999,
            gid: 9999,
            is_admin: true,
        };
        ControllerService::bind_spec_to_identity(&mut spec, Some(&id)).unwrap();
        assert_eq!(spec.user, "root");
        // uid/gid must come from NSS, not from the wire spec or the token's uid field.
        assert_eq!(spec.uid, 0, "root's uid is 0 per NSS");
        assert_eq!(spec.gid, 0, "root's gid is 0 per NSS");
    }

    #[test]
    fn bind_spec_fails_closed_for_unknown_user() {
        let mut spec = spec_for("alice", 1000, 1000);
        let id = spur_core::auth::Identity {
            user: "this_user_does_not_exist_in_nss_7f3a".to_string(),
            uid: 0,
            gid: 0,
            is_admin: false,
        };
        let err = ControllerService::bind_spec_to_identity(&mut spec, Some(&id)).unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::FailedPrecondition,
            "unknown user must fail closed, not fall back to wire uid"
        );
        // Spec must not be partially mutated.
        assert_eq!(spec.user, "alice");
        assert_eq!(spec.uid, 1000);
    }
}
