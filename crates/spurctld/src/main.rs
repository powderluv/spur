// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod accounting;
mod agent_client;
mod association_cache;
mod auth_middleware;
mod cluster;
mod cluster_k8s;
mod fairshare_cache;
mod hooks;
mod limits_cache;
mod metrics_proto;
mod metrics_server;
mod pmix_dispatch;
mod raft;
mod raft_server;
mod rest;
mod rpc_middleware;
mod rpc_stats;
mod sched_stats;
mod scheduler_loop;
mod server;

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tracing::info;

use cluster::ClusterManager;
use rpc_stats::RpcStatsCollector;
use sched_stats::SchedStatsCollector;

#[derive(Parser)]
#[command(name = "spurctld", about = "Spur controller daemon (spurctld)")]
struct Args {
    /// Configuration file path
    #[arg(short = 'f', long, default_value = "/etc/spur/spur.conf")]
    config: PathBuf,

    /// gRPC listen address (overrides config file)
    #[arg(long)]
    listen: Option<String>,

    /// State directory
    #[arg(long, default_value = "/var/spool/spur")]
    state_dir: PathBuf,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Foreground mode (don't daemonize)
    #[arg(short = 'D', long)]
    foreground: bool,

    /// Tolerate unreadable/undeserializable Raft WAL entries, votes, or
    /// snapshots during startup recovery by skipping them, instead of
    /// refusing to start. These records represent already-committed cluster
    /// state (nodes, jobs, ...), so a skipped record is silent data loss —
    /// only pass this for deliberate forensic recovery once that loss has
    /// been assessed as acceptable.
    #[arg(long)]
    allow_partial_wal_recovery: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if std::env::args_os()
        .skip(1)
        .any(|a| a == "-V" || a == "--version")
    {
        println!("{}", spur_core::version::version_string());
        return Ok(());
    }

    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.parse().unwrap()),
        )
        .init();

    info!(version = %spur_core::version::version_string(), "spurctld starting");

    // Load config if it exists, otherwise use defaults
    let mut config = if args.config.exists() {
        spur_core::config::SlurmConfig::load_from_file(&args.config)?
    } else {
        info!("no config file found, using defaults");
        spur_core::config::SlurmConfig {
            cluster_name: "spur".into(),
            controller: spur_core::config::ControllerConfig {
                listen_addr: "[::]:6817".into(),
                state_dir: args.state_dir.to_string_lossy().into(),
                ..Default::default()
            },
            ..default_config()
        }
    };

    // Fail fast on a broken submit hook (missing, non-executable, or a Lua that
    // won't compile) instead of deferring the error to the first user submission.
    hooks::validate_submit_hooks(&config.hooks)?;

    // CLI --listen overrides config file; otherwise use config's listen_addr.
    let listen_addr = args
        .listen
        .clone()
        .unwrap_or_else(|| config.controller.listen_addr.clone());

    // Keep config in sync so downstream code sees the final address.
    config.controller.listen_addr = listen_addr.clone();

    // Background update check (non-blocking — does not delay startup)
    spur_update::spawn_startup_check(
        "ROCm/spur",
        env!("CARGO_PKG_VERSION"),
        config.update.check_on_startup,
        config.update.auto_update,
        &config.update.channel,
        &config.update.cache_dir,
        spur_update::SPUR_BINARIES,
    );

    // Initialize cluster manager first so Raft recovery can apply entries.
    // Pass the config path so `scontrol reconfigure` can re-read spur.conf.
    let config_path = if args.config.exists() {
        Some(args.config.clone())
    } else {
        None
    };
    let cluster = Arc::new(ClusterManager::new_with_config_path(
        config.clone(),
        &args.state_dir,
        config_path,
    )?);

    // Raft is always-on. When no peers are configured, run a single-node
    // cluster that self-elects instantly (same pattern as Apache Kudu).
    let (peers, node_id) = if config.controller.peers.is_empty() {
        let raft_addr = config.controller.raft_listen_addr.clone();
        info!("single-node Raft mode (no peers configured)");
        (vec![raft_addr], 1u64)
    } else {
        let hostname = raft::system_hostname()?;
        let (id, source) = raft::resolve_node_id(
            config.controller.node_id,
            &hostname,
            &config.controller.peers,
        )?;
        info!(
            node_id = id,
            source = %source,
            hostname,
            peers = ?config.controller.peers,
            "initializing Raft consensus"
        );
        (config.controller.peers.clone(), id)
    };

    let handle = raft::start_raft_with_recovery_mode(
        node_id,
        &peers,
        &args.state_dir,
        cluster.clone(),
        !args.allow_partial_wal_recovery,
    )
    .await?;
    info!(node_id, "Raft node started");

    let raft_addr: std::net::SocketAddr = config.controller.raft_listen_addr.parse()?;
    let raft_instance = handle.raft.clone();
    tokio::spawn(async move {
        if let Err(e) = raft_server::serve_raft(raft_addr, raft_instance).await {
            tracing::error!(error = %e, "raft internal gRPC server failed");
        }
    });

    let raft_handle = Arc::new(handle);
    cluster.set_raft(raft_handle.raft.clone());

    let sched_stats = Arc::new(SchedStatsCollector::new(config.scheduler.plugin.clone()));
    cluster.set_sched_stats(sched_stats.clone());

    // Accounting stays best-effort so a database outage does not stop scheduling.
    // The connect, migration, notifier, and refresh loops are brought up in a
    // background task that retries until the database is reachable, so a
    // controller that boots with the database down converges on its own — its
    // caches load once the database returns and the fail-closed hold on
    // QOS/account jobs clears without operator action.
    let accounting_service = accounting::start(&config, cluster.clone(), raft_handle.clone());

    // Start scheduler loop (only schedules when this node is Raft leader)
    let sched_cluster = cluster.clone();
    let sched_raft = raft_handle.clone();
    let sched_handle = tokio::spawn(async move {
        scheduler_loop::run(sched_cluster, sched_raft).await;
    });

    // Start the k0s cluster reconcile loop (leader-gated; only when [cluster].enabled).
    if config.cluster.enabled {
        let k8s_cluster = cluster.clone();
        let k8s_raft = raft_handle.clone();
        let k8s_net = cluster_k8s::ClusterNetworking {
            wg_enabled: config.network.wg_enabled,
            mesh_cidr: config.network.wg_cidr.clone(),
            mesh_interface: config.network.wg_interface.clone(),
            pod_cidr: config.cluster.pod_cidr.clone(),
            service_cidr: config.cluster.service_cidr.clone(),
            cni_mtu: config.cluster.cni_mtu,
            cni: config.cluster.cni.clone(),
            control_plane_node: config.cluster.control_plane_node.clone(),
            provisioning_timeout: std::time::Duration::from_secs(
                config.cluster.k8s_provisioning_timeout_secs,
            ),
        };
        tokio::spawn(async move {
            cluster_k8s::run(k8s_cluster, k8s_raft, k8s_net).await;
        });
    }

    // Start node health checker (only on leader).
    const HEALTH_TICK_SECS: u64 = 30;
    let hb_timeout = config.controller.heartbeat_timeout_secs.unwrap_or(90);
    let health_cluster = cluster.clone();
    let health_raft = raft_handle.clone();
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(tokio::time::Duration::from_secs(HEALTH_TICK_SECS));
        // Floored at one tick because `LeadershipGrace::observe` is itself only
        // called once per tick: a grace shorter than that can lapse before the
        // first post-election check even runs, leaving the fleet exposed to a
        // mass DOWN before any agent heartbeat lands.
        let mut grace = cluster::LeadershipGrace::new(std::time::Duration::from_secs(
            hb_timeout.max(HEALTH_TICK_SECS),
        ));
        loop {
            interval.tick().await;
            let is_leader = health_raft.is_leader();
            // Observed before the non-leader bail so a lost term restarts the window.
            let mark_down = grace.observe(is_leader, std::time::Instant::now());
            if !is_leader {
                continue;
            }
            let evicted = health_cluster.check_node_health(hb_timeout, mark_down);
            for fin in &evicted {
                if let Some(job) = health_cluster.get_job(fin.job_id) {
                    let c = health_cluster.clone();
                    tokio::spawn(async move {
                        crate::scheduler_loop::send_cancel_to_agents(&c, &job, 9).await;
                    });
                }
            }
            health_cluster.complete_evicted_steps(&evicted);
        }
    });

    let rpc_stats = Arc::new(RpcStatsCollector::new());

    if config.metrics.enabled {
        let metrics_addr = config
            .metrics
            .effective_listen_addr()
            .map_err(|e| anyhow::anyhow!(e))?;
        let metrics_cluster = cluster.clone();
        let metrics_raft = raft_handle.clone();
        let metrics_rpc_stats = rpc_stats.clone();
        let metrics_sched_stats = sched_stats.clone();
        tokio::spawn(async move {
            if let Err(e) = metrics_server::serve(
                metrics_addr,
                metrics_cluster,
                metrics_raft,
                metrics_rpc_stats,
                metrics_sched_stats,
            )
            .await
            {
                tracing::error!(error = %e, "OpenMetrics metrics server failed");
            }
        });
    }

    if config.rest_api.enabled {
        let rest_addr: std::net::SocketAddr = config.controller.rest_addr.parse()?;
        if !rest_addr.ip().is_loopback() {
            tracing::warn!(
                addr = %rest_addr,
                "REST API is enabled on a non-loopback address and performs NO authentication: \
                 any peer that can reach it can submit and cancel jobs. Restrict it to loopback or \
                 front it with an authenticating proxy."
            );
        }
        let rest_cluster = cluster.clone();
        let rest_raft = raft_handle.clone();
        tokio::spawn(async move {
            if let Err(e) = rest::serve(rest_addr, rest_cluster, rest_raft).await {
                tracing::error!(error = %e, "REST API server failed");
            }
        });
    }

    // Start gRPC server
    let addr: std::net::SocketAddr = listen_addr.parse()?;
    // The controller presents this key as its credential to agents (spurd authenticates callers).
    crate::agent_client::set_signing_key(config.auth.jwt_key.clone().unwrap_or_default());
    crate::agent_client::set_channel_tuning(&config.controller);

    // State the authentication posture explicitly at startup: it determines whether the listening
    // port is the trust boundary or merely the transport.
    match config.auth.mode {
        spur_core::config::AuthMode::Required => {
            info!(
                mode = "required",
                "RPC callers must present a valid credential"
            )
        }
        spur_core::config::AuthMode::Permissive => tracing::warn!(
            mode = "permissive",
            "RPC callers are authenticated WHEN they present a credential, and trusted on their \
             own assertion when they do not. Unauthenticated callers are logged — roll credentials \
             out (`spur token user`), then set [auth] mode = \"required\"."
        ),
        spur_core::config::AuthMode::Disabled => tracing::warn!(
            mode = "disabled",
            "RPC callers are NOT authenticated: the identity used for authorization is supplied by \
             the client. Treat this port as an administrative boundary."
        ),
    }
    if !addr.ip().is_loopback() && config.auth.mode != spur_core::config::AuthMode::Required {
        tracing::warn!(
            %addr,
            "listening on a non-loopback address without required authentication — restrict this \
             port to trusted hosts."
        );
    }
    info!(%addr, "gRPC server listening");
    let jwt_key = server::resolve_startup_jwt_key(&config)?;
    server::serve(
        addr,
        cluster,
        raft_handle,
        rpc_stats,
        sched_stats,
        accounting_service,
        config.cluster.control_plane_replicas,
        jwt_key,
    )
    .await?;

    sched_handle.abort();
    Ok(())
}

fn default_config() -> spur_core::config::SlurmConfig {
    spur_core::config::SlurmConfig {
        cluster_name: "spur".into(),
        controller: Default::default(),
        accounting: Default::default(),
        scheduler: Default::default(),
        auth: Default::default(),
        partitions: vec![spur_core::config::PartitionConfig {
            name: "default".into(),
            default: true,
            state: "UP".into(),
            nodes: "localhost".into(),
            selector: Default::default(),
            max_time: None,
            default_time: None,
            max_nodes: None,
            min_nodes: 1,
            allow_accounts: Vec::new(),
            allow_groups: Vec::new(),
            deny_accounts: Vec::new(),
            deny_qos: Vec::new(),
            allow_qos: Vec::new(),
            priority_tier: 1,
            preempt_mode: String::new(),
            preempt_exempt_time: None,
        }],
        nodes: Vec::new(),
        network: Default::default(),
        logging: Default::default(),
        kubernetes: Default::default(),
        cluster: Default::default(),
        notifications: Default::default(),
        power: Default::default(),
        federation: Default::default(),
        topology: None,
        isolation: Default::default(),
        licenses: std::collections::HashMap::new(),
        burst_buffer: Default::default(),
        update: Default::default(),
        metrics: Default::default(),
        rest_api: Default::default(),
        hooks: Default::default(),
        devices: Default::default(),
        admission: Default::default(),
        rlimits: Default::default(),
        cgroup: Default::default(),
        mpi: Default::default(),
    }
}
