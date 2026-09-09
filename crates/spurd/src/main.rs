// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use clap::parser::ValueSource;
use clap::{CommandFactory, FromArgMatches, Parser};
use tokio::sync::Mutex;
use tracing::{info, warn};

use spur_core::config::{ConfigError, SlurmConfig};
use spur_devices::cdi::cache::CdiCache;
use spur_devices::DeviceRegistry;

use spurd::reporter::NodeReporter;
use spurd::{agent_server, auth_middleware, executor, reporter, stepd};

/// Raise spurd's own `RLIMIT_MEMLOCK` as high as it is allowed to go.
///
/// Kernels before 5.11 charge BPF program memory against this limit instead of the
/// memory cgroup, and the default admits only a handful. On 5.11+ it does not gate.
fn raise_own_memlock_rlimit() {
    let unlimited = libc::rlimit {
        rlim_cur: libc::RLIM_INFINITY,
        rlim_max: libc::RLIM_INFINITY,
    };
    // SAFETY: each call below reads or writes only the local `rlimit` it is
    // handed, which outlives the call.
    if unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &unlimited) } == 0 {
        return;
    }
    // Without CAP_SYS_RESOURCE the hard limit will not move, but the soft one can
    // still be raised to meet it.
    let mut current = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut current) } == 0 {
        let to_hard = libc::rlimit {
            rlim_cur: current.rlim_max,
            rlim_max: current.rlim_max,
        };
        if unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &to_hard) } == 0 {
            return;
        }
    }
    // Visible at the default log level: this changes what jobs can do, and the
    // operator docs promise the agent says so when it cannot raise the limit.
    warn!(
        error = %std::io::Error::last_os_error(),
        "could not raise RLIMIT_MEMLOCK; BPF program loads may fail on kernels before 5.11"
    );
}

fn log_memlock_status(memlock: spur_core::config::MemlockLimit) {
    use spur_core::config::MemlockLimit;
    let configured_desc = match memlock {
        MemlockLimit::Unlimited => "unlimited",
        MemlockLimit::Inherit => "inherit",
        MemlockLimit::Bytes(_) => "bytes",
    };
    let mut current = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut current) };
    let effective = if current.rlim_max == libc::RLIM_INFINITY {
        "unlimited".to_string()
    } else {
        format!("{} bytes", current.rlim_max)
    };
    info!(configured = configured_desc, effective_hard = %effective, "memlock rlimit");
    let is_root = unsafe { libc::geteuid() } == 0;
    if memlock == MemlockLimit::Unlimited && current.rlim_max != libc::RLIM_INFINITY && !is_root {
        warn!(
            effective_hard = %effective,
            "configured memlock=unlimited but process hard limit is finite; \
             jobs will get at most the hard limit unless spurd runs as root"
        );
    }
}

/// Record the enforcement posture at startup: `[cgroup]` decides whether a job is
/// bounded at all, and the agent log is the only place that shows what it booted with.
fn log_cgroup_status(cgroup: &spur_core::config::CgroupConfig) {
    if !cgroup.enabled {
        warn!("[cgroup] enforcement disabled; jobs run with no cpu, memory or device bound");
        return;
    }
    info!(
        required = cgroup.required,
        constrain_cores = cgroup.constrain_cores,
        cpu_quota = cgroup.cpu_quota,
        constrain_ram_space = cgroup.constrain_ram_space,
        allowed_ram_percent = cgroup.allowed_ram_percent,
        constrain_swap = cgroup.constrain_swap,
        allowed_swap_percent = cgroup.allowed_swap_percent,
        min_ram_mb = cgroup.min_ram_mb,
        oom_kill_job = cgroup.oom_kill_job,
        constrain_devices = cgroup.constrain_devices,
        extra_device_paths = ?cgroup.extra_device_paths,
        "cgroup enforcement"
    );
    // An unprivileged agent cannot create the cgroup root, so `required` turns every
    // launch into a refusal. Say so now rather than once per rejected job.
    if cgroup.required && unsafe { libc::geteuid() } != 0 {
        warn!("[cgroup] required = true but spurd is not root; every job launch will be refused");
    }
    if cgroup.constrain_swap && cgroup.allowed_swap_percent == 0 {
        info!(
            "[cgroup] swap denied outright (allowed_swap_percent = 0); a job that outgrows \
             its allocation is OOM-killed rather than paging out"
        );
    }
}

/// Whether a best-effort config load failed only because no file exists at the default
/// path, which is the expected shape for an agent configured entirely by flags.
///
/// Anything else — an explicitly requested path, or a file that is present but malformed,
/// invalid, or unreadable — means settings the operator intended are being ignored, and
/// has to stay visible.
fn absent_optional_config(explicit_path: bool, err: &ConfigError) -> bool {
    !explicit_path && matches!(err, ConfigError::Io(e) if e.kind() == std::io::ErrorKind::NotFound)
}

/// Parse a "key=value" string into a validated label.
fn parse_label(s: &str) -> Result<String, String> {
    if s.contains('=') && s.split('=').next().is_some_and(|k| !k.is_empty()) {
        Ok(s.to_string())
    } else {
        Err(format!("invalid label format '{s}', expected key=value"))
    }
}

#[derive(Parser)]
#[command(name = "spurd", about = "Spur node agent daemon")]
struct Args {
    /// Configuration file path
    #[arg(short = 'f', long, default_value = "/etc/spur/spur.conf")]
    config: std::path::PathBuf,

    /// Controller address
    #[arg(
        long,
        env = "SPUR_CONTROLLER_ADDR",
        default_value = "http://localhost:6817"
    )]
    controller: String,

    /// Agent gRPC listen address
    #[arg(long, default_value = "[::]:6818")]
    listen: String,

    /// Node name (defaults to hostname)
    #[arg(short = 'N', long)]
    hostname: Option<String>,

    /// Advertised comm address (IP or routable hostname) for inter-node reachability.
    /// If not set, auto-detected from WireGuard interface or hostname resolution.
    #[arg(long, env = "SPUR_NODE_ADDRESS")]
    address: Option<String>,

    /// Node labels for partition routing (key=value pairs).
    /// Can be specified multiple times: --label pool=gpu --label rack=a
    #[arg(long = "label", value_parser = parse_label, env = "SPUR_NODE_LABELS")]
    labels: Vec<String>,

    /// Admission join token for token-based node registration.
    #[arg(long = "token", env = "SPUR_JOIN_TOKEN")]
    token: Option<String>,

    /// Foreground mode
    #[arg(short = 'D', long)]
    foreground: bool,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Directory for this agent's own persisted runtime state (stepd sessions).
    /// Defaults to the controller's configured state_dir, then /var/spool/spur.
    #[arg(long, env = "SPUR_STEPD_STATE_DIR")]
    state_dir: Option<String>,
}

/// Parses a stepd directory's `<job_id>.<run_attempt>.<step_id>` basename.
/// Used to fence a session whose descriptor failed to parse — the full
/// identity survives in the directory name even when its contents don't.
fn parse_session_dir_name(path: &std::path::Path) -> Option<(u32, u32, spur_core::step::StepId)> {
    let name = path.file_name()?.to_str()?;
    let mut parts = name.splitn(3, '.');
    let job_id = parts.next()?.parse().ok()?;
    let run_attempt = parts.next()?.parse().ok()?;
    let step_id = parts.next()?.parse().ok()?;
    Some((job_id, run_attempt, step_id))
}

/// The cgroup to reap for a stale session. A descriptor written before its
/// cgroup was recorded still names the job, and the path follows from identity.
fn stale_cgroup_path(descriptor: &stepd::StepdDescriptor) -> std::path::PathBuf {
    if descriptor.cgroup_path.as_os_str().is_empty() {
        executor::expected_cgroup_path(descriptor.job_id, descriptor.run_attempt)
    } else {
        descriptor.cgroup_path.clone()
    }
}

/// True when retrying a runtime-recovery report can never help: the
/// controller has no jwt_key configured, or this node's own identity/token
/// is invalid — none of which a retry loop can fix on its own.
fn is_permanent_recovery_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<tonic::Status>())
        .any(|status| {
            matches!(
                status.code(),
                tonic::Code::FailedPrecondition
                    | tonic::Code::Unauthenticated
                    | tonic::Code::PermissionDenied
            )
        })
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

    let matches = Args::command().get_matches();
    // Any source but the built-in default means the operator named this path themselves.
    let explicit_config = matches.value_source("config") != Some(ValueSource::DefaultValue);
    let args = Args::from_arg_matches(&matches)?;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.parse().unwrap()),
        )
        .init();

    // Raised before anything reads or inherits it, so a `memlock = "inherit"` job
    // does not depend on whether a BPF load ran first. Needs the subscriber to log.
    raise_own_memlock_rlimit();

    let hostname = args.hostname.unwrap_or_else(|| {
        hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".into())
    });

    // Parse listen port from the listen address for registration
    let listen_port: u16 = args
        .listen
        .rsplit(':')
        .next()
        .and_then(|p| p.parse().ok())
        .unwrap_or(6818);

    info!(
        version = %spur_core::version::version_string(),
        hostname = %hostname,
        controller = %args.controller,
        listen = %args.listen,
        "spurd starting"
    );

    // Load config from spur.conf (best-effort: missing file is fine)
    let config = match SlurmConfig::load_from_file(&args.config) {
        Ok(config) => {
            info!(path = %args.config.display(), "loaded spur.conf");
            Some(config)
        }
        Err(e) if absent_optional_config(explicit_config, &e) => {
            info!(path = %args.config.display(), "no spur.conf found, using default config");
            None
        }
        Err(e) => {
            warn!(
                path = %args.config.display(),
                error = %e,
                "failed to load spur.conf, using default config"
            );
            None
        }
    };
    let hooks_config = config.as_ref().map(|c| c.hooks.clone()).unwrap_or_default();

    let stepd_state_dir = args.state_dir.clone().unwrap_or_else(|| {
        config
            .as_ref()
            .map(|c| c.controller.state_dir.clone())
            .unwrap_or_else(|| "/var/spool/spur".into())
    });
    let stepds = stepd::StepdStore::new(&stepd_state_dir);
    let discovered_sessions = stepds.discover_live()?;
    // A corrupted descriptor still names its (job_id, run_attempt) in the
    // directory it lives in — fence it the same way as a stale session
    // instead of silently forgetting the job forever.
    let mut unparseable_rejected = Vec::new();
    let mut corrupted_stepds = Vec::new();
    for (path, reason) in discovered_sessions.rejected {
        match parse_session_dir_name(&path) {
            Some(identity) => corrupted_stepds.push(identity),
            None => unparseable_rejected.push((path, reason)),
        }
    }
    for (path, reason) in unparseable_rejected {
        warn!(path = %path.display(), %reason, "ignoring unusable stepd descriptor");
    }
    let recovered_stepds = discovered_sessions.live;
    let stale_stepds = discovered_sessions.stale;
    // A stale session's supervisor is confirmed dead, but that alone doesn't
    // kill its cgroup: if only the supervisor was killed (not the whole
    // process tree), the job's own process survives as an unsupervised
    // orphan. Reap it locally instead of leaving it running indefinitely.
    for descriptor in &stale_stepds {
        executor::cleanup_cgroup(&stale_cgroup_path(descriptor));
    }
    // A corrupted descriptor has no cgroup_path to read, but the path is
    // reconstructable from identity alone — reap it the same way.
    for &(job_id, run_attempt, _) in &corrupted_stepds {
        executor::cleanup_cgroup(&executor::expected_cgroup_path(job_id, run_attempt));
    }
    if !recovered_stepds.is_empty() {
        warn!(
            sessions = recovered_stepds.len(),
            root = %stepds.root().display(),
            "reporting PID-fenced stepds before asynchronous reconnect"
        );
    }

    // WireGuard interface for mesh address/key/peers. Resolution: SPUR_WG_INTERFACE env >
    // [network] wg_interface in spur.conf > "spur0", so a non-default conf name is honored.
    let wg_iface = std::env::var("SPUR_WG_INTERFACE").ok().unwrap_or_else(|| {
        config
            .as_ref()
            .map(|c| c.network.wg_interface.clone())
            .unwrap_or_else(|| "spur0".into())
    });

    // Background update check (non-blocking)
    spur_update::spawn_startup_check(
        "ROCm/spur",
        env!("CARGO_PKG_VERSION"),
        true,
        false, // auto_update
        "stable",
        "/var/cache/spur",
        spur_update::SPUR_BINARIES,
    );

    // Detect node address (explicit --address > WireGuard > hostname)
    let explicit_addr = args.address.clone();
    let node_address = if let Some(ref addr) = explicit_addr {
        let addr_input = addr.clone();
        match tokio::task::spawn_blocking(move || spur_net::normalize_comm_address(&addr_input))
            .await
            .map_err(|e| anyhow::anyhow!("comm address normalization task failed: {e}"))?
        {
            Ok(normalized) => {
                if spur_net::normalized_comm_addr_is_unusable(&normalized) {
                    warn!(
                        comm_addr = %normalized,
                        input = %addr,
                        "explicit comm address is not routable; inter-node jobs may fail"
                    );
                } else if normalized != *addr {
                    info!(
                        input = %addr,
                        comm_addr = %normalized,
                        "normalized explicit comm address"
                    );
                } else {
                    info!(comm_addr = %normalized, "using explicit comm address");
                }
                spur_net::address::NodeAddress {
                    ip: normalized,
                    hostname: hostname.clone(),
                    port: listen_port,
                    source: spur_net::address::AddressSource::Static,
                }
            }
            Err(e) => {
                warn!(
                    input = %addr,
                    error = %e,
                    "failed to normalize comm address; using raw value"
                );
                spur_net::address::NodeAddress {
                    ip: addr.clone(),
                    hostname: hostname.clone(),
                    port: listen_port,
                    source: spur_net::address::AddressSource::Static,
                }
            }
        }
    } else {
        let detect_hostname = hostname.clone();
        let detect_iface = wg_iface.clone();
        tokio::task::spawn_blocking(move || {
            spur_net::detect_node_address(&detect_hostname, listen_port, &detect_iface)
        })
        .await
        .map_err(|e| anyhow::anyhow!("node address detection task failed: {e}"))?
    };
    info!(
        ip = %node_address.ip,
        port = node_address.port,
        source = ?node_address.source,
        "node address detected"
    );

    // Initialize device registry (CDI cache, GRES config, and discovery).
    let registry = init_device_registry(config.as_ref());
    let registry = Arc::new(Mutex::new(registry));

    // Discover local resources (CPU/memory from sysfs, GPUs from device registry)
    let resources = {
        let reg = registry.lock().await;
        reporter::discover_resources(&reg)
    };
    info!(
        cpus = resources.cpus,
        memory_mb = resources.memory_mb,
        gpus = resources.gpus.len(),
        "resources discovered"
    );

    // Parse node labels from CLI/env
    let labels: HashMap<String, String> = args
        .labels
        .iter()
        .filter_map(|s| {
            let (k, v) = s.split_once('=')?;
            Some((k.to_string(), v.to_string()))
        })
        .collect();

    // The reporter re-reads the mesh key from `wg_iface` (resolved above) each heartbeat, so a
    // key that appears/changes after startup reaches the controller.

    // Shared between the reporter (reads held ids for heartbeats) and the agent
    // service (owns/mutates it) so the controller can reconcile stale allocations.
    let running_jobs = agent_server::new_running_jobs();
    agent_server::recover_stepds(&running_jobs, recovered_stepds.clone()).await;

    // Create the node reporter
    let reporter = Arc::new(NodeReporter::new(
        hostname.clone(),
        args.controller.clone(),
        resources,
        node_address,
        labels,
        args.token.unwrap_or_default(),
        wg_iface,
        running_jobs.clone(),
    ));

    // Register with controller
    reporter.register().await?;

    // Start the heartbeat loop right after registration, before the
    // completion-replay step below: replaying unacknowledged completions
    // retries against the controller (bounded, but can add several seconds
    // per pending completion) and must never delay heartbeats from resuming
    // after a restart — a slow replay must not look like a dead node.
    let hb_reporter = reporter.clone();
    tokio::spawn(async move {
        hb_reporter.heartbeat_loop().await;
    });

    let unacknowledged_stepd_completions: std::collections::HashSet<_> = stepds
        .discover_unacknowledged_completions()?
        .into_iter()
        .map(|completion| (completion.job_id, completion.run_attempt))
        .collect();
    let reconciled_stepd_completions =
        agent_server::replay_unacknowledged_stepd_completions(&stepds, &args.controller, &hostname)
            .await?;
    let reconciled_stepd_completions: std::collections::HashSet<_> =
        reconciled_stepd_completions.into_iter().collect();
    // Runs for the daemon's life, not just when this scan found something —
    // a push notification deferred later needs the same reconciliation.
    agent_server::retry_unacknowledged_stepd_completions(
        stepds.clone(),
        args.controller.clone(),
        hostname.clone(),
    );
    let pruned_stepds = stepds.prune_finalized()?;
    if pruned_stepds > 0 {
        info!(sessions = pruned_stepds, "pruned finalized stepd state");
    }

    // Start agent gRPC server (receives job launches + cluster-component RPCs from spurctld).
    // Pass the [cluster] config so the K0sAgent uses the operator's k0s version + install path.
    let limits = match config.as_ref() {
        Some(c) => spur_core::config::JobLimits {
            memlock: c.rlimits.memlock_limit()?,
        },
        None => spur_core::config::JobLimits::default(),
    };
    log_memlock_status(limits.memlock);
    let cgroup_config = config
        .as_ref()
        .map(|c| c.cgroup.clone())
        .unwrap_or_default();
    log_cgroup_status(&cgroup_config);
    let cluster_config = config
        .as_ref()
        .map(|c| c.cluster.clone())
        .unwrap_or_default();
    let mpi_config = config.as_ref().map(|c| c.mpi.clone()).unwrap_or_default();
    // Default-deny root execution: the job uid arrives on the wire and no RPC authenticates its
    // caller, so a uid-0 request must be refused unless the operator opted in.
    let allow_root_jobs = config
        .as_ref()
        .map(|c| c.auth.allow_root_jobs)
        .unwrap_or(false);
    if allow_root_jobs {
        // The option only has an effect when spurd itself is root — a non-root spurd cannot grant
        // root regardless — so do not claim the node will run jobs as root when it cannot.
        if nix::unistd::geteuid().is_root() {
            warn!(
                "[auth] allow_root_jobs is true: this node will execute jobs as root when asked. \
                 Only safe if every submitter is already trusted with root on this node."
            );
        } else {
            info!(
                "[auth] allow_root_jobs is true but spurd is not running as root, so it has no \
                 effect: jobs already run with spurd's (unprivileged) credentials."
            );
        }
    }
    let agent_service = agent_server::AgentService::with_cluster_config(
        reporter.clone(),
        hooks_config,
        registry.clone(),
        &cluster_config,
        limits,
        cgroup_config,
        mpi_config,
        running_jobs,
        allow_root_jobs,
    )
    .with_runtime_state_dir(stepd_state_dir.clone());
    agent_service.adopt_stepds(&recovered_stepds).await;
    // stepd_state_dir is shared with spurctld's own state_dir when the operator
    // hasn't set SPUR_STEPD_STATE_DIR, so it can't be required to be
    // exclusively spurd's — use the runtime/ subdir spurd already owns.
    stepd::create_private_dir_all(stepds.root())?;
    let agent_notify_socket = stepds.agent_socket();
    if agent_notify_socket.exists() {
        std::fs::remove_file(&agent_notify_socket)?;
    }
    let agent_notify_listener = tokio::net::UnixListener::bind(&agent_notify_socket)?;
    tokio::spawn(agent_server::serve_completion_notifications(
        agent_notify_listener,
        agent_service.completion_listener_context(),
    ));
    agent_service.monitor_recovered_stepds(&recovered_stepds);
    agent_service.monitor_stepd_liveness();
    let stepd_recovery_cleanup = agent_service.stepd_recovery_cleanup();
    let stepds_handle = agent_service.stepds_handle();

    // the RPC-driven k0s component owner is idle until the controller sends
    // StartClusterComponent; k0s then runs under its OWN systemd unit — never as a spurd job/child —
    // so it survives spurd restart and stays out of the executor/monitor/time-limit job path. The
    // background loop heals the unit; the SlurmAgent start/stop/status RPCs drive it.
    // Re-adopt an already-running k0s unit (spurd restart leaves it running) so status/heal are
    // correct immediately, then spawn the heal loop.
    let k0s = agent_service.k0s();
    // Only report k0s node status when this node actually supervises a k0s unit, so non-k0s
    // deployments don't emit spur_k8s_node_* series.
    if cluster_config.enabled {
        reporter.set_k0s_status(k0s.node_state());
    }
    k0s.adopt_running_unit().await;
    tokio::spawn(k0s.supervise());

    agent_service.start_monitor(args.controller.clone());

    let addr = args.listen.parse()?;
    info!(%addr, "agent gRPC server listening");

    // Authenticate callers of the agent surface: without this, reaching this port is enough to ask
    // the node to run work, which steps around the controller's authentication entirely.
    let auth_mode = config.as_ref().map(|c| c.auth.mode).unwrap_or_default();
    let jwt_key = config
        .as_ref()
        .map(|c| c.auth.resolved_jwt_key())
        .transpose()?
        .flatten()
        .unwrap_or_default();
    match auth_mode {
        spur_core::config::AuthMode::Required if jwt_key.is_empty() => {
            anyhow::bail!(
                "[auth] mode = \"required\" but no jwt_key or jwt_key_file is configured on this \
                 node: the agent could never verify a credential and would refuse every launch"
            )
        }
        spur_core::config::AuthMode::Required => {
            info!("agent requires a cluster credential on every RPC")
        }
        spur_core::config::AuthMode::Permissive => warn!(
            "agent accepts uncredentialed RPCs (auth.mode = permissive): any peer that can reach \
             this port can ask this node to run work. Set mode = \"required\" once controllers are \
             upgraded."
        ),
        spur_core::config::AuthMode::Disabled => warn!(
            "agent does NOT authenticate callers (auth.mode = disabled): treat this port as an \
             administrative boundary."
        ),
    }

    let server_future = tonic::transport::Server::builder()
        .layer(auth_middleware::AgentAuthLayer::new(auth_mode, &jwt_key))
        .add_service(spur_proto::agent_server(agent_service))
        .serve(addr);
    let server_task = tokio::spawn(server_future);

    if !recovered_stepds.is_empty() {
        for descriptor in recovered_stepds {
            let recovery_reporter = reporter.clone();
            let recovery_cleanup = stepd_recovery_cleanup.clone();
            tokio::spawn(async move {
                loop {
                    match recovery_reporter
                        .report_stepd_recovery(
                            descriptor.job_id,
                            descriptor.run_attempt,
                            descriptor.step_id,
                            false,
                        )
                        .await
                    {
                        Ok(response) => {
                            if response.fenced {
                                warn!(
                                    job_id = descriptor.job_id,
                                    run_attempt = descriptor.run_attempt,
                                    message = %response.message,
                                    "controller fenced recovered stepd"
                                );
                                recovery_cleanup.reject(&descriptor).await;
                            } else if response.retained {
                                info!(
                                    job_id = descriptor.job_id,
                                    run_attempt = descriptor.run_attempt,
                                    "controller retained recovered stepd"
                                );
                            } else {
                                info!(
                                    job_id = descriptor.job_id,
                                    run_attempt = descriptor.run_attempt,
                                    message = %response.message,
                                    "controller ignored stale recovered stepd"
                                );
                                recovery_cleanup.reject(&descriptor).await;
                            }
                            if response.retained && !response.message.is_empty() {
                                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                continue;
                            }
                            break;
                        }
                        Err(error) if is_permanent_recovery_error(&error) => {
                            warn!(
                                job_id = descriptor.job_id,
                                run_attempt = descriptor.run_attempt,
                                ?error,
                                "runtime recovery report permanently rejected by the controller; \
                                 giving up on this session"
                            );
                            break;
                        }
                        Err(error) => {
                            warn!(
                                job_id = descriptor.job_id,
                                run_attempt = descriptor.run_attempt,
                                %error,
                                "runtime recovery report failed; retrying"
                            );
                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        }
                    }
                }
            });
        }
    }

    let unreportable_sessions = stale_stepds
        .iter()
        .map(|descriptor| {
            (
                descriptor.job_id,
                descriptor.run_attempt,
                descriptor.step_id,
            )
        })
        .chain(corrupted_stepds)
        .filter(|(job_id, run_attempt, _)| {
            let id = (*job_id, *run_attempt);
            !reconciled_stepd_completions.contains(&id)
                && !unacknowledged_stepd_completions.contains(&id)
        });
    for (job_id, run_attempt, step_id) in unreportable_sessions {
        let recovery_reporter = reporter.clone();
        tokio::spawn(async move {
            loop {
                match recovery_reporter
                    .report_stepd_recovery(job_id, run_attempt, step_id, true)
                    .await
                {
                    Ok(response) => {
                        if response.fenced {
                            warn!(
                                job_id,
                                run_attempt,
                                message = %response.message,
                                "controller fenced stale stepd"
                            );
                        }
                        break;
                    }
                    Err(error) if is_permanent_recovery_error(&error) => {
                        warn!(
                            job_id,
                            run_attempt,
                            ?error,
                            "stale runtime recovery report permanently rejected by the controller; \
                             giving up on this session"
                        );
                        break;
                    }
                    Err(error) => {
                        warn!(
                            job_id,
                            run_attempt,
                            %error,
                            "stale runtime recovery report failed; retrying"
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                }
            }
        });
    }

    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    tokio::select! {
        result = server_task => { result??; }
        _ = sigterm.recv() => {
            if !stepds_handle.lock().await.is_empty() {
                info!("received SIGTERM with held stepds; preserving controller registration");
                return Ok(());
            }
            info!("received SIGTERM, deregistering from controller");
            let dereg_reporter = reporter.clone();
            match tokio::time::timeout(
                std::time::Duration::from_secs(5),
                dereg_reporter.deregister("agent shutdown"),
            )
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(e)) => warn!(error = %e, "deregistration failed"),
                Err(_) => warn!("deregistration timed out"),
            }
        }
    }

    Ok(())
}

fn init_device_registry(config: Option<&SlurmConfig>) -> DeviceRegistry {
    let default_devices = spur_core::config::DevicesConfig::default();
    let devices_config = config.map(|c| &c.devices).unwrap_or(&default_devices);

    let cdi_cache = CdiCache::load(&devices_config.cdi_spec_dirs, devices_config.auto_detect);

    let gres_entries: Vec<spur_devices::GresEntry> = devices_config
        .gres
        .iter()
        .map(|g| spur_devices::GresEntry {
            name: g.name.clone(),
            r#type: g.r#type.clone(),
            file: g.file.clone(),
            multiple_files: g.multiple_files.clone(),
            count: g.count,
            cores: g.cores.clone(),
            links: g.links.clone(),
            flags: g.flags.clone(),
        })
        .collect();
    let gres_cache = spur_devices::GresCache::from_entries(&gres_entries);

    let mut registry = DeviceRegistry::new();
    registry.populate(&cdi_cache, &gres_cache);

    info!(
        injectable_devices = registry.injectable_count(),
        countable = registry.countable_count(),
        "device registry initialized"
    );

    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_label_valid() {
        assert_eq!(parse_label("pool=gpu").unwrap(), "pool=gpu");
        assert_eq!(parse_label("tier=").unwrap(), "tier=");
        assert_eq!(parse_label("a=b=c").unwrap(), "a=b=c");
    }

    #[test]
    fn parse_label_missing_equals() {
        assert!(parse_label("noequalssign").is_err());
    }

    #[test]
    fn parse_label_empty_key() {
        assert!(parse_label("=value").is_err());
    }

    #[test]
    fn parse_label_just_equals() {
        assert!(parse_label("=").is_err());
    }

    #[test]
    fn a_stale_session_without_a_recorded_cgroup_still_names_one_to_reap() {
        let mut descriptor = stepd::StepdDescriptor::new(
            7,
            2,
            spur_core::step::STEP_BATCH,
            0,
            0,
            std::path::PathBuf::from("/run/runtime.sock"),
            std::path::PathBuf::new(),
        );
        assert_eq!(
            stale_cgroup_path(&descriptor),
            executor::expected_cgroup_path(7, 2),
            "an allocation session records no cgroup of its own, so its steps \
             would otherwise never be reaped"
        );

        descriptor.cgroup_path = std::path::PathBuf::from("/sys/fs/cgroup/spur/job_7_2");
        assert_eq!(
            stale_cgroup_path(&descriptor),
            std::path::PathBuf::from("/sys/fs/cgroup/spur/job_7_2")
        );
    }

    #[test]
    fn parse_session_dir_name_reads_the_full_identity() {
        assert_eq!(
            parse_session_dir_name(std::path::Path::new("42.1.4294967295")),
            Some((42, 1, 4294967295))
        );
    }

    #[test]
    fn parse_session_dir_name_rejects_a_two_component_name() {
        assert_eq!(parse_session_dir_name(std::path::Path::new("42.1")), None);
    }

    #[test]
    fn absent_config_at_default_path_is_expected() {
        let err = ConfigError::Io(std::io::Error::from(std::io::ErrorKind::NotFound));
        assert!(absent_optional_config(false, &err));
    }

    #[test]
    fn absent_config_at_explicit_path_is_reported() {
        // A typo'd --config must not be silently ignored.
        let err = ConfigError::Io(std::io::Error::from(std::io::ErrorKind::NotFound));
        assert!(!absent_optional_config(true, &err));
    }

    #[test]
    fn unreadable_config_is_reported_even_at_default_path() {
        // Present but unreadable is a misconfiguration, not an absent file.
        let err = ConfigError::Io(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
        assert!(!absent_optional_config(false, &err));
    }

    #[test]
    fn malformed_config_is_reported_even_at_default_path() {
        let err = SlurmConfig::load_from_str("this is not toml").expect_err("must not parse");
        assert!(!absent_optional_config(false, &err));
    }

    #[test]
    fn invalid_config_is_reported_even_at_default_path() {
        // Parses, then fails validation — settings the operator wrote are ignored.
        let err = SlurmConfig::load_from_str("cluster_name = \"\"").expect_err("must not validate");
        assert!(!absent_optional_config(false, &err));
    }
}
