// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;

use crate::executor::RunningJob;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepdLaunchSpec {
    pub job_id: u32,
    #[serde(default = "spur_core::step::default_step_id")]
    pub step_id: spur_core::step::StepId,
    pub script: String,
    pub work_dir: String,
    pub name: String,
    pub user: String,
    pub node: String,
    pub environment: std::collections::HashMap<String, String>,
    pub stdout_path: String,
    pub stderr_path: String,
    pub stdin_path: String,
    pub cpus: u32,
    pub memory_mb: u64,
    pub cpu_ids: Vec<u32>,
    #[serde(default)]
    pub gpu_devices: Vec<u32>,
    pub open_mode: Option<String>,
    pub uid: u32,
    pub gid: u32,
    pub partition: String,
    pub nodelist: String,
    pub memlock: StepdMemlock,
    #[serde(default)]
    pub container: Option<crate::executor::ContainerLaunchConfig>,
    #[serde(default)]
    pub host_device_plan: Option<spur_devices::inject::HostInjectionPlan>,
    #[serde(default)]
    pub container_rootfs_mode: Option<crate::container::RootfsMode>,
    #[serde(default)]
    pub hooks: spur_core::config::HooksConfig,
    #[serde(default)]
    pub plugstack_path: String,
    #[serde(default)]
    pub controller_addr: String,
    #[serde(default)]
    pub reporting_node: String,
    #[serde(default)]
    pub run_attempt: u32,
    #[serde(default)]
    pub capability: String,
    #[serde(default)]
    pub allocation_only: bool,
    // Batch `--mpi=pmix` jobs skip process_group(0)/namespace isolation (see
    // executor.rs); persisted so a restarted supervisor launches identically.
    #[serde(default)]
    pub pmix_multi_task: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepdMemlock {
    Unlimited,
    Inherit,
    Bytes(u64),
}

impl From<spur_core::config::MemlockLimit> for StepdMemlock {
    fn from(limit: spur_core::config::MemlockLimit) -> Self {
        match limit {
            spur_core::config::MemlockLimit::Unlimited => StepdMemlock::Unlimited,
            spur_core::config::MemlockLimit::Inherit => StepdMemlock::Inherit,
            spur_core::config::MemlockLimit::Bytes(value) => StepdMemlock::Bytes(value),
        }
    }
}

impl From<StepdMemlock> for spur_core::config::MemlockLimit {
    fn from(limit: StepdMemlock) -> Self {
        match limit {
            StepdMemlock::Unlimited => spur_core::config::MemlockLimit::Unlimited,
            StepdMemlock::Inherit => spur_core::config::MemlockLimit::Inherit,
            StepdMemlock::Bytes(value) => spur_core::config::MemlockLimit::Bytes(value),
        }
    }
}

impl TryFrom<&crate::executor::JobLaunchConfig> for StepdLaunchSpec {
    type Error = String;

    fn try_from(config: &crate::executor::JobLaunchConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            job_id: config.job_id,
            // Overridden by the caller for an allocation-only (extern-step)
            // launch; every other launch path in this build is the batch step.
            step_id: spur_core::step::STEP_BATCH,
            script: config.script.clone(),
            work_dir: config.work_dir.clone(),
            name: config.name.clone(),
            user: config.user.clone(),
            node: config.node.clone(),
            environment: config.environment.clone(),
            stdout_path: config.stdout_path.clone(),
            stderr_path: config.stderr_path.clone(),
            stdin_path: config.stdin_path.clone(),
            cpus: config.cpus,
            memory_mb: config.memory_mb,
            gpu_devices: config.gpu_devices.clone(),
            cpu_ids: config.cpu_ids.clone(),
            open_mode: config.open_mode.clone(),
            uid: config.uid,
            gid: config.gid,
            partition: config.partition.clone(),
            nodelist: config.nodelist.clone(),
            memlock: config.memlock.into(),
            container: config.container.clone(),
            host_device_plan: config.host_device_plan.clone(),
            container_rootfs_mode: None,
            hooks: spur_core::config::HooksConfig::default(),
            plugstack_path: String::new(),
            controller_addr: String::new(),
            reporting_node: String::new(),
            run_attempt: config.run_attempt,
            capability: String::new(),
            allocation_only: false,
            pmix_multi_task: config.pmix_multi_task,
        })
    }
}

impl StepdLaunchSpec {
    pub fn into_launch_config(self) -> crate::executor::JobLaunchConfig {
        crate::executor::JobLaunchConfig {
            job_id: self.job_id,
            run_attempt: self.run_attempt,
            script: self.script,
            work_dir: self.work_dir,
            name: self.name,
            user: self.user,
            node: self.node,
            array_job_id: None,
            array_task_id: None,
            environment: self.environment,
            stdout_path: self.stdout_path,
            stderr_path: self.stderr_path,
            stdin_path: self.stdin_path,
            cpus: self.cpus,
            memory_mb: self.memory_mb,
            gpu_devices: self.gpu_devices,
            cpu_ids: self.cpu_ids,
            open_mode: self.open_mode,
            uid: self.uid,
            gid: self.gid,
            container: self.container,
            prolog_script: None,
            partition: self.partition,
            nodelist: self.nodelist,
            host_device_plan: self.host_device_plan,
            memlock: self.memlock.into(),
            io_mode: crate::executor::LaunchIo::File,
            pmix_multi_task: self.pmix_multi_task,
        }
    }
}

const DESCRIPTOR_FILE: &str = "descriptor.json";
const OBLIGATION_FILE: &str = "obligations.jsonl";
const FAILURE_FILE: &str = "failure.txt";
const FORMAT_VERSION: u32 = 1;
pub const PROTOCOL_VERSION: u32 = 1;
const CONTROL_LINE_LIMIT: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum StepdRequest {
    Hello {
        protocol_version: u32,
        capability: String,
        spurd_instance_id: String,
        run_attempt: u32,
        #[serde(default = "spur_core::step::default_step_id")]
        step_id: spur_core::step::StepId,
    },
    QueryState,
    SignalAllocation {
        signal: i32,
    },
    BeginTeardown,
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum StepdResponse {
    Hello {
        protocol_version: u32,
        job_id: u32,
        run_attempt: u32,
        step_id: spur_core::step::StepId,
    },
    State {
        job_id: u32,
        run_attempt: u32,
        step_id: spur_core::step::StepId,
        active: bool,
        exit_code: Option<i32>,
        signal: Option<i32>,
    },
    Acknowledged,
    Rejected {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "notification", rename_all = "snake_case")]
pub enum AgentNotification {
    StepdCompleted {
        job_id: u32,
        run_attempt: u32,
        step_id: spur_core::step::StepId,
        exit_code: i32,
        signal: i32,
        epilog_failed: bool,
        capability: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum AgentNotificationResponse {
    /// spurd forwarded the completion to the controller; safe to prune.
    Acknowledged,
    /// spurd released its local tracking but could not reach the
    /// controller; leave the durable record for the next startup scan.
    Deferred,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepdSnapshot {
    pub job_id: u32,
    pub run_attempt: u32,
    pub step_id: spur_core::step::StepId,
    pub active: bool,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "obligation", rename_all = "snake_case")]
pub enum StepdObligation {
    ExitObserved { exit_code: i32, signal: i32 },
    CompletionAcknowledged,
    ResourcesReleased,
}

pub struct StepdObligationLog {
    path: PathBuf,
}

impl StepdObligationLog {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn append(&self, obligation: &StepdObligation) -> io::Result<()> {
        let mut entry = serde_json::to_vec(obligation).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("serialize runtime obligation: {error}"),
            )
        })?;
        entry.push(b'\n');
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(&entry)?;
        file.sync_data()
    }

    pub fn read(&self) -> io::Result<Vec<StepdObligation>> {
        let contents = match fs::read_to_string(&self.path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        contents
            .lines()
            .enumerate()
            .map(|(line, entry)| {
                serde_json::from_str(entry).map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid runtime obligation at line {}: {error}", line + 1),
                    )
                })
            })
            .collect()
    }
}

/// `read_line` with no cap on line length; a peer that never sends `\n` grows
/// `line` unboundedly. Bound it to a sane control-protocol frame size.
pub(crate) async fn read_line_bounded<R>(reader: &mut R, line: &mut String) -> io::Result<usize>
where
    R: AsyncBufReadExt + Unpin,
{
    let n = reader
        .take(CONTROL_LINE_LIMIT as u64)
        .read_line(line)
        .await?;
    if n >= CONTROL_LINE_LIMIT && !line.ends_with('\n') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "control line exceeded maximum length",
        ));
    }
    Ok(n)
}

/// Sibling of the stepd store root, not inside it, so directory
/// scans over session state (`discover_live`, `prune_finalized`) never see it.
pub(crate) const AGENT_NOTIFY_SOCKET_NAME: &str = "agent.sock";
const STEPD_HANDSHAKE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
const CONTROL_REQUEST_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

// A Stepd supervises exactly one step's process tree — the batch script for
// STEP_BATCH today; an interactive shell (STEP_INTERACTIVE) or a numbered
// srun step gets its own separate Stepd process in follow-up work, not a
// second process hosted inside this one.

pub struct Stepd {
    job: Mutex<RunningJob>,
    snapshot: Arc<Mutex<StepdSnapshot>>,
    teardown_started: AtomicBool,
    launch_gate: Mutex<()>,
    // Base environment for work landing in a follow-up PR.
    #[allow(dead_code)]
    environment: std::collections::HashMap<String, String>,
}

impl Stepd {
    pub fn new(
        job: RunningJob,
        job_id: u32,
        run_attempt: u32,
        step_id: spur_core::step::StepId,
    ) -> Self {
        Self::with_environment(
            job,
            job_id,
            run_attempt,
            step_id,
            std::collections::HashMap::new(),
        )
    }

    pub fn with_environment(
        job: RunningJob,
        job_id: u32,
        run_attempt: u32,
        step_id: spur_core::step::StepId,
        environment: std::collections::HashMap<String, String>,
    ) -> Self {
        Self {
            job: Mutex::new(job),
            snapshot: Arc::new(Mutex::new(StepdSnapshot {
                job_id,
                run_attempt,
                step_id,
                active: true,
                exit_code: None,
                signal: None,
            })),
            teardown_started: AtomicBool::new(false),
            launch_gate: Mutex::new(()),
            environment,
        }
    }

    pub async fn snapshot(&self) -> StepdSnapshot {
        self.snapshot.lock().await.clone()
    }

    async fn take_cgroup(&self) -> Option<PathBuf> {
        self.job.lock().await.take_cgroup()
    }

    pub async fn poll_completion(&self) -> io::Result<()> {
        let (allocation_only, completed) = {
            let mut job = self.job.lock().await;
            let allocation_only = job.is_allocation_only();
            let completed = job.try_wait().map_err(io::Error::other)?;
            (allocation_only, completed)
        };
        if let Some((exit_code, signal)) = completed {
            let mut snapshot = self.snapshot.lock().await;
            snapshot.active = false;
            snapshot.exit_code = Some(exit_code);
            snapshot.signal = Some(signal);
            return Ok(());
        }
        // An extern-step (allocation-only) stepd has no process of its own to
        // wait on — a requested teardown is itself the completion signal.
        if allocation_only && self.teardown_started.load(Ordering::Acquire) {
            let mut snapshot = self.snapshot.lock().await;
            snapshot.active = false;
            snapshot.signal = Some(nix::sys::signal::Signal::SIGTERM as i32);
        }
        Ok(())
    }

    pub async fn signal(&self, signal: i32) -> io::Result<()> {
        let signal = nix::sys::signal::Signal::try_from(signal).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid signal: {error}"),
            )
        })?;
        let job = self.job.lock().await;
        job.kill_signal(signal).map_err(io::Error::other)?;
        if signal == nix::sys::signal::Signal::SIGKILL {
            if let Some(cgroup_path) = job.cgroup_path() {
                // Belt-and-suspenders: reaches descendants that detached
                // from the signaled process group (e.g. via setsid).
                if let Err(error) = crate::executor::cgroup_kill(cgroup_path) {
                    tracing::warn!(%error, path = %cgroup_path.display(), "cgroup.kill failed");
                }
            }
        }
        Ok(())
    }

    pub async fn begin_teardown(&self) {
        let launch_gate = self.launch_gate.lock().await;
        self.teardown_started.store(true, Ordering::Release);
        drop(launch_gate);
        let job = self.job.lock().await;
        if let Err(error) = job.kill_signal(nix::sys::signal::Signal::SIGTERM) {
            tracing::warn!(%error, "failed to terminate stepd process during teardown");
        }
    }
}

impl StepdSnapshot {
    pub fn response(&self) -> StepdResponse {
        StepdResponse::State {
            job_id: self.job_id,
            run_attempt: self.run_attempt,
            step_id: self.step_id,
            active: self.active,
            exit_code: self.exit_code,
            signal: self.signal,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepdDescriptor {
    pub format_version: u32,
    pub job_id: u32,
    pub run_attempt: u32,
    #[serde(default = "spur_core::step::default_step_id")]
    pub step_id: spur_core::step::StepId,
    pub pid: u32,
    pub process_start_ticks: u64,
    pub socket_path: PathBuf,
    pub cgroup_path: PathBuf,
    #[serde(default)]
    pub capability: String,
    #[serde(default)]
    pub owner: String,
    #[serde(default)]
    pub uid: u32,
    #[serde(default)]
    pub gid: u32,
    #[serde(default)]
    pub work_dir: String,
}

impl StepdDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        job_id: u32,
        run_attempt: u32,
        step_id: spur_core::step::StepId,
        pid: u32,
        process_start_ticks: u64,
        socket_path: PathBuf,
        cgroup_path: PathBuf,
    ) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            job_id,
            run_attempt,
            step_id,
            pid,
            process_start_ticks,
            socket_path,
            cgroup_path,
            capability: uuid::Uuid::new_v4().to_string(),
            owner: String::new(),
            uid: 0,
            gid: 0,
            work_dir: String::new(),
        }
    }
}

pub(crate) fn record_resources_released(descriptor: &StepdDescriptor) -> io::Result<()> {
    let session_dir = descriptor.socket_path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "runtime socket path has no session directory",
        )
    })?;
    let obligations = StepdObligationLog::new(session_dir.join(OBLIGATION_FILE));
    if obligations
        .read()?
        .iter()
        .any(|obligation| matches!(obligation, StepdObligation::ResourcesReleased))
    {
        return prune_finalized_session(session_dir, &obligations).map(|_| ());
    }
    obligations.append(&StepdObligation::ResourcesReleased)?;
    prune_finalized_session(session_dir, &obligations).map(|_| ())
}

fn finalized_obligations(obligations: &[StepdObligation]) -> bool {
    let mut exit_observed = false;
    let mut completion_acknowledged = false;
    let mut resources_released = false;

    for obligation in obligations {
        match obligation {
            StepdObligation::ExitObserved { .. } => {
                exit_observed = true;
                completion_acknowledged = false;
            }
            StepdObligation::CompletionAcknowledged if exit_observed => {
                completion_acknowledged = true;
            }
            StepdObligation::ResourcesReleased => resources_released = true,
            StepdObligation::CompletionAcknowledged => {}
        }
    }

    completion_acknowledged && resources_released
}

fn prune_finalized_session(
    session_dir: &Path,
    obligations: &StepdObligationLog,
) -> io::Result<bool> {
    if !finalized_obligations(&obligations.read()?) {
        return Ok(false);
    }
    fs::remove_dir_all(session_dir)?;
    Ok(true)
}

pub fn validate_hello(
    descriptor: &StepdDescriptor,
    capability: &str,
    expected_capability: &str,
    protocol_version: u32,
    run_attempt: u32,
    step_id: spur_core::step::StepId,
) -> StepdResponse {
    if protocol_version != PROTOCOL_VERSION {
        return StepdResponse::Rejected {
            message: format!(
                "runtime protocol {protocol_version} is incompatible with {PROTOCOL_VERSION}"
            ),
        };
    }
    if expected_capability.is_empty()
        || capability.len() != expected_capability.len()
        || !bool::from(subtle::ConstantTimeEq::ct_eq(
            capability.as_bytes(),
            expected_capability.as_bytes(),
        ))
    {
        return StepdResponse::Rejected {
            message: "runtime capability rejected".into(),
        };
    }
    if run_attempt != descriptor.run_attempt || step_id != descriptor.step_id {
        return StepdResponse::Rejected {
            message: "runtime attempt is stale".into(),
        };
    }
    StepdResponse::Hello {
        protocol_version,
        job_id: descriptor.job_id,
        run_attempt: descriptor.run_attempt,
        step_id: descriptor.step_id,
    }
}

pub async fn accept_hello(
    listener: &UnixListener,
    descriptor: &StepdDescriptor,
    expected_capability: &str,
) -> io::Result<(UnixStream, String)> {
    let (stream, _) = listener.accept().await?;
    accept_hello_stream(stream, descriptor, expected_capability).await
}

async fn accept_hello_stream(
    stream: UnixStream,
    descriptor: &StepdDescriptor,
    expected_capability: &str,
) -> io::Result<(UnixStream, String)> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    read_line_bounded(&mut reader, &mut line).await?;
    let request: StepdRequest = serde_json::from_str(&line).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid runtime request: {error}"),
        )
    })?;
    let StepdRequest::Hello {
        protocol_version,
        capability,
        spurd_instance_id,
        run_attempt,
        step_id,
    } = request
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "runtime connection did not begin with hello",
        ));
    };
    let response = validate_hello(
        descriptor,
        &capability,
        expected_capability,
        protocol_version,
        run_attempt,
        step_id,
    );
    let accepted = matches!(response, StepdResponse::Hello { .. });
    let mut stream = reader.into_inner();
    let response = serde_json::to_vec(&response).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("encode runtime response: {error}"),
        )
    })?;
    stream.write_all(&response).await?;
    stream.write_all(b"\n").await?;
    if !accepted {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "runtime hello rejected",
        ));
    }
    Ok((stream, spurd_instance_id))
}

pub async fn query_state(
    descriptor: &StepdDescriptor,
    spurd_instance_id: String,
) -> io::Result<StepdSnapshot> {
    match stepd_request(descriptor, spurd_instance_id, StepdRequest::QueryState).await? {
        StepdResponse::State {
            job_id,
            run_attempt,
            step_id,
            active,
            exit_code,
            signal,
        } if job_id == descriptor.job_id
            && run_attempt == descriptor.run_attempt
            && step_id == descriptor.step_id =>
        {
            Ok(StepdSnapshot {
                job_id,
                run_attempt,
                step_id,
                active,
                exit_code,
                signal,
            })
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "runtime state identity mismatch",
        )),
    }
}

pub async fn signal_allocation(
    descriptor: &StepdDescriptor,
    spurd_instance_id: String,
    signal: i32,
) -> io::Result<()> {
    match stepd_request(
        descriptor,
        spurd_instance_id,
        StepdRequest::SignalAllocation { signal },
    )
    .await?
    {
        StepdResponse::Acknowledged => Ok(()),
        StepdResponse::Rejected { message } => {
            Err(io::Error::new(io::ErrorKind::PermissionDenied, message))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "runtime signal response was invalid",
        )),
    }
}

pub async fn shutdown_allocation(
    descriptor: &StepdDescriptor,
    spurd_instance_id: String,
) -> io::Result<()> {
    match stepd_request(descriptor, spurd_instance_id, StepdRequest::Shutdown).await? {
        StepdResponse::Acknowledged => Ok(()),
        StepdResponse::Rejected { message } => {
            Err(io::Error::new(io::ErrorKind::PermissionDenied, message))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "runtime shutdown response was invalid",
        )),
    }
}

async fn stepd_request(
    descriptor: &StepdDescriptor,
    spurd_instance_id: String,
    request: StepdRequest,
) -> io::Result<StepdResponse> {
    let (mut reader, mut writer) = stepd_connect(descriptor, spurd_instance_id).await?;
    write_request(&mut writer, &request).await?;
    read_response(&mut reader).await
}

async fn stepd_connect(
    descriptor: &StepdDescriptor,
    spurd_instance_id: String,
) -> io::Result<(
    BufReader<tokio::net::unix::OwnedReadHalf>,
    tokio::net::unix::OwnedWriteHalf,
)> {
    let stream = UnixStream::connect(&descriptor.socket_path).await?;
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    stepd_hello(&mut reader, &mut writer, descriptor, spurd_instance_id).await?;
    Ok((reader, writer))
}

async fn stepd_hello(
    reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>,
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    descriptor: &StepdDescriptor,
    spurd_instance_id: String,
) -> io::Result<()> {
    let hello = StepdRequest::Hello {
        protocol_version: PROTOCOL_VERSION,
        capability: descriptor.capability.clone(),
        spurd_instance_id,
        run_attempt: descriptor.run_attempt,
        step_id: descriptor.step_id,
    };
    write_request(writer, &hello).await?;
    match read_response(reader).await? {
        StepdResponse::Hello {
            protocol_version,
            job_id,
            run_attempt,
            step_id,
        } if job_id == descriptor.job_id
            && run_attempt == descriptor.run_attempt
            && step_id == descriptor.step_id
            && protocol_version == PROTOCOL_VERSION =>
        {
            Ok(())
        }
        StepdResponse::Rejected { message } => {
            Err(io::Error::new(io::ErrorKind::PermissionDenied, message))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "runtime hello identity mismatch",
        )),
    }
}

async fn write_request(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    request: &StepdRequest,
) -> io::Result<()> {
    let request = serde_json::to_vec(request).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("encode runtime request: {error}"),
        )
    })?;
    writer.write_all(&request).await?;
    writer.write_all(b"\n").await
}

async fn read_response(
    reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>,
) -> io::Result<StepdResponse> {
    let mut line = String::new();
    read_line_bounded(reader, &mut line).await?;
    serde_json::from_str(&line).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid runtime response: {error}"),
        )
    })
}

const AGENT_NOTIFY_ATTEMPTS: u32 = 5;
const AGENT_NOTIFY_RETRY_GAP: std::time::Duration = std::time::Duration::from_secs(1);
const AGENT_NOTIFY_ATTEMPT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Push this session's completion to spurd. `None` means the caller must
/// leave the durable record for spurd's next-startup recovery scan.
pub(crate) async fn notify_agent_completion(
    agent_socket: &Path,
    notification: &AgentNotification,
) -> Option<AgentNotificationResponse> {
    for attempt in 1..=AGENT_NOTIFY_ATTEMPTS {
        let outcome = tokio::time::timeout(
            AGENT_NOTIFY_ATTEMPT_TIMEOUT,
            try_notify_agent(agent_socket, notification),
        )
        .await
        .unwrap_or_else(|_| {
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "spurd did not answer",
            ))
        });
        match outcome {
            Ok(response) => return Some(response),
            Err(error) => {
                tracing::warn!(
                    attempt,
                    %error,
                    "failed to notify spurd of stepd completion"
                );
                if attempt < AGENT_NOTIFY_ATTEMPTS {
                    tokio::time::sleep(AGENT_NOTIFY_RETRY_GAP).await;
                }
            }
        }
    }
    None
}

async fn try_notify_agent(
    agent_socket: &Path,
    notification: &AgentNotification,
) -> io::Result<AgentNotificationResponse> {
    let stream = UnixStream::connect(agent_socket).await?;
    let (reader, mut writer) = stream.into_split();
    let payload = serde_json::to_vec(notification).map_err(io::Error::other)?;
    writer.write_all(&payload).await?;
    writer.write_all(b"\n").await?;
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    read_line_bounded(&mut reader, &mut line).await?;
    serde_json::from_str(&line).map_err(io::Error::other)
}

pub async fn serve_control(stream: UnixStream, session: &Stepd) -> io::Result<()> {
    let mut reader = BufReader::new(stream);
    loop {
        let mut line = String::new();
        // A connected client that never sends another request (hung, or
        // deliberately holding the socket open) must not pin this task
        // forever; the handshake already bounds how long accepting a new
        // connection can take, this bounds each subsequent request on it.
        let read = tokio::time::timeout(
            CONTROL_REQUEST_IDLE_TIMEOUT,
            read_line_bounded(&mut reader, &mut line),
        )
        .await
        .map_err(|_| {
            io::Error::new(io::ErrorKind::TimedOut, "control connection idle timeout")
        })??;
        if read == 0 {
            return Ok(());
        }
        let request: StepdRequest = serde_json::from_str(&line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid runtime request: {error}"),
            )
        })?;
        let response = match request {
            StepdRequest::QueryState => session.snapshot().await.response(),
            StepdRequest::BeginTeardown | StepdRequest::Shutdown => {
                session.begin_teardown().await;
                StepdResponse::Acknowledged
            }
            StepdRequest::SignalAllocation { signal } => match session.signal(signal).await {
                Ok(()) => StepdResponse::Acknowledged,
                Err(error) => StepdResponse::Rejected {
                    message: error.to_string(),
                },
            },
            StepdRequest::Hello { .. } => StepdResponse::Rejected {
                message: "runtime hello is only valid as the first request".into(),
            },
        };
        let response = serde_json::to_vec(&response).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("encode runtime response: {error}"),
            )
        })?;
        reader.get_mut().write_all(&response).await?;
        reader.get_mut().write_all(b"\n").await?;
    }
}

pub async fn run_supervisor(
    listener: UnixListener,
    descriptor: StepdDescriptor,
    session: Arc<Stepd>,
) -> io::Result<()> {
    let mut poll_interval = tokio::time::interval(std::time::Duration::from_millis(100));
    loop {
        tokio::select! {
            _ = poll_interval.tick() => {
                session.poll_completion().await?;
                if !session.snapshot().await.active {
                    return Ok(());
                }
            }
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _)) => {
                        let session = session.clone();
                        let descriptor = descriptor.clone();
                        tokio::spawn(serve_supervisor_connection(stream, descriptor, session));
                    }
                    Err(error) => {
                        tracing::warn!(%error, "stepd connection handshake failed");
                    }
                }
            }
        }
    }
}

async fn serve_supervisor_connection(
    stream: UnixStream,
    descriptor: StepdDescriptor,
    session: Arc<Stepd>,
) {
    let capability = descriptor.capability.clone();
    match tokio::time::timeout(
        STEPD_HANDSHAKE_TIMEOUT,
        accept_hello_stream(stream, &descriptor, &capability),
    )
    .await
    {
        Ok(Ok((stream, _))) => {
            if let Err(error) = serve_control(stream, &session).await {
                tracing::warn!(%error, "stepd control connection failed");
            }
        }
        Ok(Err(error)) => {
            tracing::warn!(%error, "stepd connection handshake failed");
        }
        Err(_) => {
            tracing::warn!("stepd connection handshake timed out");
        }
    }
}

pub async fn run_process(args: &[String]) -> anyhow::Result<i32> {
    if args.len() != 4 {
        anyhow::bail!("usage: spurd __stepd <state-dir> <job-id> <attempt> <launch-spec>");
    }
    let state_dir = PathBuf::from(&args[0]);
    let job_id: u32 = args[1]
        .parse()
        .map_err(|error| anyhow::anyhow!("invalid job id: {error}"))?;
    let run_attempt: u32 = args[2]
        .parse()
        .map_err(|error| anyhow::anyhow!("invalid run attempt: {error}"))?;
    let launch_spec: StepdLaunchSpec = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    if launch_spec.job_id != job_id {
        anyhow::bail!("runtime launch spec job id does not match process arguments");
    }
    let step_id = launch_spec.step_id;
    let agent_socket = state_dir.join(AGENT_NOTIFY_SOCKET_NAME);
    let store = StepdStore::new(state_dir);
    let session_dir = store.session_dir(job_id, run_attempt, step_id);
    let obligations = store.obligations(job_id, run_attempt, step_id);
    let socket_path = session_dir.join("runtime.sock");
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }
    let pid = std::process::id();
    let mut descriptor = StepdDescriptor::new(
        job_id,
        run_attempt,
        step_id,
        pid,
        process_start_ticks(pid)?,
        socket_path.clone(),
        PathBuf::new(),
    );
    if !launch_spec.capability.is_empty() {
        descriptor.capability = launch_spec.capability.clone();
    }
    descriptor.owner = launch_spec.user.clone();
    descriptor.uid = launch_spec.uid;
    descriptor.gid = launch_spec.gid;
    descriptor.work_dir = launch_spec.work_dir.clone();
    store.publish(&descriptor)?;
    let listener = UnixListener::bind(&socket_path)?;
    let runtime_environment = launch_spec.environment.clone();
    let container_rootfs_mode = launch_spec.container_rootfs_mode.clone();
    let hooks = launch_spec.hooks.clone();
    let spank = load_runtime_spank(&launch_spec.plugstack_path);
    let hook_context = spur_core::hooks::HookContext {
        job_id,
        work_dir: launch_spec.work_dir.clone(),
        uid: launch_spec.uid,
        gid: launch_spec.gid,
        partition: launch_spec.partition.clone(),
        nodelist: launch_spec.nodelist.clone(),
        script_context: "epilog_slurmd".into(),
        gpu_devices: launch_spec.gpu_devices.clone(),
        cpus: launch_spec.cpus,
        memory_mb: launch_spec.memory_mb,
    };
    let job = if launch_spec.allocation_only {
        RunningJob::AllocationOnly
    } else {
        match crate::executor::launch_job(&launch_spec.into_launch_config(), spank.as_ref()).await {
            Ok(result) => result.job,
            Err(error) => {
                if let Some(rootfs_mode) = container_rootfs_mode.as_ref() {
                    crate::container::cleanup_rootfs(job_id, rootfs_mode);
                }
                crate::executor::cleanup_job_spool(job_id);
                return Err(anyhow::anyhow!(error.to_string()));
            }
        }
    };
    if let Some(cgroup_path) = job.cgroup_path() {
        descriptor.cgroup_path = cgroup_path.to_path_buf();
        if let Err(error) = store.publish(&descriptor) {
            tracing::warn!(job_id, %error, "failed to republish runtime descriptor with cgroup path");
        }
    }
    let session = Arc::new(Stepd::with_environment(
        job,
        job_id,
        run_attempt,
        step_id,
        runtime_environment,
    ));
    let capability = descriptor.capability.clone();
    let result = run_supervisor(listener, descriptor, session.clone()).await;
    let _ = std::fs::remove_file(socket_path);
    if let Some(cgroup) = session.take_cgroup().await {
        crate::executor::cleanup_cgroup(&cgroup).await;
    }
    if let Some(rootfs_mode) = container_rootfs_mode.as_ref() {
        crate::container::cleanup_rootfs(job_id, rootfs_mode);
    }
    crate::executor::cleanup_job_spool(job_id);
    if let Err(error) = result {
        let failure_path = session_dir.join(FAILURE_FILE);
        if let Err(write_error) = std::fs::write(&failure_path, error.to_string()) {
            tracing::warn!(%write_error, path = %failure_path.display(), "failed to record stepd failure");
        }
        return Err(error.into());
    }
    let snapshot = session.snapshot().await;
    let exit_code = snapshot
        .exit_code
        .unwrap_or_else(|| 128 + snapshot.signal.unwrap_or(0));
    let epilog_failed = if let Some(epilog) = hooks.epilog.as_deref() {
        if let Err(error) = spur_core::hooks::run_hook(epilog, &hook_context).await {
            tracing::error!(job_id, %error, "runtime epilog hook failed");
            true
        } else {
            false
        }
    } else {
        false
    };
    if let Some(spank) = spank.as_ref() {
        let context = spur_spank::SpankContext {
            job_id,
            uid: hook_context.uid,
            gid: hook_context.gid,
            ..Default::default()
        };
        let mut handle = spur_spank::SpankHandle::new(context, HashMap::new());
        for hook in [
            spur_spank::SpankHook::TaskExit,
            spur_spank::SpankHook::JobEpilog,
        ] {
            if let Err(error) = spank.invoke_hook(hook, &mut handle) {
                tracing::warn!(job_id, %error, hook = hook.symbol_name(), "runtime SPANK exit hook failed");
            }
        }
    }
    let signal = snapshot.signal.unwrap_or(0);
    obligations.append(&StepdObligation::ExitObserved { exit_code, signal })?;
    let notification = AgentNotification::StepdCompleted {
        job_id,
        run_attempt,
        step_id,
        exit_code,
        signal,
        epilog_failed,
        capability,
    };
    if notify_agent_completion(&agent_socket, &notification).await
        == Some(AgentNotificationResponse::Acknowledged)
    {
        store.acknowledge_completion(&PendingStepdCompletion {
            job_id,
            run_attempt,
            step_id,
            exit_code,
            signal,
        })?;
    }
    Ok(exit_code)
}

fn load_runtime_spank(plugstack_path: &str) -> Option<spur_spank::SpankHost> {
    if plugstack_path.is_empty() || !Path::new(plugstack_path).exists() {
        return None;
    }
    let entries = match spur_spank::parse_plugstack(Path::new(plugstack_path)) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(path = plugstack_path, %error, "failed to parse runtime plugstack");
            return None;
        }
    };
    let mut host = spur_spank::SpankHost::new();
    for entry in entries {
        if let Err(error) = host.load_plugin(&entry.path, &entry.args) {
            tracing::warn!(plugin = %entry.path.display(), %error, required = entry.required, "runtime SPANK plugin failed to load");
        }
    }
    (host.plugin_count() > 0).then_some(host)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepdLiveness {
    Live,
    Stale,
}

#[derive(Debug)]
pub(crate) struct DiscoveredStepds {
    pub live: Vec<StepdDescriptor>,
    pub stale: Vec<StepdDescriptor>,
    pub rejected: Vec<(PathBuf, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PendingStepdCompletion {
    pub job_id: u32,
    pub run_attempt: u32,
    pub step_id: spur_core::step::StepId,
    pub exit_code: i32,
    pub signal: i32,
}

#[derive(Clone)]
pub struct StepdStore {
    root: PathBuf,
}

impl StepdStore {
    pub fn new(state_dir: impl Into<PathBuf>) -> Self {
        Self {
            root: state_dir.into().join("runtime"),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn session_dir(
        &self,
        job_id: u32,
        run_attempt: u32,
        step_id: spur_core::step::StepId,
    ) -> PathBuf {
        self.root.join(format!("{job_id}.{run_attempt}.{step_id}"))
    }

    pub fn obligations(
        &self,
        job_id: u32,
        run_attempt: u32,
        step_id: spur_core::step::StepId,
    ) -> StepdObligationLog {
        StepdObligationLog::new(
            self.session_dir(job_id, run_attempt, step_id)
                .join(OBLIGATION_FILE),
        )
    }

    pub fn prepare_session_dir(
        &self,
        job_id: u32,
        run_attempt: u32,
        step_id: spur_core::step::StepId,
    ) -> io::Result<PathBuf> {
        let state_dir = self.root.parent().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "runtime root has no state directory",
            )
        })?;
        if !state_dir.exists() {
            fs::create_dir_all(state_dir)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(state_dir, fs::Permissions::from_mode(0o700))?;
            }
        }
        create_private_dir(&self.root)?;
        let session_dir = self.session_dir(job_id, run_attempt, step_id);
        create_private_dir(&session_dir)?;
        Ok(session_dir)
    }

    pub fn publish(&self, descriptor: &StepdDescriptor) -> io::Result<()> {
        let session_dir = self.prepare_session_dir(
            descriptor.job_id,
            descriptor.run_attempt,
            descriptor.step_id,
        )?;
        let temporary_path =
            session_dir.join(format!("{DESCRIPTOR_FILE}.{}.tmp", uuid::Uuid::new_v4()));
        let descriptor_path = session_dir.join(DESCRIPTOR_FILE);
        let contents = serde_json::to_vec(descriptor).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("serialize runtime descriptor: {error}"),
            )
        })?;
        let mut temporary = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary_path)?;
        temporary.write_all(&contents)?;
        temporary.sync_all()?;
        drop(temporary);
        fs::rename(&temporary_path, descriptor_path)?;
        fs::File::open(session_dir)?.sync_all()
    }

    pub(crate) fn discover_live(&self) -> io::Result<DiscoveredStepds> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(DiscoveredStepds {
                    live: Vec::new(),
                    stale: Vec::new(),
                    rejected: Vec::new(),
                });
            }
            Err(error) => return Err(error),
        };

        let mut live = Vec::new();
        let mut stale = Vec::new();
        let mut rejected = Vec::new();
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    rejected.push((self.root.clone(), error.to_string()));
                    continue;
                }
            };
            let path = entry.path();
            if !entry.file_type()?.is_dir() {
                rejected.push((path, "stepd entry is not a directory".into()));
                continue;
            }

            match self.load_descriptor(&path) {
                Ok(descriptor) => match stepd_liveness(&descriptor) {
                    Ok(StepdLiveness::Live) => live.push(descriptor),
                    Ok(StepdLiveness::Stale) => stale.push(descriptor),
                    Err(error) => rejected.push((path, error.to_string())),
                },
                Err(error) => rejected.push((path, error.to_string())),
            }
        }

        Ok(DiscoveredStepds {
            live,
            stale,
            rejected,
        })
    }

    pub(crate) fn discover_unacknowledged_completions(
        &self,
    ) -> io::Result<Vec<PendingStepdCompletion>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut completions = Vec::new();
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let session_dir = entry.path();
            let descriptor = match self.load_descriptor(&session_dir) {
                Ok(descriptor) => descriptor,
                Err(_) => continue,
            };
            if !matches!(stepd_liveness(&descriptor), Ok(StepdLiveness::Stale)) {
                continue;
            }
            let obligations = self.obligations(
                descriptor.job_id,
                descriptor.run_attempt,
                descriptor.step_id,
            );
            let mut observed_exit = None;
            let mut acknowledged = false;
            for obligation in obligations.read()? {
                match obligation {
                    StepdObligation::ExitObserved { exit_code, signal } => {
                        observed_exit = Some((exit_code, signal));
                        acknowledged = false;
                    }
                    StepdObligation::CompletionAcknowledged if observed_exit.is_some() => {
                        acknowledged = true;
                    }
                    StepdObligation::CompletionAcknowledged
                    | StepdObligation::ResourcesReleased => {}
                }
            }
            if let Some((exit_code, signal)) = observed_exit.filter(|_| !acknowledged) {
                completions.push(PendingStepdCompletion {
                    job_id: descriptor.job_id,
                    run_attempt: descriptor.run_attempt,
                    step_id: descriptor.step_id,
                    exit_code,
                    signal,
                });
            }
        }
        Ok(completions)
    }

    pub(crate) fn prune_finalized(&self) -> io::Result<usize> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error),
        };

        let mut pruned = 0;
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let session_dir = entry.path();
            let descriptor = match self.load_descriptor(&session_dir) {
                Ok(descriptor) => descriptor,
                Err(_) => continue,
            };
            let obligations = self.obligations(
                descriptor.job_id,
                descriptor.run_attempt,
                descriptor.step_id,
            );
            if prune_finalized_session(&session_dir, &obligations)? {
                pruned += 1;
            }
        }
        Ok(pruned)
    }

    pub(crate) fn acknowledge_completion(
        &self,
        completion: &PendingStepdCompletion,
    ) -> io::Result<()> {
        let obligations = self.obligations(
            completion.job_id,
            completion.run_attempt,
            completion.step_id,
        );
        obligations.append(&StepdObligation::CompletionAcknowledged)?;
        prune_finalized_session(
            &self.session_dir(
                completion.job_id,
                completion.run_attempt,
                completion.step_id,
            ),
            &obligations,
        )
        .map(|_| ())
    }

    pub(crate) fn observed_exit(
        &self,
        job_id: u32,
        run_attempt: u32,
        step_id: spur_core::step::StepId,
    ) -> io::Result<Option<(i32, i32)>> {
        let obligations = self.obligations(job_id, run_attempt, step_id).read()?;
        Ok(obligations
            .iter()
            .rev()
            .find_map(|obligation| match obligation {
                StepdObligation::ExitObserved { exit_code, signal } => Some((*exit_code, *signal)),
                _ => None,
            }))
    }

    pub(crate) fn load_descriptor(&self, session_dir: &Path) -> io::Result<StepdDescriptor> {
        let descriptor_path = session_dir.join(DESCRIPTOR_FILE);
        let contents = fs::read(&descriptor_path)?;
        let descriptor: StepdDescriptor = serde_json::from_slice(&contents).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid {}: {e}", descriptor_path.display()),
            )
        })?;
        if descriptor.format_version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported runtime descriptor version {}",
                    descriptor.format_version
                ),
            ));
        }
        if session_dir
            != self.session_dir(
                descriptor.job_id,
                descriptor.run_attempt,
                descriptor.step_id,
            )
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "runtime descriptor identity does not match its directory",
            ));
        }
        Ok(descriptor)
    }
}

#[cfg(unix)]
fn create_private_dir(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    match fs::create_dir(path) {
        Ok(()) => fs::set_permissions(path, fs::Permissions::from_mode(0o700)),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => verify_private_dir(path),
        Err(error) => Err(error),
    }
}

#[cfg(unix)]
fn verify_private_dir(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("runtime path {} is not a directory", path.display()),
        ));
    }
    if metadata.uid() != unsafe { libc::geteuid() } || metadata.permissions().mode() & 0o077 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("runtime directory {} is not private", path.display()),
        ));
    }
    Ok(())
}

pub(crate) fn process_start_ticks(pid: u32) -> io::Result<u64> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let (_, fields) = stat
        .rsplit_once(") ")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid /proc stat format"))?;
    fields
        .split_ascii_whitespace()
        .nth(19)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing process start time"))?
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub(crate) fn stepd_liveness(descriptor: &StepdDescriptor) -> io::Result<StepdLiveness> {
    match process_start_ticks(descriptor.pid) {
        Ok(start_ticks) if start_ticks == descriptor.process_start_ticks => Ok(StepdLiveness::Live),
        Ok(_) => Ok(StepdLiveness::Stale),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(StepdLiveness::Stale),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A nonexistent pid has no /proc entry, so stepd_liveness reports Stale
    // regardless of the stored ticks value — 0 is fine for that case.
    fn descriptor(job_id: u32, run_attempt: u32, pid: u32) -> StepdDescriptor {
        StepdDescriptor::new(
            job_id,
            run_attempt,
            spur_core::step::STEP_BATCH,
            pid,
            process_start_ticks(pid).unwrap_or(0),
            PathBuf::from("/run/spur/runtime.sock"),
            PathBuf::from("/sys/fs/cgroup/spur/test"),
        )
    }

    fn write_descriptor(store: &StepdStore, descriptor: &StepdDescriptor) -> PathBuf {
        let session_dir = store.session_dir(
            descriptor.job_id,
            descriptor.run_attempt,
            descriptor.step_id,
        );
        fs::create_dir_all(&session_dir).expect("create session directory");
        fs::write(
            session_dir.join(DESCRIPTOR_FILE),
            serde_json::to_vec(descriptor).expect("serialize descriptor"),
        )
        .expect("write descriptor");
        session_dir
    }

    fn launch_spec() -> StepdLaunchSpec {
        StepdLaunchSpec {
            job_id: 42,
            step_id: spur_core::step::STEP_BATCH,
            script: "true".into(),
            work_dir: "/tmp".into(),
            name: "runtime-test".into(),
            user: "spur".into(),
            node: "node-a".into(),
            environment: HashMap::new(),
            stdout_path: String::new(),
            stderr_path: String::new(),
            stdin_path: String::new(),
            cpus: 1,
            memory_mb: 0,
            gpu_devices: Vec::new(),
            cpu_ids: Vec::new(),
            open_mode: None,
            uid: nix::unistd::geteuid().as_raw(),
            gid: nix::unistd::getegid().as_raw(),
            partition: "default".into(),
            nodelist: "node-a".into(),
            memlock: StepdMemlock::Inherit,
            container: None,
            host_device_plan: None,
            container_rootfs_mode: None,
            hooks: spur_core::config::HooksConfig::default(),
            plugstack_path: String::new(),
            controller_addr: String::new(),
            reporting_node: String::new(),
            run_attempt: 1,
            capability: "test-capability".into(),
            allocation_only: false,
            pmix_multi_task: false,
        }
    }

    #[test]
    fn launch_spec_preserves_pmix_multi_task_execution_mode() {
        let mut spec = launch_spec();
        spec.pmix_multi_task = true;
        assert!(spec.into_launch_config().pmix_multi_task);
    }

    #[test]
    fn launch_spec_persists_gpu_injection_plan() {
        let mut spec = launch_spec();
        spec.gpu_devices = vec![3, 7];
        spec.host_device_plan = Some(spur_devices::inject::HostInjectionPlan {
            env: HashMap::from([("ROCR_VISIBLE_DEVICES".into(), "3,7".into())]),
            visible_devices: vec!["/dev/dri/renderD128".into()],
            device_paths: vec!["/dev/dri/renderD128".into()],
        });
        let restored: StepdLaunchSpec =
            serde_json::from_slice(&serde_json::to_vec(&spec).expect("encode launch spec"))
                .expect("decode launch spec");
        let config = restored.into_launch_config();
        assert_eq!(config.gpu_devices, vec![3, 7]);
        assert_eq!(
            config
                .host_device_plan
                .as_ref()
                .and_then(|plan| plan.env.get("ROCR_VISIBLE_DEVICES")),
            Some(&"3,7".to_string())
        );
    }

    #[test]
    fn legacy_launch_spec_deserializes_runtime_defaults() {
        let mut serialized = serde_json::to_value(launch_spec()).expect("encode launch spec");
        let fields = serialized
            .as_object_mut()
            .expect("launch spec must encode as an object");
        for field in [
            "step_id",
            "gpu_devices",
            "container",
            "host_device_plan",
            "container_rootfs_mode",
            "hooks",
            "plugstack_path",
            "controller_addr",
            "reporting_node",
            "run_attempt",
            "capability",
            "allocation_only",
            "pmix_multi_task",
        ] {
            fields.remove(field);
        }

        let restored: StepdLaunchSpec =
            serde_json::from_value(serialized).expect("decode legacy launch spec");
        assert_eq!(restored.step_id, spur_core::step::STEP_BATCH);
        assert!(restored.gpu_devices.is_empty());
        assert!(restored.container.is_none());
        assert!(restored.host_device_plan.is_none());
        assert!(restored.container_rootfs_mode.is_none());
        assert_eq!(
            serde_json::to_value(&restored.hooks).expect("encode default hooks"),
            serde_json::to_value(spur_core::config::HooksConfig::default())
                .expect("encode expected hooks")
        );
        assert!(restored.plugstack_path.is_empty());
        assert!(restored.controller_addr.is_empty());
        assert!(restored.reporting_node.is_empty());
        assert_eq!(restored.run_attempt, 0);
        assert!(restored.capability.is_empty());
        assert!(!restored.allocation_only);
        assert!(!restored.pmix_multi_task);
    }

    #[test]
    fn legacy_descriptor_deserializes_runtime_defaults() {
        let mut serialized =
            serde_json::to_value(descriptor(42, 1, std::process::id())).expect("encode descriptor");
        let fields = serialized
            .as_object_mut()
            .expect("descriptor must encode as an object");
        for field in ["step_id", "capability", "owner", "uid", "gid", "work_dir"] {
            fields.remove(field);
        }
        let restored: StepdDescriptor =
            serde_json::from_value(serialized).expect("decode legacy descriptor");
        assert_eq!(restored.step_id, spur_core::step::STEP_BATCH);
        assert!(restored.capability.is_empty());
        assert!(restored.owner.is_empty());
        assert_eq!(restored.uid, 0);
        assert_eq!(restored.gid, 0);
        assert!(restored.work_dir.is_empty());
    }

    #[test]
    fn reads_own_process_start_time() {
        assert!(process_start_ticks(std::process::id()).is_ok());
    }

    #[tokio::test]
    async fn sigkill_also_kills_the_job_cgroup() {
        let cgroup = tempfile::tempdir().expect("tempdir");
        std::fs::write(cgroup.path().join("cgroup.kill"), b"").expect("seed cgroup.kill");
        let mut command = tokio::process::Command::new("/bin/sh");
        command.arg("-c").arg("trap '' TERM; while :; do :; done");
        command.process_group(0);
        let child = command.spawn().expect("spawn managed job");
        let job = RunningJob::Managed {
            child,
            cgroup_path: Some(cgroup.path().to_path_buf()),
        };
        let session = Stepd::new(job, 83, 1, spur_core::step::STEP_BATCH);

        session
            .signal(nix::sys::signal::Signal::SIGTERM as i32)
            .await
            .expect("send SIGTERM");
        assert_eq!(
            std::fs::read(cgroup.path().join("cgroup.kill")).expect("read cgroup.kill"),
            b"",
            "SIGTERM must not trigger cgroup.kill"
        );

        session
            .signal(nix::sys::signal::Signal::SIGKILL as i32)
            .await
            .expect("send SIGKILL");
        assert_eq!(
            std::fs::read(cgroup.path().join("cgroup.kill")).expect("read cgroup.kill"),
            b"1",
            "SIGKILL must escalate through cgroup.kill"
        );
    }

    #[tokio::test]
    async fn allocation_only_signals_preserve_session_liveness() {
        let session = Stepd::new(
            RunningJob::AllocationOnly,
            82,
            1,
            spur_core::step::STEP_BATCH,
        );
        for signal in [
            nix::sys::signal::Signal::SIGSTOP,
            nix::sys::signal::Signal::SIGCONT,
            nix::sys::signal::Signal::SIGTERM,
        ] {
            session
                .signal(signal as i32)
                .await
                .expect("signal allocation-only stepd");
            assert!(
                session.snapshot().await.active,
                "{signal:?} must not end session"
            );
        }
    }

    #[test]
    fn discovers_only_live_identity_matched_sessions() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = StepdStore::new(temp.path());
        let pid = std::process::id();
        let live = descriptor(42, 3, pid);
        write_descriptor(&store, &live);
        let stale = descriptor(43, 1, 999_999);
        write_descriptor(&store, &stale);

        let discovered = store.discover_live().expect("discover live sessions");
        assert_eq!(discovered.live, vec![live]);
        assert_eq!(discovered.stale, vec![stale]);
        assert!(discovered.rejected.is_empty());
    }

    #[test]
    fn rejects_descriptor_with_mismatched_directory_identity() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = StepdStore::new(temp.path());
        let mismatched = descriptor(42, 3, std::process::id());
        let session_dir = store.session_dir(99, 1, spur_core::step::STEP_BATCH);
        fs::create_dir_all(&session_dir).expect("create session directory");
        fs::write(
            session_dir.join(DESCRIPTOR_FILE),
            serde_json::to_vec(&mismatched).expect("serialize descriptor"),
        )
        .expect("write descriptor");

        let discovered = store.discover_live().expect("discover live sessions");
        assert!(discovered.live.is_empty());
        assert!(discovered.stale.is_empty());
        assert_eq!(discovered.rejected.len(), 1);
    }

    #[test]
    fn publish_writes_a_private_reconnectable_descriptor() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = StepdStore::new(temp.path());
        let descriptor = descriptor(7, 2, std::process::id());
        store.publish(&descriptor).expect("publish descriptor");

        let session_dir = store.session_dir(7, 2, spur_core::step::STEP_BATCH);
        let loaded = store
            .load_descriptor(&session_dir)
            .expect("load published descriptor");
        assert_eq!(loaded, descriptor);
    }

    #[test]
    fn prepares_a_missing_configured_state_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let state_dir = temp.path().join("does-not-exist-yet");
        let store = StepdStore::new(&state_dir);
        let session_dir = store
            .prepare_session_dir(1, 1, spur_core::step::STEP_BATCH)
            .expect("prepare session dir");
        assert!(session_dir.exists());
    }

    #[test]
    fn obligations_preserve_exit_and_acknowledgement_order() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = StepdStore::new(temp.path());
        store
            .prepare_session_dir(1, 1, spur_core::step::STEP_BATCH)
            .expect("prepare session directory");
        let obligations = store.obligations(1, 1, spur_core::step::STEP_BATCH);
        obligations
            .append(&StepdObligation::ExitObserved {
                exit_code: 0,
                signal: 0,
            })
            .expect("append exit observed");
        obligations
            .append(&StepdObligation::CompletionAcknowledged)
            .expect("append acknowledged");
        let read = obligations.read().expect("read obligations");
        assert_eq!(read.len(), 2);
    }

    #[test]
    fn resource_release_obligation_is_idempotent() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = StepdStore::new(temp.path());
        let mut descriptor = descriptor(9, 1, std::process::id());
        descriptor.socket_path = store
            .session_dir(9, 1, spur_core::step::STEP_BATCH)
            .join("runtime.sock");
        store.publish(&descriptor).expect("publish descriptor");
        record_resources_released(&descriptor).expect("record resources released");
        record_resources_released(&descriptor).expect("record resources released again");
        let obligations = store
            .obligations(9, 1, spur_core::step::STEP_BATCH)
            .read()
            .expect("read obligations");
        assert_eq!(
            obligations
                .iter()
                .filter(|o| matches!(o, StepdObligation::ResourcesReleased))
                .count(),
            1
        );
    }

    #[test]
    fn stale_exit_without_acknowledgement_is_recoverable() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = StepdStore::new(temp.path());
        let descriptor = descriptor(11, 1, 999_999);
        store.publish(&descriptor).expect("publish descriptor");
        store
            .obligations(11, 1, spur_core::step::STEP_BATCH)
            .append(&StepdObligation::ExitObserved {
                exit_code: 1,
                signal: 0,
            })
            .expect("append exit observed");

        let pending = store
            .discover_unacknowledged_completions()
            .expect("discover unacknowledged completions");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].job_id, 11);
    }

    #[test]
    fn observed_exit_remains_available_after_completion_acknowledgement() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = StepdStore::new(temp.path());
        let mut descriptor = descriptor(5, 1, std::process::id());
        descriptor.socket_path = store
            .session_dir(5, 1, spur_core::step::STEP_BATCH)
            .join("runtime.sock");
        store.publish(&descriptor).expect("publish descriptor");
        store
            .obligations(5, 1, spur_core::step::STEP_BATCH)
            .append(&StepdObligation::ExitObserved {
                exit_code: 7,
                signal: 0,
            })
            .expect("append exit observed");
        assert_eq!(
            store
                .observed_exit(5, 1, spur_core::step::STEP_BATCH)
                .expect("observed exit"),
            Some((7, 0))
        );
        store
            .acknowledge_completion(&PendingStepdCompletion {
                job_id: 5,
                run_attempt: 1,
                step_id: spur_core::step::STEP_BATCH,
                exit_code: 7,
                signal: 0,
            })
            .expect("acknowledge completion");
        assert_eq!(
            store
                .observed_exit(5, 1, spur_core::step::STEP_BATCH)
                .expect("observed exit"),
            Some((7, 0))
        );
    }

    #[test]
    fn finalized_attempt_is_pruned_only_after_acknowledgement_and_resource_release() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = StepdStore::new(temp.path());
        let descriptor = descriptor(13, 1, std::process::id());
        store.publish(&descriptor).expect("publish descriptor");
        let obligations = store.obligations(13, 1, spur_core::step::STEP_BATCH);
        obligations
            .append(&StepdObligation::ExitObserved {
                exit_code: 0,
                signal: 0,
            })
            .expect("append exit observed");
        assert_eq!(store.prune_finalized().expect("prune"), 0);
        obligations
            .append(&StepdObligation::CompletionAcknowledged)
            .expect("append acknowledged");
        assert_eq!(store.prune_finalized().expect("prune"), 0);
        obligations
            .append(&StepdObligation::ResourcesReleased)
            .expect("append released");
        assert_eq!(store.prune_finalized().expect("prune"), 1);
    }

    #[test]
    fn startup_pruning_keeps_unacknowledged_and_unreleased_attempts() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = StepdStore::new(temp.path());
        let descriptor = descriptor(17, 1, std::process::id());
        store.publish(&descriptor).expect("publish descriptor");
        store
            .obligations(17, 1, spur_core::step::STEP_BATCH)
            .append(&StepdObligation::ExitObserved {
                exit_code: 0,
                signal: 0,
            })
            .expect("append exit observed");
        assert_eq!(store.prune_finalized().expect("prune"), 0);
        assert!(store
            .session_dir(17, 1, spur_core::step::STEP_BATCH)
            .exists());
    }

    #[test]
    fn hello_requires_compatible_version_capability_attempt_and_step() {
        let descriptor = descriptor(1, 4, std::process::id());
        let step_id = spur_core::step::STEP_BATCH;
        assert!(matches!(
            validate_hello(&descriptor, "cap", "cap", PROTOCOL_VERSION, 4, step_id),
            StepdResponse::Hello { .. }
        ));
        assert!(matches!(
            validate_hello(&descriptor, "cap", "cap", PROTOCOL_VERSION + 1, 4, step_id),
            StepdResponse::Rejected { .. }
        ));
        assert!(matches!(
            validate_hello(&descriptor, "wrong", "cap", PROTOCOL_VERSION, 4, step_id),
            StepdResponse::Rejected { .. }
        ));
        assert!(matches!(
            validate_hello(&descriptor, "cap", "cap", PROTOCOL_VERSION, 1, step_id),
            StepdResponse::Rejected { .. }
        ));
        assert!(matches!(
            validate_hello(
                &descriptor,
                "cap",
                "cap",
                PROTOCOL_VERSION,
                4,
                spur_core::step::STEP_INTERACTIVE
            ),
            StepdResponse::Rejected { .. }
        ));
    }

    #[tokio::test]
    async fn unix_socket_hello_authenticates_before_returning_connection() {
        let temp = tempfile::tempdir().expect("tempdir");
        let socket_path = temp.path().join("runtime.sock");
        let listener = UnixListener::bind(&socket_path).expect("bind runtime socket");
        let descriptor = StepdDescriptor::new(
            1,
            1,
            spur_core::step::STEP_BATCH,
            std::process::id(),
            process_start_ticks(std::process::id()).expect("start ticks"),
            socket_path.clone(),
            PathBuf::new(),
        );
        let expected_capability = descriptor.capability.clone();
        let accept_descriptor = descriptor.clone();
        let server = tokio::spawn(async move {
            accept_hello(&listener, &accept_descriptor, &expected_capability).await
        });
        let mut client = UnixStream::connect(&socket_path)
            .await
            .expect("connect to runtime socket");
        let hello = StepdRequest::Hello {
            protocol_version: PROTOCOL_VERSION,
            capability: descriptor.capability.clone(),
            spurd_instance_id: "spurd-a".into(),
            run_attempt: descriptor.run_attempt,
            step_id: descriptor.step_id,
        };
        client
            .write_all(format!("{}\n", serde_json::to_string(&hello).unwrap()).as_bytes())
            .await
            .expect("write hello");
        let (_, spurd_instance_id) = server.await.expect("server task").expect("accept hello");
        assert_eq!(spurd_instance_id, "spurd-a");
    }

    #[tokio::test]
    async fn incomplete_hello_does_not_block_a_later_control_connection() {
        let temp = tempfile::tempdir().expect("tempdir");
        let socket_path = temp.path().join("runtime.sock");
        let listener = UnixListener::bind(&socket_path).expect("bind socket");
        let mut descriptor = descriptor(42, 3, std::process::id());
        descriptor.socket_path = socket_path;
        let session = Arc::new(Stepd::new(
            RunningJob::AllocationOnly,
            42,
            3,
            spur_core::step::STEP_BATCH,
        ));
        let supervisor = tokio::spawn(run_supervisor(listener, descriptor.clone(), session));

        let _partial = UnixStream::connect(&descriptor.socket_path)
            .await
            .expect("connect incomplete client");
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }

        let query_descriptor = descriptor.clone();
        let query =
            tokio::spawn(async move { query_state(&query_descriptor, "agent-1".into()).await });
        for _ in 0..64 {
            if query.is_finished() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            query.is_finished(),
            "a partial hello must not prevent later control connections"
        );
        assert!(
            query
                .await
                .expect("query task")
                .expect("query state")
                .active
        );

        supervisor.abort();
        let _ = supervisor.await;
    }

    #[tokio::test]
    async fn control_loop_reports_live_state_and_records_teardown() {
        let (server_stream, client_stream) = UnixStream::pair().expect("socket pair");
        let session = Stepd::new(
            RunningJob::AllocationOnly,
            42,
            3,
            spur_core::step::STEP_BATCH,
        );
        let server = tokio::spawn(async move { serve_control(server_stream, &session).await });
        let (reader, mut writer) = client_stream.into_split();
        for request in [StepdRequest::QueryState, StepdRequest::BeginTeardown] {
            writer
                .write_all(
                    format!(
                        "{}\n",
                        serde_json::to_string(&request).expect("encode request")
                    )
                    .as_bytes(),
                )
                .await
                .expect("write request");
        }
        drop(writer);
        let mut reader = BufReader::new(reader);
        let mut state = String::new();
        reader.read_line(&mut state).await.expect("read state");
        assert!(matches!(
            serde_json::from_str::<StepdResponse>(&state).expect("decode state"),
            StepdResponse::State { active: true, .. }
        ));
        let mut acknowledged = String::new();
        reader
            .read_line(&mut acknowledged)
            .await
            .expect("read teardown acknowledgement");
        assert_eq!(
            serde_json::from_str::<StepdResponse>(&acknowledged).expect("decode acknowledgement"),
            StepdResponse::Acknowledged
        );
        server.await.expect("server task").expect("serve control");
    }

    #[tokio::test]
    async fn allocation_signals_do_not_end_the_stepd() {
        let (server_stream, client_stream) = UnixStream::pair().expect("socket pair");
        let session = Arc::new(Stepd::new(
            RunningJob::AllocationOnly,
            42,
            3,
            spur_core::step::STEP_BATCH,
        ));
        let server_session = session.clone();
        let server =
            tokio::spawn(async move { serve_control(server_stream, &server_session).await });
        let (reader, mut writer) = client_stream.into_split();
        writer
            .write_all(
                format!(
                    "{}\n",
                    serde_json::to_string(&StepdRequest::SignalAllocation {
                        signal: nix::sys::signal::Signal::SIGSTOP as i32,
                    })
                    .expect("encode signal request")
                )
                .as_bytes(),
            )
            .await
            .expect("write signal request");
        drop(writer);
        let mut reader = BufReader::new(reader);
        let mut response = String::new();
        reader
            .read_line(&mut response)
            .await
            .expect("read response");
        assert_eq!(
            serde_json::from_str::<StepdResponse>(&response).expect("decode response"),
            StepdResponse::Acknowledged
        );
        server.await.expect("server task").expect("serve control");
        assert!(session.snapshot().await.active);
    }

    #[tokio::test]
    async fn shutdown_ends_an_allocation_stepd() {
        let (server_stream, client_stream) = UnixStream::pair().expect("socket pair");
        let session = Arc::new(Stepd::new(
            RunningJob::AllocationOnly,
            42,
            3,
            spur_core::step::STEP_BATCH,
        ));
        let server_session = session.clone();
        let server =
            tokio::spawn(async move { serve_control(server_stream, &server_session).await });
        let (reader, mut writer) = client_stream.into_split();
        writer
            .write_all(
                format!(
                    "{}\n",
                    serde_json::to_string(&StepdRequest::Shutdown).expect("encode shutdown")
                )
                .as_bytes(),
            )
            .await
            .expect("write shutdown request");
        drop(writer);
        let mut reader = BufReader::new(reader);
        let mut response = String::new();
        reader
            .read_line(&mut response)
            .await
            .expect("read response");
        assert_eq!(
            serde_json::from_str::<StepdResponse>(&response).expect("decode response"),
            StepdResponse::Acknowledged
        );
        server.await.expect("server task").expect("serve control");
        session
            .poll_completion()
            .await
            .expect("poll teardown completion");
        assert!(!session.snapshot().await.active);
    }
}
