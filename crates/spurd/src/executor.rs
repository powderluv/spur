// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::ffi::CString;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{bail, Context};
use nix::sys::signal::{self, SaFlags, SigAction, SigHandler, SigSet, Signal};
use nix::unistd::Pid;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::{debug, info, warn};

use spur_core::config::{CgroupConfig, CgroupLimits, MemlockLimit};
use spur_core::job::JobId;
use spur_spank::{SpankContext, SpankHandle, SpankHost};

use crate::device_cgroup;

/// Typed launch errors so callers can distinguish a broken node from a job that
/// simply cannot run here.
pub enum LaunchError {
    PrologFailed(anyhow::Error),
    /// The node itself cannot host work: an I/O failure in spurd's own spool
    /// tree, so every subsequent job will fail identically.
    NodeFault(anyhow::Error),
    Other(anyhow::Error),
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `{:#}` renders the whole cause chain. Plain `{}` prints only the
        // outermost context, which would reduce a drain reason to "create job
        // spool dir" and drop the errno an operator needs to act on.
        match self {
            Self::PrologFailed(e) => write!(f, "prolog failed: {e:#}"),
            Self::NodeFault(e) => write!(f, "launch failed: {e:#}"),
            Self::Other(e) => write!(f, "{e:#}"),
        }
    }
}

impl LaunchError {
    /// Reason for the agent to drain itself, or `None` when the controller owns
    /// the decision. A prolog failure drains too, but the controller does it,
    /// because only the controller can pair the drain with the hold that stops
    /// the job walking the cluster.
    pub fn drain_reason(&self) -> Option<String> {
        match self {
            Self::NodeFault(_) => Some(self.to_string()),
            Self::PrologFailed(_) | Self::Other(_) => None,
        }
    }
}

impl From<anyhow::Error> for LaunchError {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e)
    }
}

/// True when the error chain carries a real OS-level I/O failure that the node
/// itself is responsible for.
///
/// An exclusion list, mirroring Slurm's "all others drain the node" default: the
/// spool tree is root-owned and every path under it is built from the job id
/// alone, so a submission cannot steer the errno. Requiring a real
/// `raw_os_error` keeps a plain `anyhow!("...")` out, and `EDQUOT` stays
/// excluded as a property of a user on a shared filesystem, not of the node.
fn is_node_fault_io_error(err: &anyhow::Error) -> bool {
    err.chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .any(is_node_fault_errno)
}

fn is_node_fault_errno(err: &std::io::Error) -> bool {
    matches!(err.raw_os_error(), Some(errno) if errno != libc::EDQUOT)
}

/// True when `dir` lives in the spool tree spurd owns, as opposed to the
/// world-writable temp fallback [`create_job_spool_dir`] drops to on a non-root
/// dev run. Only the owned tree may condemn a node: `/tmp` exhaustion is
/// something any single job can cause, so draining on it would let one runaway
/// job take the cluster down node by node.
fn is_node_owned_spool(dir: &Path) -> bool {
    dir.starts_with(SPOOL_ROOT)
}

/// Classify a failed write to a job's spool directory. An I/O failure under the
/// node's own spool root condemns the node; anything else is just this job's
/// problem.
///
/// Only spool writes may reach this. Writes to the job's `work_dir` must not use
/// it: that path is user-controlled and frequently a shared mount, where one user
/// filling their quota would otherwise drain every node in turn.
fn classify_spool_error(dir: &Path, err: anyhow::Error) -> LaunchError {
    if is_node_owned_spool(dir) && is_node_fault_io_error(&err) {
        LaunchError::NodeFault(err)
    } else {
        LaunchError::Other(err)
    }
}

use crate::container::ContainerConfig;

/// Cgroup root for slurmd-managed jobs.
const CGROUP_ROOT: &str = "/sys/fs/cgroup/spur";

/// Node-local spool root for spurd's per-job scratch (job script, namespace
/// wrapper). Deliberately off the user's work_dir so these root-side writes
/// never hit an NFS root_squash mount. Mirrors Slurm's SlurmdSpoolDir.
const SPOOL_ROOT: &str = "/var/spool/spur";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerLaunchConfig {
    pub config: ContainerConfig,
    pub rootfs: PathBuf,
}

/// Everything an agent needs to launch a job process on this node.
///
/// Groups the resolved execution parameters that come from multiple sources
/// (JobSpec, scheduler allocation, agent config) into a single value.
/// How the job's I/O is connected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LaunchIo {
    /// Traditional file-based stdout/stderr capture.
    #[default]
    File,
    /// PTY-backed: stdout/stderr/stdin all go through a pseudo-terminal.
    /// The master fd is returned in `LaunchResult::pty_master`.
    Pty,
}

pub struct JobLaunchConfig {
    pub job_id: JobId,
    pub run_attempt: u32,
    pub script: String,
    pub work_dir: String,
    /// Needed to expand `%x`/`%u`/`%N`/`%a`/`%A` in output paths as the controller does.
    pub name: String,
    pub user: String,
    pub node: String,
    pub array_job_id: Option<JobId>,
    pub array_task_id: Option<u32>,
    pub environment: HashMap<String, String>,
    pub stdout_path: String,
    pub stderr_path: String,
    pub stdin_path: String,
    pub cpus: u32,
    pub memory_mb: u64,
    pub gpu_devices: Vec<u32>,
    pub cpu_ids: Vec<u32>,
    pub open_mode: Option<String>,
    pub uid: u32,
    pub gid: u32,
    pub container: Option<ContainerLaunchConfig>,
    pub prolog_script: Option<String>,
    pub partition: String,
    pub nodelist: String,
    /// Registry-based device injection plan for host (non-container) jobs.
    pub host_device_plan: Option<spur_devices::inject::HostInjectionPlan>,
    /// RLIMIT_MEMLOCK to apply before exec (while still privileged).
    pub memlock: MemlockLimit,
    /// cgroup-v2 enforcement settings from `[cgroup]`.
    pub cgroup: CgroupConfig,
    /// I/O mode for the job.
    pub io_mode: LaunchIo,
    /// Direct multi-rank PMIx launch via a wrapper script (batch `--mpi=pmix`).
    pub pmix_multi_task: bool,
}

pub struct LaunchResult {
    pub job: RunningJob,
    pub stdout_path: String,
    pub stderr_path: String,
    /// Master fd of the PTY (only set when `io_mode == LaunchIo::Pty`).
    pub pty_master: Option<OwnedFd>,
    /// The job's cgroup, released to the caller now that the launch succeeded.
    /// The caller owns its removal from here on.
    pub cgroup_path: Option<PathBuf>,
}

/// Owns the resolved fds for a job's stdio, built once and consumed by both
/// the container (raw fork) and non-container (tokio::Command) spawn paths.
enum JobIo {
    File {
        stdin: Option<OwnedFd>,
        stdout: OwnedFd,
        stderr: OwnedFd,
    },
    Pty {
        master: OwnedFd,
        slave: OwnedFd,
    },
}

/// `Copy` snapshot of raw fds from a `JobIo`, safe to move into a `pre_exec`
/// closure or use in a raw-fork child. The parent retains ownership of the
/// underlying `OwnedFd`s so they stay valid through the fork boundary.
#[derive(Clone, Copy)]
pub(crate) enum JobIoRaw {
    File {
        stdin: Option<RawFd>,
        stdout: RawFd,
        stderr: RawFd,
    },
    Pty {
        master: RawFd,
        slave: RawFd,
    },
}

impl JobIo {
    fn raw(&self) -> JobIoRaw {
        match self {
            JobIo::File {
                stdin,
                stdout,
                stderr,
            } => JobIoRaw::File {
                stdin: stdin.as_ref().map(|fd| fd.as_raw_fd()),
                stdout: stdout.as_raw_fd(),
                stderr: stderr.as_raw_fd(),
            },
            JobIo::Pty { master, slave } => JobIoRaw::Pty {
                master: master.as_raw_fd(),
                slave: slave.as_raw_fd(),
            },
        }
    }

    /// Parent-side: extract the PTY master fd, dropping everything else.
    fn into_master(self) -> Option<OwnedFd> {
        match self {
            JobIo::Pty { master, .. } => Some(master),
            JobIo::File { .. } => None,
        }
    }
}

impl JobIoRaw {
    /// Wire this job's stdio into the current process.
    ///
    /// For File mode: dup2 stdin/stdout/stderr from the opened files.
    /// For PTY mode: setsid + TIOCSCTTY + dup2 slave + close master.
    ///
    /// # Safety
    /// Must only be called in a child process (post-fork or inside pre_exec).
    /// All operations are async-signal-safe.
    pub(crate) unsafe fn wire(self) -> std::io::Result<()> {
        match self {
            JobIoRaw::File {
                stdin,
                stdout,
                stderr,
            } => {
                if let Some(fd) = stdin {
                    crate::pty::checked_dup2(fd, libc::STDIN_FILENO)?;
                    if fd > 2 {
                        libc::close(fd);
                    }
                }
                crate::pty::checked_dup2(stdout, libc::STDOUT_FILENO)?;
                if stdout > 2 {
                    libc::close(stdout);
                }
                crate::pty::checked_dup2(stderr, libc::STDERR_FILENO)?;
                if stderr > 2 && stderr != stdout {
                    libc::close(stderr);
                }
                Ok(())
            }
            JobIoRaw::Pty { master, slave } => crate::pty::pty_pre_exec(slave, master),
        }
    }

    /// Wire stdin only (stdout/stderr stay as inherited pipe fds).
    ///
    /// Used for batch `--mpi=pmix` multi-rank wrappers: Open MPI's PMIx client
    /// initializes correctly when stdout is a pipe (srun parity) but falls back
    /// to singleton worlds when stdout is dup2'd to a regular file.
    ///
    /// # Safety
    /// Same constraints as [`Self::wire`].
    pub(crate) unsafe fn wire_stdin_only(self) -> std::io::Result<()> {
        match self {
            JobIoRaw::File { stdin, .. } => {
                if let Some(fd) = stdin {
                    crate::pty::checked_dup2(fd, libc::STDIN_FILENO)?;
                    if fd > 2 {
                        libc::close(fd);
                    }
                }
                Ok(())
            }
            JobIoRaw::Pty { .. } => self.wire(),
        }
    }
}

/// A running job process — either a tokio-managed child or a raw-forked container.
pub enum RunningJob {
    /// Non-container jobs managed by tokio::process::Child.
    Managed { child: tokio::process::Child },
    /// Container jobs: raw fork with optional pidfd for PID-recycling safety.
    Forked {
        pid: i32,
        /// Holds a kernel reference preventing PID recycling. None on kernels < 5.3.
        _pidfd: Option<OwnedFd>,
        reaped: bool,
    },
    /// Allocation registered without a batch process (standalone srun).
    AllocationOnly,
}

/// Split a finished process's wait status into (exit_code, signal).
/// Slurm parity: WIFEXITED -> (code, 0); WIFSIGNALED -> (0, sig).
pub fn decode_wait_status(status: nix::sys::wait::WaitStatus) -> (i32, i32) {
    match status {
        nix::sys::wait::WaitStatus::Exited(_, code) => (code, 0),
        nix::sys::wait::WaitStatus::Signaled(_, sig, _) => (0, sig as i32),
        _ => (-1, 0), // unreachable from try_wait (only Exited/Signaled reach here); -1 = shouldn't-happen sentinel
    }
}

/// Set RLIMIT_MEMLOCK in the current process. Best-effort: a non-root spurd
/// cannot raise the hard limit beyond what it inherited.
pub(crate) fn apply_memlock(limit: MemlockLimit) {
    let v = match limit {
        MemlockLimit::Inherit => return,
        MemlockLimit::Unlimited => libc::RLIM_INFINITY,
        MemlockLimit::Bytes(n) => n as libc::rlim_t,
    };
    let rl = libc::rlimit {
        rlim_cur: v,
        rlim_max: v,
    };
    if unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &rl) } == 0 {
        return;
    }
    // Non-root cannot raise hard limit. Fall back: raise soft to current hard.
    let mut current = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut current) } == 0 {
        let fallback = libc::rlimit {
            rlim_cur: current.rlim_max,
            rlim_max: current.rlim_max,
        };
        unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &fallback) };
    }
}

fn pidfd_open(pid: i32) -> std::io::Result<OwnedFd> {
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as RawFd;
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

impl RunningJob {
    pub fn managed(child: tokio::process::Child) -> Self {
        Self::Managed { child }
    }

    pub fn pid(&self) -> Option<u32> {
        match self {
            RunningJob::Managed { child, .. } => child.id(),
            RunningJob::Forked { pid, .. } => Some(*pid as u32),
            RunningJob::AllocationOnly => None,
        }
    }

    pub fn is_allocation_only(&self) -> bool {
        matches!(self, RunningJob::AllocationOnly)
    }

    /// Non-blocking check for process exit. Returns (exit_code, signal) if done.
    pub fn try_wait(&mut self) -> anyhow::Result<Option<(i32, i32)>> {
        match self {
            RunningJob::Managed { child, .. } => match child.try_wait() {
                Ok(Some(status)) => {
                    use std::os::unix::process::ExitStatusExt;
                    Ok(Some((
                        status.code().unwrap_or(0),
                        status.signal().unwrap_or(0),
                    )))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(e.into()),
            },
            RunningJob::Forked { pid, reaped, .. } => {
                if *reaped {
                    return Ok(None);
                }
                match nix::sys::wait::waitpid(
                    Pid::from_raw(*pid),
                    Some(nix::sys::wait::WaitPidFlag::WNOHANG),
                ) {
                    Ok(nix::sys::wait::WaitStatus::StillAlive) => Ok(None),
                    Ok(status @ nix::sys::wait::WaitStatus::Exited(_, _))
                    | Ok(status @ nix::sys::wait::WaitStatus::Signaled(_, _, _)) => {
                        *reaped = true;
                        Ok(Some(decode_wait_status(status)))
                    }
                    Ok(_) => Ok(None),
                    Err(e) => Err(e.into()),
                }
            }
            RunningJob::AllocationOnly => Ok(None),
        }
    }

    /// Send a signal to the running process.
    ///
    /// Managed jobs are spawned as their own process-group leader, so we signal
    /// the whole group (negative pid) to reach the batch shell and its children
    /// (e.g. an inner `sleep`), not just the tracked process.
    /// For container (Forked) jobs, signals the entire process subtree
    /// since the tracked PID is the intermediate parent and the actual
    /// workload runs as a grandchild inside a PID namespace.
    pub fn kill_signal(&self, sig: Signal) -> anyhow::Result<()> {
        match self {
            RunningJob::Managed { child, .. } => {
                if let Some(pid) = child.id() {
                    // Negative pid = the job's process group.
                    signal::kill(Pid::from_raw(-(pid as i32)), sig)?;
                }
                Ok(())
            }
            RunningJob::Forked { pid, reaped, .. } => {
                if *reaped {
                    return Ok(());
                }
                kill_process_tree(*pid, sig);
                Ok(())
            }
            RunningJob::AllocationOnly => Ok(()),
        }
    }
}

/// Launch a job script on this node.
///
/// If `container` is `Some`, the job runs inside a container via explicit
/// `fork()` + `container_init()` (namespace, mounts, pivot_root, priv drop).
/// Otherwise, it uses the standard `tokio::Command` path with optional
/// `build_namespace_wrapper()` for non-container namespace isolation.
pub async fn launch_job(
    cfg: &JobLaunchConfig,
    spank: Option<&SpankHost>,
) -> Result<LaunchResult, LaunchError> {
    // Run prolog before anything else
    if let Some(ref prolog) = cfg.prolog_script {
        let ctx = spur_core::hooks::HookContext {
            job_id: cfg.job_id,
            work_dir: cfg.work_dir.clone(),
            uid: cfg.uid,
            gid: cfg.gid,
            partition: cfg.partition.clone(),
            nodelist: cfg.nodelist.clone(),
            script_context: "prolog_slurmd".into(),
            gpu_devices: cfg.gpu_devices.clone(),
            cpus: cfg.cpus,
            memory_mb: cfg.memory_mb,
        };
        spur_core::hooks::run_hook(prolog, &ctx)
            .await
            .map_err(LaunchError::PrologFailed)?;
    }

    spawn_job_process(cfg, spank).await
}

async fn spawn_job_process(
    cfg: &JobLaunchConfig,
    spank: Option<&SpankHost>,
) -> Result<LaunchResult, LaunchError> {
    let JobLaunchConfig {
        job_id,
        run_attempt,
        ref script,
        ref work_dir,
        ref environment,
        ref stdout_path,
        ref stderr_path,
        ref stdin_path,
        cpus,
        memory_mb,
        gpu_devices: _,
        ref cpu_ids,
        ref open_mode,
        uid,
        gid,
        ref container,
        ..
    } = *cfg;
    info!(job_id, work_dir, "launching job");

    // Set up cgroup for isolation
    let device_paths = allocated_device_paths(cfg.host_device_plan.as_ref());
    let cgroup_path = CgroupGuard(setup_cgroup(
        job_id,
        &cfg.cgroup,
        run_attempt,
        cpus,
        memory_mb,
        cpu_ids,
        device_paths,
    )?);

    // Ensure work_dir exists on this node (the submitted path may only exist on the submitting
    // node); falls back to a per-job scratch directory when it can't be created here.
    let effective_work_dir = resolve_effective_work_dir(job_id, run_attempt, work_dir, uid, gid);
    let work_dir = effective_work_dir.as_str();

    // The directive is per job, so it comes from the job's environment. Reading
    // the daemon's own env instead would only ever see a cluster-wide value.
    let script = match environment.get("SPUR_BURST_BUFFER") {
        Some(bb) if !bb.is_empty() => wrap_with_burst_buffer(script, bb),
        _ => script.to_string(),
    };
    let script = script.as_str();

    // Script + wrapper live in the node-local spool dir, not work_dir (see
    // SPOOL_ROOT), so root-side writes survive NFS root_squash work_dirs.
    let spool_dir = create_job_spool_dir(job_id, uid, gid)?;
    let script_path = spool_dir.join("spur_job.sh");
    write_job_scratch(&script_path, script, uid, gid)
        .context("failed to write job script")
        .map_err(|e| classify_spool_error(&spool_dir, e))?;

    // Build resolved output paths (empty for PTY mode since output goes to the terminal).
    let (stdout_resolved, stderr_resolved) = if cfg.io_mode == LaunchIo::Pty {
        ("/dev/null".to_string(), "/dev/null".to_string())
    } else {
        (
            resolve_output_path(cfg, work_dir, stdout_path),
            resolve_output_path(cfg, work_dir, stderr_path),
        )
    };

    // Build JobIo: a single object owning the fds for either file or PTY mode.
    let job_io = match cfg.io_mode {
        LaunchIo::Pty => {
            let (master, slave) = crate::pty::openpty_with_winsize(None).context("PTY openpty")?;
            JobIo::Pty { master, slave }
        }
        LaunchIo::File => {
            let stdin_resolved = if stdin_path.is_empty() {
                None
            } else {
                let r = resolve_output_path(cfg, work_dir, stdin_path);
                if r == stdout_resolved || r == stderr_resolved {
                    return Err(anyhow::anyhow!(
                        "stdin path {} overlaps with an output path; this would truncate the input",
                        r
                    )
                    .into());
                }
                Some(r)
            };

            let use_append = open_mode
                .as_deref()
                .map(|m| m.eq_ignore_ascii_case("append"))
                .unwrap_or(false);

            let (out, err) =
                open_job_output(uid, gid, use_append, &stdout_resolved, &stderr_resolved)
                    .context("failed to open job output files")?;

            let stdin_fd = match stdin_resolved {
                None => None,
                Some(ref resolved) => {
                    if uid > 0 {
                        use std::os::unix::fs::MetadataExt;
                        let meta = std::fs::metadata(resolved)
                            .with_context(|| format!("stdin file not found: {}", resolved))?;
                        let (fuid, fgid, mode) = (meta.uid(), meta.gid(), meta.mode());
                        let readable = (fuid == uid && mode & 0o400 != 0)
                            || (fgid == gid && mode & 0o040 != 0)
                            || (mode & 0o004 != 0);
                        if !readable {
                            return Err(anyhow::anyhow!(
                                "stdin file {} is not readable by uid {}",
                                resolved,
                                uid
                            )
                            .into());
                        }
                    }
                    let f = std::fs::File::open(resolved)
                        .with_context(|| format!("failed to open stdin file: {}", resolved))?;
                    Some(OwnedFd::from(f))
                }
            };

            JobIo::File {
                stdin: stdin_fd,
                stdout: OwnedFd::from(out),
                stderr: OwnedFd::from(err),
            }
        }
    };

    let mut env = environment.clone();

    if cfg.pmix_multi_task {
        crate::mpi_plugin::strip_launcher_mpi_env(&mut env);
    }

    // GPU isolation via registry-based device injection plan.
    if let Some(ref plan) = cfg.host_device_plan {
        for (key, value) in &plan.env {
            env.insert(key.clone(), value.clone());
        }
    }

    // Environment-based CPU/thread limiting — works even without cgroups.
    // Well-behaved applications (OpenMP, MKL, PyTorch, etc.) read these.
    if !cfg.pmix_multi_task {
        env.insert("OMP_NUM_THREADS".into(), cpus.to_string());
        env.insert("MKL_NUM_THREADS".into(), cpus.to_string());
        env.insert("OPENBLAS_NUM_THREADS".into(), cpus.to_string());
        env.insert("VECLIB_MAXIMUM_THREADS".into(), cpus.to_string());
        env.insert("NUMEXPR_NUM_THREADS".into(), cpus.to_string());
    }

    // Run SPANK Init/TaskInit against a handle seeded with the assembled env,
    // then fold plugin edits back so both the container and command paths pick
    // them up. Hooks run in the spurd (root) process, not the forked task.
    if let Some(spank) = spank {
        if !cfg.pmix_multi_task {
            let context = SpankContext {
                job_id,
                uid,
                gid,
                ..Default::default()
            };
            let mut handle = SpankHandle::new(context, env);
            for hook in [spur_spank::SpankHook::Init, spur_spank::SpankHook::TaskInit] {
                if let Err(e) = spank.invoke_hook(hook, &mut handle) {
                    warn!(job_id, error = %e, "SPANK hook failed");
                }
            }
            env = handle.env;
        }
    }

    // Container jobs: use explicit fork() + container_init() instead of bash wrapper.
    if let Some(ctn) = container {
        if !stdin_path.is_empty() && matches!(job_io, JobIo::File { .. }) {
            warn!(
                job_id,
                "stdin redirection is not supported for container jobs, ignoring"
            );
        }
        let (job, pty_master) = launch_container_job(cfg, ctn, &env, job_io, &cgroup_path).await?;
        return Ok(LaunchResult {
            job,
            stdout_path: stdout_resolved,
            stderr_path: stderr_resolved,
            pty_master,
            cgroup_path: cgroup_path.into_inner(),
        });
    }

    // --- Non-container jobs: existing tokio::Command path ---

    // Issue #99: If root, wrap job with namespace isolation.
    // Batch `--mpi=pmix` multi-rank wrappers must stay in the host mount/PID
    // namespace so Open MPI's PMIx client can reach spurd's embedded server
    // (same as standalone `srun` via `run_command`, which never uses unshare).
    let use_namespaces = nix::unistd::geteuid().is_root() && !cfg.pmix_multi_task;
    let (launch_cmd, launch_args) = if use_namespaces {
        let wrapper_path = spool_dir.join("spur_ns.sh");
        let visible_devices = cfg
            .host_device_plan
            .as_ref()
            .map(|p| p.visible_devices.as_slice())
            .unwrap_or(&[]);
        let wrapper = build_namespace_wrapper(uid, gid, visible_devices, &script_path);
        write_job_scratch(&wrapper_path, &wrapper, uid, gid)
            .map_err(|e| classify_spool_error(&spool_dir, e))?;
        debug!(job_id, "namespace isolation wrapper created");
        (
            "/usr/bin/unshare".to_string(),
            vec![
                "--pid".into(),
                "--mount".into(),
                "--fork".into(),
                "/bin/bash".into(),
                wrapper_path.to_string_lossy().to_string(),
            ],
        )
    } else {
        (
            "/bin/bash".to_string(),
            vec![script_path.to_string_lossy().to_string()],
        )
    };

    // Launch the process
    let piped_mpi_stdio = cfg.pmix_multi_task && cfg.io_mode == LaunchIo::File;
    let mut cmd = Command::new(&launch_cmd);
    // Always its own process group (run_command does the same for pmix step
    // launches) so signal()/kill_signal's group-kill reaches the whole job
    // regardless of PMIx — only namespace isolation is pmix-conditional above.
    cmd.args(&launch_args)
        .current_dir(work_dir)
        .envs(&env)
        .process_group(0);
    if piped_mpi_stdio {
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
    } else {
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
    }

    // Reset signal dispositions to default before exec. spurd is launched in the
    // background (SIGINT/SIGQUIT/SIGHUP set to SIG_IGN), and a child inherits that
    // ignore mask — which would make a job's own `kill -INT $$` a no-op and break
    // Slurm-parity signal reporting (e.g. SIGINT -> RaisedSignal:2). The job must
    // start with default handlers.
    unsafe {
        cmd.pre_exec(|| {
            // Use sigaction (async-signal-safe) rather than signal() to reset
            // dispositions; pre_exec runs post-fork in a multi-threaded process.
            let dfl = SigAction::new(SigHandler::SigDfl, SaFlags::empty(), SigSet::empty());
            for sig in [
                Signal::SIGINT,
                Signal::SIGQUIT,
                Signal::SIGHUP,
                Signal::SIGPIPE,
            ] {
                let _ = signal::sigaction(sig, &dfl);
            }
            Ok(())
        });
    }

    // RLIMIT_MEMLOCK: raise before privilege drop so RDMA/NCCL ibv_reg_mr works.
    let memlock = cfg.memlock;
    unsafe {
        cmd.pre_exec(move || {
            apply_memlock(memlock);
            Ok(())
        });
    }

    // Join pre-exec, not parent-side after spawn: under `unshare --fork` a
    // parent-side move races the fork and misses the workload's cgroup.
    let cgroup_procs = cgroup_path
        .path()
        .and_then(|p| CString::new(p.join("cgroup.procs").as_os_str().as_bytes()).ok());
    // fd 2 is redirected to the job's stdio before pre_exec runs, so hand the
    // child a dup of spurd's stderr (CLOEXEC) to report a join failure.
    let mut cgroup_log_fd: RawFd = -1;
    if let Some(procs) = cgroup_procs {
        cgroup_log_fd = dup_cloexec(libc::STDERR_FILENO);
        let log_fd = cgroup_log_fd;
        unsafe {
            cmd.pre_exec(move || {
                // Best-effort here; `required` is enforced parent-side below via cgroup_has_pid.
                let _ = join_cgroup_self(&procs, log_fd);
                Ok(())
            });
        }
    }

    // Issue #99, #107: Run job as the submitting user (not root).
    // Must set supplementary groups (video, render) so the process can
    // access GPU device nodes.
    //
    // Issue #128: when use_namespaces is true, the wrapper handles the priv
    // drop *after* unshare runs (via setpriv). Dropping priv here would cause
    // unshare(2) to fail with EPERM since the unprivileged user lacks
    // CAP_SYS_ADMIN.
    if !use_namespaces {
        if let Some(pd) = crate::privdrop::PrivDrop::resolve_if_needed(uid, gid) {
            unsafe {
                cmd.pre_exec(move || {
                    pd.apply()
                        .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;
                    Ok(())
                });
            }
            debug!(
                job_id,
                uid, gid, "job will run as non-root user with supplementary groups"
            );
        }
    }

    // Issue #99: Apply seccomp-BPF syscall filter (opt-in via SPUR_SECCOMP=1).
    let enable_seccomp = std::env::var("SPUR_SECCOMP")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);
    if enable_seccomp {
        unsafe {
            cmd.pre_exec(|| {
                if let Err(e) = crate::seccomp::apply_seccomp_filter() {
                    eprintln!("spur: seccomp filter not applied: {e}");
                }
                Ok(())
            });
        }
    }

    // Issue #99: Apply Landlock filesystem restrictions (opt-in via SPUR_LANDLOCK=1).
    let work_dir_for_landlock = work_dir.to_string();
    let enable_landlock = std::env::var("SPUR_LANDLOCK")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);
    if enable_landlock {
        unsafe {
            cmd.pre_exec(move || {
                if let Err(e) = crate::landlock::apply_landlock_rules(&work_dir_for_landlock) {
                    eprintln!("spur: landlock not applied: {e}");
                }
                Ok(())
            });
        }
    }

    // Wire job I/O (file dup2 or PTY setsid+TIOCSCTTY+dup2) in the child.
    let raw_io = job_io.raw();
    let wire_stdin_only = piped_mpi_stdio;
    unsafe {
        cmd.pre_exec(move || {
            if wire_stdin_only {
                raw_io.wire_stdin_only()
            } else {
                raw_io.wire()
            }
        });
    }

    let spawn_result = cmd.spawn();
    if cgroup_log_fd >= 0 {
        unsafe {
            libc::close(cgroup_log_fd);
        }
    }
    let mut child = spawn_result.context("failed to spawn job process")?;

    if piped_mpi_stdio {
        let shared = stderr_resolved == stdout_resolved;
        let use_append = open_mode
            .as_deref()
            .map(|m| m.eq_ignore_ascii_case("append"))
            .unwrap_or(false);
        spawn_mpi_stdio_drains(
            child.stdout.take(),
            child.stderr.take(),
            MpiStdioDrainOpts {
                uid,
                gid,
                stdout_path: &stdout_resolved,
                stderr_path: &stderr_resolved,
                shared,
                use_append,
            },
        );
    }

    // Drop the slave fd immediately so the master gets EOF when the child exits.
    let pty_master = job_io.into_master();

    // The child joined its own cgroup pre-exec; confirm it landed so `required`
    // can refuse a job that would otherwise run outside every limit.
    if cfg.cgroup.required {
        if let (Some(cgroup), Some(pid)) = (cgroup_path.path(), child.id()) {
            if !cgroup_has_pid(cgroup, pid) {
                // Reaps as well as kills, so the guard finds the cgroup empty.
                let _ = child.kill().await;
                return Err(anyhow::anyhow!(
                    "[cgroup] required but the job did not join its cgroup"
                )
                .into());
            }
        }
    }

    debug!(
        job_id,
        pid = child.id(),
        script = %script_path.display(),
        "job process spawned"
    );

    Ok(LaunchResult {
        job: RunningJob::Managed { child },
        stdout_path: stdout_resolved,
        stderr_path: stderr_resolved,
        pty_master,
        cgroup_path: cgroup_path.into_inner(),
    })
}

/// Render a job's cgroup-v2 control files as (filename, content) pairs. Pure so
/// the layout is testable without a cgroupfs; values come from `limits_for`.
fn cgroup_limit_files(limits: &CgroupLimits) -> Vec<(&'static str, String)> {
    let mut files: Vec<(&'static str, String)> = Vec::new();
    if let Some(quota) = limits.cpu_quota_us {
        files.push(("cpu.max", format!("{} {}", quota, limits.cpu_period_us)));
    }
    if let Some(bytes) = limits.memory_max_bytes {
        files.push(("memory.max", bytes.to_string()));
    }
    if let Some(bytes) = limits.memory_high_bytes {
        files.push(("memory.high", bytes.to_string()));
    }
    if let Some(bytes) = limits.swap_max_bytes {
        files.push(("memory.swap.max", bytes.to_string()));
    }
    if limits.oom_kill_job {
        files.push(("memory.oom.group", "1".to_string()));
    }
    files.push(("pids.max", limits.pids_max.to_string()));
    if !limits.cpuset_cpus.is_empty() {
        let cpuset = limits
            .cpuset_cpus
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        files.push(("cpuset.cpus", cpuset));
    }
    files
}

/// No injection plan means no devices were allocated, and must stay an empty list:
/// the filter allows exactly what it is handed.
fn allocated_device_paths(plan: Option<&spur_devices::inject::HostInjectionPlan>) -> &[String] {
    plan.map(|p| p.device_paths.as_slice()).unwrap_or(&[])
}

/// Set up a cgroups v2 hierarchy for a job.
// Keyed by attempt, not just job_id: a redispatch must never land in a
// still-occupied cgroup left by a not-yet-reaped prior attempt.
fn cgroup_path_for(cgroup_root: &Path, job_id: JobId, run_attempt: u32) -> PathBuf {
    cgroup_root.join(format!("job_{}_{}", job_id, run_attempt))
}

/// Reconstructs a job's cgroup path from its identity alone — usable even
/// when the session descriptor that would normally carry it is unreadable.
pub fn expected_cgroup_path(job_id: JobId, run_attempt: u32) -> PathBuf {
    cgroup_path_for(Path::new(CGROUP_ROOT), job_id, run_attempt)
}

pub(crate) fn setup_cgroup(
    job_id: JobId,
    cgroup: &CgroupConfig,
    run_attempt: u32,
    cpus: u32,
    memory_mb: u64,
    cpu_ids: &[u32],
    device_paths: &[String],
) -> anyhow::Result<Option<PathBuf>> {
    let Some(mut limits) = cgroup.limits_for(cpus, memory_mb, cpu_ids) else {
        debug!(job_id, "cgroup enforcement disabled by config");
        return Ok(None);
    };

    let cgroup_root = PathBuf::from(CGROUP_ROOT);
    let cgroup_path = cgroup_path_for(&cgroup_root, job_id, run_attempt);

    // Delegate controllers to children: in cgroup-v2 a child only gets
    // memory.*/cpu.*/pids.* files if the parent lists them in subtree_control;
    // without this the per-job memory limit is never enforced. Root failure fatal.
    // Steps below degrade to a warning by default; `required` turns the first
    // degrade into a refusal so a node that cannot enforce stops pretending.
    let mut degraded: Option<String> = None;
    let mut degrade = |what: &str| {
        if degraded.is_none() {
            degraded = Some(what.to_string());
        }
    };

    if let Err(e) = std::fs::create_dir_all(&cgroup_root) {
        if nix::unistd::geteuid().is_root() {
            anyhow::bail!("cgroup root creation failed as root: {}", e);
        }
        if cgroup.required {
            anyhow::bail!("[cgroup] required but the cgroup root is unavailable: {e}");
        }
        warn!(job_id, error = %e, "cgroup creation failed (not root), running without isolation");
        return Ok(None);
    }
    let subtree = cgroup_root.join("cgroup.subtree_control");
    for ctrl in ["+memory", "+cpu", "+pids", "+cpuset"] {
        if let Err(e) = std::fs::write(&subtree, ctrl) {
            warn!(job_id, controller = ctrl, error = %e, "failed to delegate cgroup controller");
            degrade(&format!("controller {ctrl} not delegated"));
        }
    }
    if let Err(e) = claim_cgroup_dir(&cgroup_path) {
        match classify_cgroup_claim_failure(
            &e,
            nix::unistd::geteuid().is_root(),
            cgroup.required,
            &cgroup_path,
        ) {
            CgroupClaim::Fatal(msg) => anyhow::bail!(msg),
            CgroupClaim::Degrade => {
                warn!(job_id, error = %e, "cgroup unavailable; job runs without isolation");
                return Ok(None);
            }
        }
    }

    // Core ids come from a synthesized 0..n range, which can name cores this
    // cgroup may not hold. Drop those rather than let the whole write fail.
    if !limits.cpuset_cpus.is_empty() {
        match std::fs::read_to_string(cgroup_root.join("cpuset.cpus.effective")) {
            Ok(effective) => {
                let permitted = permitted_cores(&limits.cpuset_cpus, effective.trim());
                if permitted.len() != limits.cpuset_cpus.len() {
                    warn!(
                        job_id,
                        allocated = ?limits.cpuset_cpus,
                        permitted = ?permitted,
                        effective = effective.trim(),
                        "some allocated cores lie outside the cgroup's permitted set"
                    );
                }
                limits.cpuset_cpus = permitted;
            }
            Err(e) => {
                warn!(job_id, error = %e, "cannot read cpuset.cpus.effective; writing the allocated set unchecked");
            }
        }
    }

    for (name, content) in cgroup_limit_files(&limits) {
        if let Err(e) = std::fs::write(cgroup_path.join(name), &content) {
            warn!(job_id, file = name, error = %e, "failed to write cgroup control file");
            degrade(&format!("{name} not applied"));
        }
    }

    // The cpuset is the only CPU bound once the quota is off, and a rejected write
    // reads back empty — i.e. inherit everything. Verify rather than trust.
    if cgroup.constrain_cores && !limits.cpuset_cpus.is_empty() {
        let applied = std::fs::read_to_string(cgroup_path.join("cpuset.cpus")).unwrap_or_default();
        let applied = parse_cpu_list(applied.trim());
        if applied.is_empty() {
            warn!(job_id, allocated = ?limits.cpuset_cpus, "cpuset not applied; job runs without a CPU bound");
            degrade("cpuset not applied");
        } else if applied != limits.cpuset_cpus.iter().copied().collect() {
            warn!(job_id, expected = ?limits.cpuset_cpus, applied = ?applied, "cpuset differs from the allocated cores");
            degrade("cpuset differs from the allocated cores");
        }
    } else if cgroup.constrain_cores {
        // No cpuset is written at all in this case, so the job is free to run on
        // every core on the node — the clamp above can empty a non-empty set.
        warn!(job_id, "no cores to pin; job runs without a CPU bound");
        degrade("no cores to pin");
    }

    // Attach before any of the job's processes join: the filter is consulted at
    // open(2), so a device a process already holds open stays readable regardless.
    if cgroup.constrain_devices {
        let paths = device_cgroup::device_paths_for_job(device_paths, &cgroup.extra_device_paths);
        let rules = device_cgroup::rules_for_device_paths(&paths, device_cgroup::stat_device_node);
        if let Err(e) = device_cgroup::install_device_filter(&cgroup_path, &rules) {
            warn!(job_id, error = %e, "device filter not installed; job runs without device isolation");
            degrade("device filter not installed");
        }
    }

    if let Some(reason) = degraded {
        if cgroup.required {
            let _ = std::fs::remove_dir(&cgroup_path);
            anyhow::bail!("[cgroup] required but enforcement is incomplete: {reason}");
        }
    }

    debug!(
        job_id,
        cpus,
        memory_mb,
        path = %cgroup_path.display(),
        "cgroup created"
    );

    Ok(Some(cgroup_path))
}

/// Parse a cgroup cpu list (`"0-3"`, `"0-1,4"`, `""`) into core ids.
fn parse_cpu_list(spec: &str) -> std::collections::BTreeSet<u32> {
    let mut ids = std::collections::BTreeSet::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((lo, hi)) => {
                if let (Ok(lo), Ok(hi)) = (lo.parse::<u32>(), hi.parse::<u32>()) {
                    ids.extend(lo..=hi);
                }
            }
            None => {
                if let Ok(id) = part.parse::<u32>() {
                    ids.insert(id);
                }
            }
        }
    }
    ids
}

/// Cores from `requested` the parent cgroup permits. A child cpuset must be a
/// subset: an id the parent lacks fails ERANGE and leaves the job unpinned.
fn permitted_cores(requested: &[u32], parent_effective: &str) -> Vec<u32> {
    let allowed = parse_cpu_list(parent_effective);
    requested
        .iter()
        .copied()
        .filter(|id| allowed.contains(id))
        .collect()
}

/// Join the calling process to a cgroup (pid → `cgroup.procs`), returning whether it landed.
/// Async-signal-safe (raw syscalls only) for post-fork pre-exec use; warns to `log_fd` on failure.
#[must_use]
fn join_cgroup_self(procs_path: &std::ffi::CStr, log_fd: RawFd) -> bool {
    let pid = unsafe { libc::getpid() };

    let mut buf = [0u8; 24];
    let mut i = buf.len();
    let mut n = pid.max(0) as u64;
    loop {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    let digits = &buf[i..];

    unsafe {
        let fd = libc::open(procs_path.as_ptr(), libc::O_WRONLY);
        if fd < 0 {
            warn_cgroup_join_failed(log_fd);
            return false;
        }
        let written = libc::write(fd, digits.as_ptr() as *const libc::c_void, digits.len());
        libc::close(fd);
        // cgroup.procs takes the whole pid in one write or errors; a short count is a failure.
        if written != digits.len() as isize {
            warn_cgroup_join_failed(log_fd);
            return false;
        }
    }
    true
}

/// Async-signal-safe warning for a failed cgroup join: a fixed message to
/// `log_fd`, or a no-op when it is negative.
fn warn_cgroup_join_failed(log_fd: RawFd) {
    if log_fd < 0 {
        return;
    }
    const MSG: &[u8] = b"spur: failed to join cgroup; job runs without resource limits\n";
    unsafe {
        libc::write(log_fd, MSG.as_ptr() as *const libc::c_void, MSG.len());
    }
}

/// Duplicate `fd` with CLOEXEC set, returning the new fd or -1. The copy is
/// usable in the pre-exec child but closes automatically at exec.
fn dup_cloexec(fd: RawFd) -> RawFd {
    unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) }
}

/// A child's pre-built cgroup join, ready to run from `pre_exec`. Built
/// parent-side because nothing between fork and exec may allocate.
pub(crate) struct CgroupJoin {
    procs: CString,
    log_fd: RawFd,
}

impl CgroupJoin {
    /// `None` when the job has no cgroup, which leaves the child uncontained
    /// rather than unlaunched — the degrade a non-root agent already produces.
    pub(crate) fn for_cgroup(cgroup_path: Option<&Path>) -> Option<Self> {
        let procs = CString::new(cgroup_path?.join("cgroup.procs").as_os_str().as_bytes()).ok()?;
        // fd 2 is the child's own stdio by the time pre_exec runs, so warn on a
        // dup of spurd's stderr instead of into the job's output.
        let log_fd = dup_cloexec(libc::STDERR_FILENO);
        Some(Self { procs, log_fd })
    }

    /// Call from `pre_exec`, while the child is still root: an unprivileged process
    /// cannot write another cgroup's `cgroup.procs`. Returns whether the join landed.
    pub(crate) fn join(&self) -> bool {
        join_cgroup_self(&self.procs, self.log_fd)
    }

    #[cfg(test)]
    fn procs_path(&self) -> &std::ffi::CStr {
        &self.procs
    }
}

impl Drop for CgroupJoin {
    fn drop(&mut self) {
        if self.log_fd >= 0 {
            unsafe { libc::close(self.log_fd) };
        }
    }
}

/// Whether `pid` is in the job's cgroup. The child joins itself pre-exec, so
/// this verifies the join rather than performing it.
pub(crate) fn cgroup_has_pid(cgroup_path: &Path, pid: u32) -> bool {
    let Ok(procs) = std::fs::read_to_string(cgroup_path.join("cgroup.procs")) else {
        return false;
    };
    procs.lines().any(|line| line.trim() == pid.to_string())
}

/// Atomically SIGKILLs every process in the cgroup (cgroup-v2 `cgroup.kill`),
/// reaching descendants that detached from the signaled process group.
pub fn cgroup_kill(cgroup_path: &Path) -> std::io::Result<()> {
    std::fs::write(cgroup_path.join("cgroup.kill"), b"1")
}

/// Signals every pid in the cgroup with any signal (unlike `cgroup_kill`,
/// SIGKILL-only). Returns the count signaled; `Ok(0)` means no tracked pids.
pub fn cgroup_signal(cgroup_path: &Path, sig: Signal) -> std::io::Result<usize> {
    let pids = std::fs::read_to_string(cgroup_path.join("cgroup.procs"))?;
    let mut signaled = 0;
    for pid_str in pids.lines() {
        if let Ok(pid) = pid_str.trim().parse::<i32>() {
            if signal::kill(Pid::from_raw(pid), sig).is_ok() {
                signaled += 1;
            }
        }
    }
    Ok(signaled)
}

/// Whether the job's cgroup recorded an OOM kill (cgroup-v2 `memory.events`).
/// False if the file is absent/unreadable. Call before `cleanup_cgroup`.
pub fn cgroup_oom_killed(cgroup_path: &Path) -> bool {
    let Ok(events) = std::fs::read_to_string(cgroup_path.join("memory.events")) else {
        return false;
    };
    events.lines().any(|line| {
        let mut it = line.split_whitespace();
        matches!((it.next(), it.next()), (Some("oom_kill"), Some(n)) if n != "0")
    })
}

/// Owns a job's cgroup until something else takes responsibility: a launch hands
/// it to the caller, a teardown removes it. Error returns would strand it.
#[must_use = "the cgroup is removed when this guard drops; bind it to choose when"]
pub(crate) struct CgroupGuard(Option<PathBuf>);

impl CgroupGuard {
    /// Take over a cgroup already detached from its job. Removal is deferred to
    /// the drop, so the holder picks a point where blocking is acceptable.
    pub(crate) fn new(path: Option<PathBuf>) -> Self {
        Self(path)
    }

    fn path(&self) -> Option<&Path> {
        self.0.as_deref()
    }

    /// Hand the cgroup to the caller; dropping the guard no longer removes it.
    fn into_inner(mut self) -> Option<PathBuf> {
        self.0.take()
    }
}

impl Drop for CgroupGuard {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            cleanup_cgroup(&path);
        }
    }
}

/// Bounded to 200ms: cleanup sits on the completion-reporting path, and the next
/// `setup_cgroup` clears any directory this gives up on.
const CGROUP_REMOVE_ATTEMPTS: u32 = 20;
const CGROUP_REMOVE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(10);

/// Take a job's cgroup directory, clearing a leftover one first. `AlreadyExists` means the
/// leftover could not be removed, which the caller refuses to adopt: it may carry another
/// run's processes, limits and device filter.
fn claim_cgroup_dir(cgroup_path: &Path) -> std::io::Result<()> {
    if cgroup_path.exists() {
        cleanup_cgroup(cgroup_path);
    }
    std::fs::create_dir(cgroup_path)
}

/// Whether a failed cgroup claim must fail the launch or may run without isolation.
enum CgroupClaim {
    Fatal(String),
    Degrade,
}

/// A surviving or uncreatable cgroup is never adopted; the choice is only fail vs. run
/// unconstrained. Fail closed for root (a survival means live processes it should reap) or
/// `required`; else degrade, so a root-owned leftover a non-root agent cannot remove never wedges.
fn classify_cgroup_claim_failure(
    err: &std::io::Error,
    is_root: bool,
    required: bool,
    cgroup_path: &Path,
) -> CgroupClaim {
    if is_root {
        if err.kind() == std::io::ErrorKind::AlreadyExists {
            return CgroupClaim::Fatal(format!(
                "job cgroup {} could not be cleared for reuse (live processes still in it)",
                cgroup_path.display()
            ));
        }
        return CgroupClaim::Fatal(format!("cgroup creation failed as root: {err}"));
    }
    if required {
        return CgroupClaim::Fatal(format!(
            "[cgroup] required but the job cgroup could not be created: {err}"
        ));
    }
    CgroupClaim::Degrade
}

/// Kill any leftover processes in the job's cgroup and remove the directory.
pub fn cleanup_cgroup(cgroup_path: &Path) {
    // Kill any remaining processes
    if let Ok(pids) = std::fs::read_to_string(cgroup_path.join("cgroup.procs")) {
        for pid_str in pids.lines() {
            if let Ok(pid) = pid_str.trim().parse::<i32>() {
                let _ = signal::kill(Pid::from_raw(pid), Signal::SIGKILL);
            }
        }
    }

    // `kill` returning does not mean the process has left the cgroup, and rmdir
    // fails EBUSY until it has. An abandoned dir strands its device program.
    for attempt in 1..=CGROUP_REMOVE_ATTEMPTS {
        match std::fs::remove_dir(cgroup_path) {
            Ok(()) => return,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(e) if attempt == CGROUP_REMOVE_ATTEMPTS => {
                warn!(error = %e, path = %cgroup_path.display(), "failed to remove cgroup");
            }
            Err(_) => std::thread::sleep(CGROUP_REMOVE_INTERVAL),
        }
    }
}

/// Recursively signal a process and all its descendants (children first).
pub(crate) fn kill_process_tree(pid: i32, sig: Signal) {
    let children = get_child_pids(pid);
    for child in &children {
        kill_process_tree(*child, sig);
    }
    let _ = signal::kill(Pid::from_raw(pid), sig);
}

/// Read immediate child PIDs from /proc/<pid>/task/<pid>/children.
fn get_child_pids(pid: i32) -> Vec<i32> {
    let path = format!("/proc/{}/task/{}/children", pid, pid);
    std::fs::read_to_string(&path)
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect()
}

/// Whether output file/dir creation must be performed as the submitting user.
/// Only meaningful when spurd is root and the job targets a non-root user.
fn should_run_as_user(uid: u32) -> bool {
    uid > 0 && nix::unistd::geteuid().is_root()
}

/// Resolve and apply user credentials for container fork children.
/// Delegates to the centralized `PrivDrop` implementation.
fn resolve_user_creds(uid: u32, gid: u32) -> Option<crate::privdrop::PrivDrop> {
    crate::privdrop::PrivDrop::resolve_if_needed(uid, gid)
}

/// Open a single output file, creating parent directories. Runs in whatever
/// credentials the caller holds — as the submitting user when invoked from the
/// forked helper.
fn open_output_file(path: &str, use_append: bool) -> std::io::Result<std::fs::File> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true);
    if use_append {
        opts.append(true);
    } else {
        opts.truncate(true);
    }
    opts.open(path)
}

/// A step's stdout/stderr spool files, open for the child to inherit and their
/// paths for the agent to tail. Steps stream through these files (StreamJobOutput
/// follows the growing file) instead of buffering their whole output in the RPC
/// response, so a step's output is bounded on the compute node and visible before
/// the step exits.
pub(crate) struct StepOutputFiles {
    pub stdout: std::fs::File,
    pub stderr: std::fs::File,
    pub stdout_path: PathBuf,
    pub stderr_path: PathBuf,
}

/// The candidate spool paths for a step's stdout or stderr, mirroring the spool
/// roots [`open_step_output_files`] (via [`create_job_spool_dir`]) chooses
/// between. A reader that did not open the file cannot see which root the writer
/// picked, so it checks both. Kept in sync with `open_step_output_files`' names.
pub(crate) fn step_output_path_candidates(
    job_id: JobId,
    step_id: u32,
    stderr: bool,
) -> Vec<PathBuf> {
    let name = format!("step{step_id}.{}", if stderr { "err" } else { "out" });
    [PathBuf::from(SPOOL_ROOT), std::env::temp_dir().join("spur")]
        .into_iter()
        .map(|base| base.join(format!("job{job_id}")).join(&name))
        .collect()
}

/// The step's spool file if it already exists on disk, so a reader can tail a
/// step that finished before it observed the step's active_steps entry.
pub(crate) fn existing_step_output_path(
    job_id: JobId,
    step_id: u32,
    stderr: bool,
) -> Option<String> {
    step_output_path_candidates(job_id, step_id, stderr)
        .into_iter()
        .find(|p| p.exists())
        .map(|p| p.to_string_lossy().into_owned())
}

/// Open a step's stdout/stderr spool files under the job spool dir, creating the
/// dir if needed. The agent (root) opens the files and hands the write fds to the
/// child via stdio redirection, so the child writes even after dropping to its
/// uid; the files stay agent-readable so `stream_job_output` can tail them.
/// Lives under the job spool tree so `cleanup_job_spool` reclaims it at job end.
pub(crate) fn open_step_output_files(
    job_id: JobId,
    step_id: u32,
    uid: u32,
    gid: u32,
) -> Result<StepOutputFiles, LaunchError> {
    let spool_dir = create_job_spool_dir(job_id, uid, gid)?;
    let stdout_path = spool_dir.join(format!("step{step_id}.out"));
    let stderr_path = spool_dir.join(format!("step{step_id}.err"));
    let open = |path: &Path| -> Result<std::fs::File, LaunchError> {
        let file = open_output_file(&path.to_string_lossy(), false).map_err(|e| {
            LaunchError::NodeFault(
                anyhow::Error::new(e).context(format!("open step output file {}", path.display())),
            )
        })?;
        // These files hold arbitrary user output, so keep them private (0600) —
        // they can otherwise become world-readable under a typical umask. The
        // child inherits the write fd, so it writes regardless of ownership; hand
        // ownership to the job user when spurd is root so only they (and root)
        // can read it, matching write_job_scratch.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
            if should_run_as_user(uid) {
                use nix::unistd::{Gid, Uid};
                let _ =
                    nix::unistd::chown(path, Some(Uid::from_raw(uid)), Some(Gid::from_raw(gid)));
            }
        }
        Ok(file)
    };
    let stdout = open(&stdout_path)?;
    let stderr = open(&stderr_path)?;
    Ok(StepOutputFiles {
        stdout,
        stderr,
        stdout_path,
        stderr_path,
    })
}

/// Send file descriptors to a peer over a Unix socket via SCM_RIGHTS.
fn send_fds(sock: RawFd, fds: &[RawFd]) -> nix::Result<()> {
    use nix::sys::socket::{sendmsg, ControlMessage, MsgFlags};
    let iov = [std::io::IoSlice::new(b"F")];
    let cmsgs = [ControlMessage::ScmRights(fds)];
    sendmsg::<()>(sock, &iov, &cmsgs, MsgFlags::empty(), None)?;
    Ok(())
}

/// Receive file descriptors sent via SCM_RIGHTS. Returns an empty vec if the
/// peer closed without sending (e.g. the helper failed before passing fds).
fn recv_fds(sock: RawFd) -> nix::Result<Vec<OwnedFd>> {
    use nix::sys::socket::{recvmsg, ControlMessageOwned, MsgFlags};
    let mut buf = [0u8; 8];
    let mut iov = [std::io::IoSliceMut::new(&mut buf)];
    let mut cmsg = nix::cmsg_space!([RawFd; 2]);
    let msg = recvmsg::<()>(sock, &mut iov, Some(&mut cmsg), MsgFlags::empty())?;
    let mut fds = Vec::new();
    for cmsg in msg.cmsgs()? {
        if let ControlMessageOwned::ScmRights(received) = cmsg {
            for fd in received {
                fds.push(unsafe { OwnedFd::from_raw_fd(fd) });
            }
        }
    }
    Ok(fds)
}

/// Copy a PMIx batch wrapper's piped stdout/stderr into the job output files.
///
/// Open MPI's PMIx client path matches standalone `srun` when stdio is a pipe;
/// dup2'ing stdout/stderr to regular files before `MPI_Init` yields singleton
/// worlds even with correct per-rank `PMIX_*` exports in the wrapper.
struct MpiStdioDrainOpts<'a> {
    uid: u32,
    gid: u32,
    stdout_path: &'a str,
    stderr_path: &'a str,
    shared: bool,
    use_append: bool,
}

fn spawn_mpi_stdio_drains(
    stdout_pipe: Option<impl tokio::io::AsyncRead + Unpin + Send + 'static>,
    stderr_pipe: Option<impl tokio::io::AsyncRead + Unpin + Send + 'static>,
    opts: MpiStdioDrainOpts<'_>,
) {
    let MpiStdioDrainOpts {
        uid,
        gid,
        stdout_path,
        stderr_path,
        shared,
        use_append,
    } = opts;
    let Ok((out, err)) = open_job_output(uid, gid, use_append, stdout_path, stderr_path) else {
        warn!(
            stdout = stdout_path,
            stderr = stderr_path,
            "failed to open PMIx batch output files for pipe drain"
        );
        return;
    };

    if shared {
        let sink = std::sync::Arc::new(tokio::sync::Mutex::new(tokio::fs::File::from_std(out)));
        if let Some(pipe) = stdout_pipe {
            let sink = sink.clone();
            tokio::spawn(async move {
                let mut file = sink.lock().await;
                let _ = tokio::io::copy(&mut tokio::io::BufReader::new(pipe), &mut *file).await;
            });
        }
        if let Some(pipe) = stderr_pipe {
            tokio::spawn(async move {
                let mut file = sink.lock().await;
                let _ = tokio::io::copy(&mut tokio::io::BufReader::new(pipe), &mut *file).await;
            });
        }
    } else {
        if let Some(pipe) = stdout_pipe {
            let mut file = tokio::fs::File::from_std(out);
            tokio::spawn(async move {
                let _ = tokio::io::copy(&mut tokio::io::BufReader::new(pipe), &mut file).await;
            });
        }
        if let Some(pipe) = stderr_pipe {
            let mut file = tokio::fs::File::from_std(err);
            tokio::spawn(async move {
                let _ = tokio::io::copy(&mut tokio::io::BufReader::new(pipe), &mut file).await;
            });
        }
    }
}

/// Open a job's stdout/stderr, creating parent directories.
///
/// When spurd is root and the job targets a non-root user, a forked child drops
/// to the user's credentials before touching the filesystem and passes the open
/// fds back over a socketpair. Resolving paths as the user (not root) is what
/// prevents a job from coercing root into creating, truncating, or owning files
/// outside the user's reach; it also makes the files user-owned without a chown.
/// Otherwise the files are opened in-process.
fn open_job_output(
    uid: u32,
    gid: u32,
    use_append: bool,
    stdout_path: &str,
    stderr_path: &str,
) -> anyhow::Result<(std::fs::File, std::fs::File)> {
    // When stderr follows stdout (same resolved path, e.g. `srun -o` with no
    // `-e`), stderr must share stdout's open file description via dup so both
    // streams advance a single shared write offset and interleave correctly.
    // Opening the path a second time would give stderr an independent offset,
    // and subsequent stdout writes would clobber whatever stderr wrote.
    let shared = stderr_path == stdout_path;

    if !should_run_as_user(uid) {
        let out = open_output_file(stdout_path, use_append).context("open stdout")?;
        let err = if shared {
            out.try_clone().context("clone stdout fd for stderr")?
        } else {
            open_output_file(stderr_path, use_append).context("open stderr")?
        };
        return Ok((out, err));
    }

    // Resolve credentials before the fork; see resolve_user_creds.
    let creds = resolve_user_creds(uid, gid);

    use nix::sys::socket::{socketpair, AddressFamily, SockFlag, SockType};
    let (parent_sock, child_sock) = socketpair(
        AddressFamily::Unix,
        SockType::Datagram,
        None,
        SockFlag::empty(),
    )
    .context("socketpair for output fd passing")?;

    match unsafe { nix::unistd::fork().context("fork for output open")? } {
        nix::unistd::ForkResult::Child => {
            // CRITICAL: post-fork, so synchronous + async-signal-safe only
            // (tokio is broken here). _exit skips atexit/stdio flushing that
            // could deadlock on a lock a sibling thread held at fork time.
            // Exit codes distinguish failure stages.
            drop(parent_sock);
            let code = 'open: {
                if let Some(ref pd) = creds {
                    if pd.apply().is_err() {
                        break 'open 1;
                    }
                }
                let Ok(out) = open_output_file(stdout_path, use_append) else {
                    break 'open 2;
                };
                // Same fd (dup) when stderr follows stdout; SCM_RIGHTS preserves
                // the shared open file description, so both land one offset.
                let err = if shared {
                    match out.try_clone() {
                        Ok(f) => f,
                        Err(_) => break 'open 3,
                    }
                } else {
                    match open_output_file(stderr_path, use_append) {
                        Ok(f) => f,
                        Err(_) => break 'open 3,
                    }
                };
                if send_fds(child_sock.as_raw_fd(), &[out.as_raw_fd(), err.as_raw_fd()]).is_err() {
                    break 'open 4;
                }
                0
            };
            unsafe { libc::_exit(code) };
        }
        nix::unistd::ForkResult::Parent { child } => {
            drop(child_sock);
            // Reap first: the helper sends the fds before exiting, and a datagram
            // socket buffers them past the sender's lifetime, so we can wait for
            // the exit code and only then read. Recv-first would hang on the
            // failure path — a closed datagram peer yields no reliable EOF.
            let status = nix::sys::wait::waitpid(child, None);
            if !matches!(status, Ok(nix::sys::wait::WaitStatus::Exited(_, 0))) {
                bail!("output helper failed to open job output (status: {status:?})");
            }
            let fds =
                recv_fds(parent_sock.as_raw_fd()).context("receive output fds from helper")?;
            if fds.len() != 2 {
                bail!("output helper returned {} fds, expected 2", fds.len());
            }
            let mut it = fds.into_iter();
            let out = std::fs::File::from(it.next().unwrap());
            let err = std::fs::File::from(it.next().unwrap());
            Ok((out, err))
        }
    }
}

/// Resolves the work_dir a job runs in. Falls back to a per-job scratch
/// directory (not shared /tmp directly) if the submitted path can't be
/// created here, since a relative output path anchored to bare /tmp can
/// collide with another job's or user's file of the same name.
fn resolve_effective_work_dir(
    job_id: JobId,
    run_attempt: u32,
    work_dir: &str,
    uid: u32,
    gid: u32,
) -> String {
    // `create_dir_all("")` is a silent no-op success (no path components to
    // create), so an empty work_dir must be checked explicitly — otherwise
    // it would be treated as already resolved instead of falling through to
    // the scratch-dir default below.
    if !work_dir.is_empty() && create_dir_as_user(Path::new(work_dir), uid, gid) {
        return work_dir.to_string();
    }
    let scratch_dir = std::env::temp_dir().join(format!("spur-job_{job_id}_{run_attempt}"));
    if create_dir_as_user(&scratch_dir, uid, gid) {
        warn!(job_id, work_dir, scratch_dir = %scratch_dir.display(),
            "work_dir unavailable on this node, using a per-job scratch directory");
        return scratch_dir.to_string_lossy().into_owned();
    }
    warn!(
        job_id,
        work_dir, "work_dir and per-job scratch directory both unavailable, using /tmp"
    );
    "/tmp".to_string()
}

/// Create `dir` and any missing parents as the submitting user (forking to drop
/// privilege when spurd is root), so directory creation resolves symlinks and
/// permissions with the user's authority. Returns whether the tree now exists.
fn create_dir_as_user(dir: &Path, uid: u32, gid: u32) -> bool {
    if !should_run_as_user(uid) {
        return std::fs::create_dir_all(dir).is_ok();
    }
    // Resolve credentials before the fork.
    let creds = resolve_user_creds(uid, gid);
    match unsafe { nix::unistd::fork() } {
        Ok(nix::unistd::ForkResult::Child) => {
            // _exit skips atexit/stdio flushing, unsafe in a post-fork child.
            let ok = creds.as_ref().map(|c| c.apply().is_ok()).unwrap_or(true)
                && std::fs::create_dir_all(dir).is_ok();
            unsafe { libc::_exit(if ok { 0 } else { 1 }) };
        }
        Ok(nix::unistd::ForkResult::Parent { child }) => {
            matches!(
                nix::sys::wait::waitpid(child, None),
                Ok(nix::sys::wait::WaitStatus::Exited(_, 0))
            )
        }
        Err(_) => false,
    }
}

/// Create a node-local spool directory for a job's scratch files. Prefers
/// `SPOOL_ROOT`; falls back to a temp dir when it isn't writable (e.g. non-root
/// dev runs). When spurd is root and the job targets a user, the dir is handed
/// to that user so the job — which runs as the user — can traverse it.
fn create_job_spool_dir(job_id: JobId, uid: u32, gid: u32) -> Result<PathBuf, LaunchError> {
    let mut failures = Vec::new();
    for base in [PathBuf::from(SPOOL_ROOT), std::env::temp_dir().join("spur")] {
        let dir = base.join(format!("job{}", job_id));
        match std::fs::create_dir_all(&dir) {
            Ok(()) => {
                if should_run_as_user(uid) {
                    use nix::unistd::{Gid, Uid};
                    // Path-based chown is safe here: the spool tree is
                    // root-owned, not user-controlled, so no symlink TOCTOU.
                    let _ = nix::unistd::chown(
                        &dir,
                        Some(Uid::from_raw(uid)),
                        Some(Gid::from_raw(gid)),
                    );
                }
                return Ok(dir);
            }
            Err(e) => failures.push((dir, e)),
        }
    }
    Err(spool_dir_error(failures))
}

/// Build the error for a spool dir that could not be created under any candidate
/// root. Prefers the owned root's failure over the temp fallback's, since that
/// is the one an operator configured and the only one whose failure condemns the
/// node.
///
/// The `io::Error` must stay a source rather than be formatted into the message:
/// [`is_node_fault_io_error`] detects the fault by walking the chain, so a
/// flattened errno would silently downgrade a node fault to a job failure.
fn spool_dir_error(mut failures: Vec<(PathBuf, std::io::Error)>) -> LaunchError {
    if failures.is_empty() {
        return LaunchError::Other(anyhow::anyhow!("no spool root candidates configured"));
    }
    let chosen = failures
        .iter()
        .position(|(dir, _)| is_node_owned_spool(dir))
        .unwrap_or(0);
    let (dir, err) = failures.swap_remove(chosen);
    let err = anyhow::Error::new(err).context(format!("create job spool dir {}", dir.display()));
    classify_spool_error(&dir, err)
}

/// Private per-job directory for srun step scripts under the step work dir.
pub(crate) fn prepare_step_script_dir(
    work_dir: &str,
    job_id: JobId,
    uid: u32,
    gid: u32,
) -> anyhow::Result<PathBuf> {
    let dir = PathBuf::from(work_dir).join(format!(".spur_step_{job_id}"));
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
        if should_run_as_user(uid) {
            use nix::unistd::{Gid, Uid};
            nix::unistd::chown(&dir, Some(Uid::from_raw(uid)), Some(Gid::from_raw(gid)))
                .with_context(|| format!("chown {}", dir.display()))?;
        }
    }
    Ok(dir)
}

/// Write a scratch file (job script, namespace wrapper) executable. When spurd
/// is root and the job targets a user, hand ownership to that user and keep the
/// file private (0700), so only the job and root can read it — matching Slurm's
/// batch script handling.
pub(crate) fn write_job_scratch(
    path: &Path,
    content: &str,
    uid: u32,
    gid: u32,
) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, content).with_context(|| format!("write {}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    if should_run_as_user(uid) {
        use nix::unistd::{Gid, Uid};
        nix::unistd::chown(path, Some(Uid::from_raw(uid)), Some(Gid::from_raw(gid)))
            .with_context(|| format!("chown {}", path.display()))?;
    }
    Ok(())
}

/// Remove a job's spool directory (best-effort), mirroring Slurm purging its
/// batchdir after completion. Tries both candidate roots since the fallback
/// location isn't recorded.
pub fn cleanup_job_spool(job_id: JobId) {
    for base in [PathBuf::from(SPOOL_ROOT), std::env::temp_dir().join("spur")] {
        let _ = std::fs::remove_dir_all(base.join(format!("job{}", job_id)));
    }
}

/// Resolve output path patterns (%j → job_id, etc.)
/// Resolve a pattern against the *effective* work_dir (may be the `/tmp`
/// fallback) via the shared resolver, so agent and controller paths match.
fn resolve_output_path(cfg: &JobLaunchConfig, work_dir: &str, pattern: &str) -> String {
    spur_core::job::resolve_output_pattern(
        pattern,
        &spur_core::job::OutputPathContext {
            job_id: cfg.job_id,
            name: &cfg.name,
            user: &cfg.user,
            work_dir,
            node: (!cfg.node.is_empty()).then_some(cfg.node.as_str()),
            array_job_id: cfg.array_job_id,
            array_task_id: cfg.array_task_id,
        },
    )
}

/// Launch a containerized job via explicit fork() + container_init().
///
/// The child process does all container setup (namespaces, mounts, pivot_root,
/// priv drop) in Rust, then execs the job. No generated bash scripts, no
/// dependency on host binaries inside the container.
///
/// The parent tracks the child PID via a sync pipe and wraps waitpid in a
/// blocking tokio task so it doesn't stall the async runtime.
async fn launch_container_job(
    cfg: &JobLaunchConfig,
    ctn: &ContainerLaunchConfig,
    env: &HashMap<String, String>,
    job_io: JobIo,
    cgroup_path: &CgroupGuard,
) -> anyhow::Result<(RunningJob, Option<OwnedFd>)> {
    let job_id = cfg.job_id;

    // Sync pipe: child writes status, parent reads.
    // Convert OwnedFd to raw fds for manual lifecycle management across fork.
    let (pipe_r, pipe_w) = nix::unistd::pipe().context("create sync pipe")?;
    // Prevent read end from leaking into exec'd process
    nix::fcntl::fcntl(
        &pipe_r,
        nix::fcntl::FcntlArg::F_SETFD(nix::fcntl::FdFlag::FD_CLOEXEC),
    )
    .ok();
    let ready_r = pipe_r.as_raw_fd();
    let ready_w = pipe_w.as_raw_fd();
    // Owners close these; the raw copies are only for the forked child, which
    // execs or _exits without running destructors.
    let pipe_r_owner = pipe_r;
    let pipe_w_owner = pipe_w;

    // Snapshot raw I/O fds before fork — the Copy JobIoRaw can be used
    // in the child without owning the fds (parent's OwnedFds keep them alive
    // across the fork boundary).
    let raw_io = job_io.raw();

    // Snapshot everything the child needs (must not reference async state after fork)
    let config = &ctn.config;
    let rootfs = ctn.rootfs.clone();
    // The image's own config.Env is the base environment, as with docker run;
    // the job environment layers on top of it.
    let env_snapshot = spur_net::oci::container_base_env(&rootfs, env.clone());
    let container_env = config.container_env.clone();
    let entrypoint = config.entrypoint.clone();

    // Precomputed before fork; the forked child joins the cgroup itself (below).
    let cgroup_procs = cgroup_path
        .path()
        .and_then(|p| CString::new(p.join("cgroup.procs").as_os_str().as_bytes()).ok());
    // Dup spurd's stderr (CLOEXEC) so the child can report a join failure; its
    // own stderr is wired to the job before the join runs.
    let cgroup_log_fd = if cgroup_procs.is_some() {
        dup_cloexec(libc::STDERR_FILENO)
    } else {
        -1
    };

    match unsafe { nix::unistd::fork().context("fork for container job")? } {
        nix::unistd::ForkResult::Child => {
            // === CHILD PROCESS ===
            // CRITICAL: synchronous code only. Tokio runtime is broken after fork.
            unsafe {
                libc::close(ready_r);
            }

            // Reset signal handlers
            unsafe {
                libc::signal(libc::SIGCHLD, libc::SIG_DFL);
                libc::signal(libc::SIGPIPE, libc::SIG_DFL);
            }

            unsafe {
                if let Err(e) = raw_io.wire() {
                    let msg = format!("E:stdio wire failed: {:#}", e);
                    let _ = libc::write(ready_w, msg.as_ptr() as *const _, msg.len());
                    libc::_exit(1);
                }
            }

            // Join while still root, before pivot_root hides the host cgroupfs and
            // before close_inherited_fds reaps the log fd. `required` is verified
            // parent-side after readiness, so a failure here is best-effort.
            if let Some(ref procs) = cgroup_procs {
                let _ = join_cgroup_self(procs, cgroup_log_fd);
            }

            crate::container::close_inherited_fds(ready_w);

            // RLIMIT_MEMLOCK: raise while still root, before container_init drops privileges.
            apply_memlock(cfg.memlock);

            // Run container init: namespaces, mounts, pivot_root, priv drop
            let hook_env = match crate::container::container_init(config, &rootfs) {
                Ok(env) => env,
                Err(e) => {
                    let msg = format!("E:{:#}", e);
                    unsafe {
                        libc::write(ready_w, msg.as_ptr() as *const _, msg.len());
                    }
                    std::process::exit(1);
                }
            };

            // Signal parent: setup complete
            unsafe {
                libc::write(ready_w, b"OK".as_ptr() as *const _, 2);
                libc::close(ready_w);
            }

            // Build final environment: base + container_env + hook environ.d
            let mut final_env = env_snapshot;
            for (k, v) in &container_env {
                final_env.insert(k.clone(), v.clone());
            }
            for (k, v) in hook_env {
                final_env.insert(k, v);
            }
            let c_env: Vec<CString> = final_env
                .iter()
                .filter_map(|(k, v)| CString::new(format!("{}={}", k, v)).ok())
                .collect();
            let c_env_refs: Vec<&std::ffi::CStr> = c_env.iter().map(|s| s.as_c_str()).collect();

            // Pick a shell that exists in the container
            let shell = if Path::new("/bin/bash").exists() {
                "/bin/bash"
            } else {
                "/bin/sh"
            };
            let c_shell = CString::new(shell).unwrap();
            let exec_args: Vec<CString> = if let Some(ref ep) = entrypoint {
                let cmd = format!("{} && {} /tmp/spur_job_{}.sh", ep, shell, job_id);
                vec![
                    c_shell.clone(),
                    CString::new("-c").unwrap(),
                    CString::new(cmd).unwrap(),
                ]
            } else {
                vec![
                    c_shell.clone(),
                    CString::new(format!("/tmp/spur_job_{}.sh", job_id)).unwrap(),
                ]
            };
            let exec_arg_refs: Vec<&std::ffi::CStr> =
                exec_args.iter().map(|s| s.as_c_str()).collect();

            let _ = nix::unistd::execve(&c_shell, &exec_arg_refs, &c_env_refs);
            eprintln!("spur: execve failed: {}", std::io::Error::last_os_error());
            std::process::exit(1);
        }

        nix::unistd::ForkResult::Parent { child } => {
            drop(pipe_w_owner);
            unsafe {
                if cgroup_log_fd >= 0 {
                    libc::close(cgroup_log_fd);
                }
            }

            // Drop the slave fd immediately so the master gets EOF when the child exits.
            let pty_master = job_io.into_master();

            let child_pid = child.as_raw();

            // pidfd prevents PID recycling; falls back gracefully on kernels < 5.3
            let pidfd = pidfd_open(child_pid).ok();
            if pidfd.is_none() {
                debug!("pidfd_open unavailable, falling back to raw PID tracking");
            }

            let mut buf = [0u8; 512];
            let n = unsafe { libc::read(ready_r, buf.as_mut_ptr() as *mut _, buf.len()) };
            let n = n.max(0) as usize;
            drop(pipe_r_owner);

            if n < 2 || &buf[..2] != b"OK" {
                let msg = String::from_utf8_lossy(&buf[..n]);
                bail!("container init failed for job {}: {}", job_id, msg);
            }

            // Only now is the child's self-join guaranteed to have run. Verify
            // rather than write (DC2): the container tree inherits this membership.
            if cfg.cgroup.required {
                let joined = cgroup_path
                    .path()
                    .is_none_or(|cgroup| cgroup_has_pid(cgroup, child_pid as u32));
                if !joined {
                    signal::kill(Pid::from_raw(child_pid), Signal::SIGKILL).ok();
                    // Reap before returning: the guard's rmdir races an exit
                    // that has been signalled but not yet completed.
                    let _ = nix::sys::wait::waitpid(child, None);
                    anyhow::bail!("[cgroup] required but the container did not join its cgroup");
                }
            }

            info!(
                job_id,
                pid = child_pid,
                rootfs = %ctn.rootfs.display(),
                "containerized job launched (fork + pivot_root)"
            );

            Ok((
                RunningJob::Forked {
                    pid: child_pid,
                    _pidfd: pidfd,
                    reaped: false,
                },
                pty_master,
            ))
        }
    }
}

/// Wrap a job script with burst buffer stage-in (before) and stage-out (after).
///
/// The `bb` string contains semicolon-separated directives:
///   - `stage_in:<cmd>` — run before the job
///   - `stage_out:<cmd>` — run after the job (best-effort, ignores failures)
///
/// Build the bash wrapper that runs inside the unshare PID/mount namespace.
///
/// The wrapper executes as root (the same uid as spurd), so it can perform
/// the proc/tmpfs/dri mounts that need CAP_SYS_ADMIN. Once isolation is in
/// place, it drops privilege via `setpriv --init-groups` and exec's the user
/// script.
///
/// Issue #128: previously the priv drop happened in `Command::pre_exec` before
/// exec'ing unshare, which made the unshare(2) syscall fail with EPERM and
/// the mounts silently no-op. Doing the drop inside the wrapper (after the
/// mounts) keeps the unshare and mounts privileged while still landing the
/// user payload as the unprivileged uid.
fn build_namespace_wrapper(
    uid: u32,
    gid: u32,
    visible_device_paths: &[String],
    script_path: &Path,
) -> String {
    let dri_nodes: Vec<&str> = visible_device_paths
        .iter()
        .filter(|p| p.starts_with("/dev/dri/"))
        .filter_map(|p| p.rsplit('/').next())
        .filter(|b| !b.is_empty())
        .collect();
    let stash_dri = dri_nodes
        .iter()
        .map(|b| format!("  cp -a /dev/dri/{b} $SPUR_HOST_DRI/{b} 2>/dev/null || true\n"))
        .collect::<String>();

    // Gated on the mount: without the tmpfs the restore lands on the host's real
    // /dev/dri, where `cp` unlinks the node before recreating it.
    const MOUNT_DRI: &str = "mount -t tmpfs tmpfs /dev/dri 2>/dev/null";
    let mount_and_restore_dri = if dri_nodes.is_empty() {
        format!("  {MOUNT_DRI} || true\n")
    } else {
        let restore = dri_nodes
            .iter()
            .map(|b| format!("    cp -a $SPUR_HOST_DRI/{b} /dev/dri/{b} 2>/dev/null || true\n"))
            .collect::<String>();
        format!("  if {MOUNT_DRI}; then\n{restore}  fi\n")
    };

    let final_exec = if uid > 0 {
        format!(
            "exec setpriv --reuid={uid} --regid={gid} --init-groups -- /bin/bash {script}\n",
            uid = uid,
            gid = gid,
            script = script_path.display(),
        )
    } else {
        format!("exec /bin/bash {}\n", script_path.display())
    };

    format!(
        concat!(
            "#!/bin/bash\n",
            "# Namespace isolation wrapper — all mounts best-effort\n",
            "mount -t proc proc /proc 2>/dev/null || true\n",
            "mount -t tmpfs tmpfs /dev/shm 2>/dev/null || true\n",
            "# GPU device restriction: stash the allocated /dev/dri nodes, replace\n",
            "# the directory with a tmpfs, then copy only those back. Staging any\n",
            "# other node would mknod a device the cgroup device filter denies.\n",
            "SPUR_HOST_DRI=$(mktemp -d /tmp/.spur_dri_XXXXXX 2>/dev/null || echo /tmp/.spur_dri)\n",
            "if [ -d /dev/dri ]; then\n",
            "  mkdir -p $SPUR_HOST_DRI 2>/dev/null || true\n",
            "{stash_dri}",
            "{mount_and_restore_dri}",
            "fi\n",
            "{final_exec}",
        ),
        stash_dri = stash_dri,
        mount_and_restore_dri = mount_and_restore_dri,
        final_exec = final_exec,
    )
}

fn wrap_with_burst_buffer(script: &str, bb: &str) -> String {
    let mut stage_in = Vec::new();
    let mut stage_out = Vec::new();

    for directive in bb.split(';') {
        let directive = directive.trim();
        if let Some(cmd) = directive.strip_prefix("stage_in:") {
            stage_in.push(cmd.trim().to_string());
        } else if let Some(cmd) = directive.strip_prefix("stage_out:") {
            stage_out.push(cmd.trim().to_string());
        }
    }

    if stage_in.is_empty() && stage_out.is_empty() {
        return script.to_string();
    }

    let mut wrapper = String::from("#!/bin/bash\n");

    // Stage-in commands (fail-fast)
    for cmd in &stage_in {
        wrapper.push_str(&format!("# Burst buffer stage-in\n{} || exit 1\n", cmd));
    }

    // The user script (inline)
    wrapper.push_str("# User script\n");
    // Remove shebang from user script if present to avoid nested shebangs
    let user_body = if script.starts_with("#!") {
        script.split_once('\n').map(|x| x.1).unwrap_or("")
    } else {
        script
    };
    wrapper.push_str(user_body);
    wrapper.push_str("\nSPUR_BB_EXIT=$?\n");

    // Stage-out commands (best-effort)
    for cmd in &stage_out {
        wrapper.push_str(&format!("# Burst buffer stage-out\n{} || true\n", cmd));
    }

    wrapper.push_str("exit $SPUR_BB_EXIT\n");
    wrapper
}

#[cfg(test)]
mod cpuset_tests {
    use super::{parse_cpu_list, permitted_cores};

    #[test]
    fn parses_ranges_lists_and_empty() {
        assert_eq!(parse_cpu_list("0-3"), [0, 1, 2, 3].into_iter().collect());
        assert_eq!(parse_cpu_list("0,2,4"), [0, 2, 4].into_iter().collect());
        assert_eq!(
            parse_cpu_list("0-1,4-5"),
            [0, 1, 4, 5].into_iter().collect()
        );
        assert!(parse_cpu_list("").is_empty());
        assert!(parse_cpu_list("\n").is_empty());
    }

    #[test]
    fn drops_cores_the_parent_cgroup_does_not_hold() {
        // Core ids are synthesized from a count, so a sparse online mask can name
        // cores that do not exist. Writing one fails ERANGE, leaving it unbounded.
        assert_eq!(permitted_cores(&[0, 1, 99], "0-3"), vec![0, 1]);
        assert_eq!(permitted_cores(&[64, 65], "0-63"), Vec::<u32>::new());
    }

    #[test]
    fn keeps_the_full_set_when_the_parent_permits_it() {
        assert_eq!(permitted_cores(&[0, 1, 2, 3], "0-3"), vec![0, 1, 2, 3]);
        assert_eq!(permitted_cores(&[2, 3], "0-1,2-3"), vec![2, 3]);
    }
}

#[cfg(test)]
mod cgroup_join_tests {
    use super::CgroupJoin;
    use std::path::Path;

    #[test]
    fn a_job_with_a_cgroup_is_wired_to_join_its_procs_file() {
        let join = CgroupJoin::for_cgroup(Some(Path::new("/sys/fs/cgroup/spur/job_7")))
            .expect("a job with a cgroup must be wired to join it");
        assert_eq!(
            join.procs_path().to_bytes(),
            b"/sys/fs/cgroup/spur/job_7/cgroup.procs"
        );
    }

    #[test]
    fn a_job_without_a_cgroup_has_nothing_to_join() {
        // Degraded, not fatal: a non-root agent creates no cgroup, and the
        // child still has to run.
        assert!(CgroupJoin::for_cgroup(None).is_none());
    }
}

#[cfg(test)]
mod cgroup_guard_tests {
    use super::{cleanup_cgroup, CgroupGuard};

    #[test]
    fn dropping_the_guard_removes_the_cgroup() {
        let dir = tempfile::tempdir().unwrap();
        let cgroup = dir.path().join("job_1");
        std::fs::create_dir(&cgroup).unwrap();

        drop(CgroupGuard(Some(cgroup.clone())));
        assert!(
            !cgroup.exists(),
            "a failed launch must not strand its cgroup"
        );
    }

    #[test]
    fn releasing_the_guard_keeps_the_cgroup() {
        // The running job owns the cgroup once a launch succeeds; removing it
        // here would unbound the job it was created for.
        let dir = tempfile::tempdir().unwrap();
        let cgroup = dir.path().join("job_2");
        std::fs::create_dir(&cgroup).unwrap();

        let released = CgroupGuard(Some(cgroup.clone())).into_inner();
        assert_eq!(released.as_deref(), Some(cgroup.as_path()));
        assert!(cgroup.exists());
    }

    #[test]
    fn a_disabled_cgroup_is_a_no_op() {
        drop(CgroupGuard(None));
    }

    #[test]
    fn cleanup_removes_an_empty_cgroup() {
        let dir = tempfile::tempdir().unwrap();
        let cgroup = dir.path().join("job_3");
        std::fs::create_dir(&cgroup).unwrap();

        cleanup_cgroup(&cgroup);
        assert!(!cgroup.exists());
    }

    #[test]
    fn cleanup_gives_up_on_a_cgroup_that_never_empties() {
        // A non-empty dir fails rmdir the way a busy cgroup does, so this exhausts
        // the retry budget; reaching the assert at all is the property under test.
        let dir = tempfile::tempdir().unwrap();
        let cgroup = dir.path().join("job_4");
        std::fs::create_dir(&cgroup).unwrap();
        std::fs::write(cgroup.join("cgroup.procs"), "not-a-pid\n").unwrap();

        cleanup_cgroup(&cgroup);
        assert!(cgroup.exists());
    }

    #[test]
    fn cleanup_of_a_missing_cgroup_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        cleanup_cgroup(&dir.path().join("job_5"));
    }
}

#[cfg(test)]
mod cgroup_files_tests {
    use super::cgroup_limit_files;
    use spur_core::config::CgroupConfig;

    fn files_for(
        cfg: &CgroupConfig,
        cpus: u32,
        memory_mb: u64,
        cpu_ids: &[u32],
    ) -> Vec<(&'static str, String)> {
        cgroup_limit_files(&cfg.limits_for(cpus, memory_mb, cpu_ids).expect("enabled"))
    }

    #[test]
    fn writes_defaults_without_a_cfs_quota() {
        let files = files_for(&CgroupConfig::default(), 4, 2048, &[0, 1, 2, 3]);
        let get = |k: &str| files.iter().find(|(n, _)| *n == k).map(|(_, v)| v.as_str());
        let expect_mem = (2048u64 * 1024 * 1024).to_string();
        assert_eq!(get("cpu.max"), None);
        assert_eq!(get("memory.max"), Some(expect_mem.as_str()));
        // AllowedRAMSpace=100 collapses the soft limit onto the hard one.
        assert_eq!(get("memory.high"), Some(expect_mem.as_str()));
        // Swap is unconstrained by default, as in Slurm.
        assert_eq!(get("memory.swap.max"), None);
        assert_eq!(get("memory.oom.group"), Some("1"));
        assert_eq!(get("pids.max"), Some("1024")); // max(4*256, 1024)
        assert_eq!(get("cpuset.cpus"), Some("0,1,2,3"));
    }

    #[test]
    fn cpu_quota_knob_emits_cpu_max() {
        let cfg = CgroupConfig {
            cpu_quota: true,
            ..CgroupConfig::default()
        };
        let files = files_for(&cfg, 4, 0, &[]);
        assert_eq!(
            files
                .iter()
                .find(|(n, _)| *n == "cpu.max")
                .map(|(_, v)| v.as_str()),
            Some("400000 100000")
        );
    }

    #[test]
    fn headroom_emits_a_lower_memory_high_than_memory_max() {
        let cfg = CgroupConfig {
            allowed_ram_percent: 150,
            ..CgroupConfig::default()
        };
        let files = files_for(&cfg, 1, 1024, &[]);
        let get = |k: &str| files.iter().find(|(n, _)| *n == k).map(|(_, v)| v.as_str());
        let high = (1024u64 * 1024 * 1024).to_string();
        let max = (1536u64 * 1024 * 1024).to_string();
        assert_eq!(get("memory.high"), Some(high.as_str()));
        assert_eq!(get("memory.max"), Some(max.as_str()));
    }

    #[test]
    fn omits_memory_and_swap_for_an_unbounded_job() {
        let files = files_for(&CgroupConfig::default(), 1, 0, &[]);
        let has = |k: &str| files.iter().any(|(n, _)| *n == k);
        assert!(!has("memory.max"));
        assert!(!has("memory.high"));
        assert!(!has("memory.swap.max"));
        assert!(!has("cpuset.cpus"));
        assert!(has("pids.max"));
    }

    #[test]
    fn oom_kill_job_can_be_turned_off() {
        let cfg = CgroupConfig {
            oom_kill_job: false,
            ..CgroupConfig::default()
        };
        let files = files_for(&cfg, 1, 1024, &[]);
        assert!(!files.iter().any(|(n, _)| *n == "memory.oom.group"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cgroup_path_is_scoped_to_the_attempt_not_just_the_job() {
        let root = Path::new("/sys/fs/cgroup/spur");
        let first = cgroup_path_for(root, 4, 1);
        let second = cgroup_path_for(root, 4, 2);
        assert_ne!(
            first, second,
            "a redispatch must never share a cgroup with a not-yet-reaped prior attempt"
        );
    }

    #[test]
    fn cgroup_signal_delivers_to_every_tracked_pid() {
        use std::os::unix::process::ExitStatusExt;

        let cgroup = tempfile::tempdir().expect("cgroup directory");
        let mut child = std::process::Command::new("sleep")
            .arg("5")
            .spawn()
            .expect("spawn sleep");
        std::fs::write(cgroup.path().join("cgroup.procs"), child.id().to_string())
            .expect("seed cgroup.procs");

        let signaled =
            cgroup_signal(cgroup.path(), Signal::SIGKILL).expect("signal every tracked pid");

        assert_eq!(signaled, 1);
        let status = child.wait().expect("wait for signaled child");
        assert_eq!(status.signal(), Some(libc::SIGKILL));
    }

    #[test]
    fn cgroup_signal_reports_the_read_failure_when_theres_no_cgroup() {
        let missing = std::path::Path::new("/nonexistent/spur-cgroup-signal-test");
        assert!(cgroup_signal(missing, Signal::SIGTERM).is_err());
    }

    #[tokio::test]
    async fn cleanup_cgroup_retries_past_a_transient_removal_failure() {
        let cgroup = tempfile::tempdir().expect("cgroup directory");
        // A directory (not a plain file) at the cgroup.kill path makes the
        // write fail, so cleanup_cgroup falls back to the per-pid sweep and
        // this blocker is the only thing standing in remove_dir's way.
        let blocker = cgroup.path().join("cgroup.kill");
        std::fs::create_dir(&blocker).expect("seed blocker directory");

        let blocker_removed = blocker.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(25));
            std::fs::remove_dir(&blocker_removed).expect("clear blocker");
        });

        cleanup_cgroup(cgroup.path());

        assert!(
            !cgroup.path().exists(),
            "cleanup_cgroup must retry past a transient removal failure"
        );
    }

    #[test]
    fn decode_wait_status_splits_exit_and_signal() {
        use nix::sys::wait::WaitStatus;
        use nix::unistd::Pid;
        let p = Pid::from_raw(1);
        assert_eq!(decode_wait_status(WaitStatus::Exited(p, 7)), (7, 0));
        assert_eq!(
            decode_wait_status(WaitStatus::Signaled(
                p,
                nix::sys::signal::Signal::SIGKILL,
                false
            )),
            (0, 9)
        );
        assert_eq!(
            decode_wait_status(WaitStatus::Signaled(
                p,
                nix::sys::signal::Signal::SIGTERM,
                false
            )),
            (0, 15)
        );
        assert_eq!(decode_wait_status(WaitStatus::StillAlive), (-1, 0));
    }

    // ── launch error classification / node drain ─────────────────

    fn disk_full_error(context: &str) -> anyhow::Error {
        // Same shape the production paths produce: an io::Error from the
        // filesystem, wrapped by the call site's .context().
        anyhow::Error::new(std::io::Error::from_raw_os_error(libc::ENOSPC))
            .context(context.to_owned())
    }

    fn owned_spool() -> PathBuf {
        PathBuf::from(SPOOL_ROOT).join("job1")
    }

    fn fallback_spool() -> PathBuf {
        std::env::temp_dir().join("spur").join("job1")
    }

    #[test]
    fn spool_disk_exhaustion_is_a_node_fault_and_drains() {
        // create_job_spool_dir / write_job_scratch target SPOOL_ROOT, which
        // spurd owns, so a full filesystem there condemns the node.
        let err = classify_spool_error(&owned_spool(), disk_full_error("create job spool dir"));
        assert!(matches!(err, LaunchError::NodeFault(_)));
        let reason = err.drain_reason().expect("node fault must drain");
        assert!(reason.contains("No space left on device"), "{reason}");
    }

    #[test]
    fn a_full_temp_fallback_spool_does_not_drain() {
        // The fallback root is world-writable, so any single job can fill it.
        // Draining on that would let one runaway job walk the cluster, taking
        // out every node the scheduler retries it on.
        let err = classify_spool_error(&fallback_spool(), disk_full_error("write job script"));
        assert!(matches!(err, LaunchError::Other(_)));
        assert!(
            err.drain_reason().is_none(),
            "a full world-writable /tmp must never drain the node"
        );
    }

    #[test]
    fn exhausted_spool_roots_stay_classifiable_as_a_node_fault() {
        // Every candidate root failing to mkdir is what an exhausted rootfs
        // looks like, since SPOOL_ROOT and the temp fallback usually share a
        // filesystem. Formatting the errno into the message here would hide it
        // from classification, so the node would keep accepting jobs it cannot
        // launch — the retry storm this whole path exists to stop.
        let err = spool_dir_error(vec![
            (
                owned_spool(),
                std::io::Error::from_raw_os_error(libc::ENOSPC),
            ),
            (
                fallback_spool(),
                std::io::Error::from_raw_os_error(libc::ENOSPC),
            ),
        ]);
        assert!(matches!(err, LaunchError::NodeFault(_)));
        let reason = err.drain_reason().expect("node fault must drain");
        assert!(
            reason.contains(&owned_spool().display().to_string()),
            "the configured spool root must be the one named, got: {reason}"
        );
    }

    #[test]
    fn an_errno_rendered_into_the_message_is_not_recoverable() {
        // Why spool_dir_error keeps the io::Error as a source. Classification
        // walks the chain, so an errno turned into text is gone for good; this
        // is how the all-roots-failed path used to lose node faults.
        let flattened = anyhow::anyhow!(
            "failed to create job spool dir: {:?}",
            std::io::Error::from_raw_os_error(libc::ENOSPC)
        );
        assert!(
            !is_node_fault_io_error(&flattened),
            "an errno in the message text must not be mistaken for a real source"
        );
    }

    #[test]
    fn a_failure_confined_to_the_fallback_root_does_not_drain() {
        // Only the world-writable fallback failed. The node's own spool is
        // fine, so this is a job failure, not grounds for taking the node out
        // of service. This is the path check doing the work, not the errno.
        let err = spool_dir_error(vec![(
            fallback_spool(),
            std::io::Error::from_raw_os_error(libc::ENOSPC),
        )]);
        assert!(matches!(err, LaunchError::Other(_)));
        assert!(err.drain_reason().is_none());
    }

    #[test]
    fn an_error_with_no_io_source_never_drains() {
        // Everything under the owned root drains except EDQUOT, so the errno
        // check is what keeps a plain anyhow error out. Without it a container
        // or config problem would start condemning nodes.
        let err =
            classify_spool_error(&owned_spool(), anyhow::anyhow!("spool root not configured"));
        assert!(matches!(err, LaunchError::Other(_)));
        assert!(err.drain_reason().is_none());
    }

    #[test]
    fn a_permission_failure_on_the_owned_spool_root_is_a_node_fault() {
        // The spool tree is root-owned and every path under it is built by
        // spurd from the job id, so a submission cannot steer the errno. EACCES
        // there means the node is misconfigured or its filesystem is broken,
        // and leaving it eligible just feeds it more jobs to fail.
        let err = spool_dir_error(vec![(
            owned_spool(),
            std::io::Error::from_raw_os_error(libc::EACCES),
        )]);
        assert!(matches!(err, LaunchError::NodeFault(_)));
        assert!(err.drain_reason().is_some());
    }

    #[test]
    fn a_hardware_io_error_on_the_owned_spool_root_is_a_node_fault() {
        let err = classify_spool_error(
            &owned_spool(),
            anyhow::Error::new(std::io::Error::from_raw_os_error(libc::EIO))
                .context("write job script"),
        );
        assert!(matches!(err, LaunchError::NodeFault(_)));
    }

    #[test]
    fn write_job_scratch_keeps_the_errno_downcastable() {
        // The whole classification scheme rests on write_job_scratch leaving a
        // real io::Error in the chain. If it ever formatted the errno into its
        // message instead, every classification test above would still pass
        // while production silently stopped draining broken nodes.
        let err = write_job_scratch(
            Path::new("/nonexistent-spur-audit-dir/job.sh"),
            "#!/bin/sh\n",
            0,
            0,
        )
        .expect_err("writing under a nonexistent parent must fail");
        assert!(
            err.chain()
                .any(|c| c.downcast_ref::<std::io::Error>().is_some()),
            "the io::Error must survive as a source, not be flattened into text"
        );
    }

    #[test]
    fn read_only_spool_is_a_node_fault() {
        let err = classify_spool_error(
            &owned_spool(),
            anyhow::Error::new(std::io::Error::from_raw_os_error(libc::EROFS))
                .context("write job script"),
        );
        assert!(matches!(err, LaunchError::NodeFault(_)));
    }

    #[test]
    fn output_file_disk_exhaustion_does_not_drain() {
        // open_job_output writes to paths resolved against the job's work_dir,
        // which is user-controlled and frequently a shared mount. Its errors
        // reach the caller through `?`, i.e. From<anyhow::Error>, so they must
        // classify as Other: draining here would take a healthy node offline,
        // and the scheduler would then repeat it on every remaining node.
        let err: LaunchError = disk_full_error("failed to open job output files").into();
        assert!(matches!(err, LaunchError::Other(_)));
        assert!(
            err.drain_reason().is_none(),
            "a full user filesystem must never drain the node"
        );
    }

    #[test]
    fn user_quota_exhaustion_is_not_a_node_fault() {
        // EDQUOT is a property of a user on a shared filesystem, not of the
        // node, and no quota applies to the root-owned spool tree.
        let err = classify_spool_error(
            &owned_spool(),
            anyhow::Error::new(std::io::Error::from_raw_os_error(libc::EDQUOT))
                .context("write job script"),
        );
        assert!(matches!(err, LaunchError::Other(_)));
        assert!(err.drain_reason().is_none());
    }

    #[test]
    fn a_spool_failure_with_no_errno_does_not_drain() {
        let err =
            classify_spool_error(&owned_spool(), anyhow::anyhow!("container image not found"));
        assert!(matches!(err, LaunchError::Other(_)));
        assert!(err.drain_reason().is_none());
    }

    #[test]
    fn the_agent_does_not_self_drain_on_a_prolog_failure() {
        // The drain still happens, but the controller issues it, so it can pair
        // it with the hold. An agent-side drain would retry the job elsewhere
        // and walk a job-caused failure across the cluster.
        let err = LaunchError::PrologFailed(anyhow::anyhow!("exit status 1"));
        assert!(err.drain_reason().is_none());
        assert_eq!(
            err.to_string(),
            "prolog failed: exit status 1",
            "this text reaches the controller as the launch error and becomes \
             the drain reason there, so it must not be double-prefixed"
        );
    }

    // These exercise the in-process (non-fork) branch of the helpers: as a
    // non-root test runner, should_run_as_user() is false, so no privilege drop
    // or fork happens and behaviour is deterministic regardless of the test uid.

    #[test]
    fn create_dir_as_user_creates_full_tree() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a/b/c");
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();
        assert!(create_dir_as_user(&nested, uid, gid));
        assert!(nested.is_dir());
        // Idempotent over an existing tree.
        assert!(create_dir_as_user(&nested, uid, gid));
    }

    #[test]
    fn resolve_effective_work_dir_uses_the_submitted_path_when_creatable() {
        let dir = tempfile::tempdir().unwrap();
        let work_dir = dir.path().join("job-work-dir");
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();

        let resolved = resolve_effective_work_dir(1, 1, &work_dir.to_string_lossy(), uid, gid);

        assert_eq!(resolved, work_dir.to_string_lossy());
        assert!(work_dir.is_dir());
    }

    #[test]
    fn resolve_effective_work_dir_treats_empty_as_unset_not_already_resolved() {
        // create_dir_all("") is a silent no-op success, so an empty work_dir
        // must not be mistaken for an already-usable path.
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();

        let resolved = resolve_effective_work_dir(9001, 2, "", uid, gid);

        let expected = std::env::temp_dir().join("spur-job_9001_2");
        assert_eq!(resolved, expected.to_string_lossy());
        assert!(expected.is_dir());
        std::fs::remove_dir(&expected).ok();
    }

    #[test]
    fn resolve_effective_work_dir_falls_back_to_a_scoped_scratch_dir_not_bare_tmp() {
        // A path nested under a plain file can never be created — deterministic,
        // uid-independent way to force the fallback branch.
        let blocker = tempfile::NamedTempFile::new().unwrap();
        let unusable_work_dir = blocker.path().join("subdir");
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();

        let resolved =
            resolve_effective_work_dir(4242, 3, &unusable_work_dir.to_string_lossy(), uid, gid);

        let expected = std::env::temp_dir().join("spur-job_4242_3");
        assert_eq!(resolved, expected.to_string_lossy());
        assert!(expected.is_dir(), "scratch dir must actually be created");
        std::fs::remove_dir(&expected).ok();
    }

    #[test]
    fn open_job_output_creates_files_and_parent_dirs() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("sub/nested/job.out");
        let err = dir.path().join("sub/nested/job.err");
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();
        let (mut of, mut ef) = open_job_output(
            uid,
            gid,
            false,
            out.to_str().unwrap(),
            err.to_str().unwrap(),
        )
        .unwrap();
        of.write_all(b"o").unwrap();
        ef.write_all(b"e").unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "o");
        assert_eq!(std::fs::read_to_string(&err).unwrap(), "e");
    }

    #[test]
    fn open_job_output_append_preserves_existing_content() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("a.out");
        let err = dir.path().join("a.err");
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();
        let (op, ep) = (out.to_str().unwrap(), err.to_str().unwrap());

        let (mut of, _ef) = open_job_output(uid, gid, false, op, ep).unwrap();
        of.write_all(b"first\n").unwrap();
        drop(of);

        let (mut of, _ef) = open_job_output(uid, gid, true, op, ep).unwrap();
        of.write_all(b"second\n").unwrap();
        drop(of);

        assert_eq!(std::fs::read_to_string(&out).unwrap(), "first\nsecond\n");
    }

    #[test]
    fn open_job_output_truncate_replaces_existing_content() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("t.out");
        let err = dir.path().join("t.err");
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();
        let (op, ep) = (out.to_str().unwrap(), err.to_str().unwrap());

        let (mut of, _ef) = open_job_output(uid, gid, false, op, ep).unwrap();
        of.write_all(b"old content").unwrap();
        drop(of);

        let (mut of, _ef) = open_job_output(uid, gid, false, op, ep).unwrap();
        of.write_all(b"new").unwrap();
        drop(of);

        assert_eq!(std::fs::read_to_string(&out).unwrap(), "new");
    }

    #[test]
    fn open_job_output_shared_path_shares_offset() {
        // `srun -o file` with no `-e` makes stderr follow stdout (same path).
        // stderr must share stdout's fd (dup) so the two streams advance one
        // offset and interleave; independent offsets would clobber each other.
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("job.out");
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();
        let p = shared.to_str().unwrap();

        let (mut of, mut ef) = open_job_output(uid, gid, false, p, p).unwrap();
        // Interleave: an out write after an err write must not overwrite it.
        of.write_all(b"out1\n").unwrap();
        of.flush().unwrap();
        ef.write_all(b"err1\n").unwrap();
        ef.flush().unwrap();
        of.write_all(b"out2\n").unwrap();
        of.flush().unwrap();
        ef.write_all(b"err2\n").unwrap();
        ef.flush().unwrap();

        let contents = std::fs::read_to_string(&shared).unwrap();
        assert_eq!(
            contents, "out1\nerr1\nout2\nerr2\n",
            "streams clobbered: {contents:?}"
        );
    }

    #[test]
    fn claiming_a_cgroup_dir_creates_it_when_nothing_is_there() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("job_1");
        claim_cgroup_dir(&path).expect("a fresh id must get its directory");
        assert!(path.is_dir());
    }

    #[test]
    fn claiming_a_cgroup_dir_clears_an_empty_leftover() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("job_1");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("marker"), "stale").unwrap();
        std::fs::remove_file(path.join("marker")).unwrap();

        claim_cgroup_dir(&path).expect("an empty leftover must be cleared and remade");
        assert!(path.is_dir());
    }

    // A directory outliving its cleanup is how a live cgroup presents: rmdir fails while
    // processes remain, and adopting it would hand them this run's limits and filter.
    #[test]
    fn claiming_refuses_a_leftover_that_survives_cleanup() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("job_1");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("occupant"), "still here").unwrap();

        let err = claim_cgroup_dir(&path).expect_err("a surviving cgroup must not be adopted");
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
    }

    // A non-root agent inheriting a root-owned leftover cgroup it cannot remove gets EEXIST from
    // the claim. Under required=false that must degrade to no isolation, not refuse the launch.
    #[test]
    fn a_nonroot_agent_degrades_when_a_leftover_cgroup_cannot_be_cleared() {
        let eexist = std::io::Error::from(std::io::ErrorKind::AlreadyExists);
        assert!(matches!(
            classify_cgroup_claim_failure(&eexist, false, false, Path::new("/x/job_1")),
            CgroupClaim::Degrade
        ));
        // A plain creation error degrades the same way for a non-root, non-required agent.
        let eperm = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        assert!(matches!(
            classify_cgroup_claim_failure(&eperm, false, false, Path::new("/x/job_1")),
            CgroupClaim::Degrade
        ));
    }

    // The safety directions the degrade must not weaken: a root agent that cannot clear a
    // leftover has live processes to reckon with, and `required` means enforce or refuse.
    #[test]
    fn a_root_or_required_agent_still_fails_on_an_unclaimable_cgroup() {
        let eexist = std::io::Error::from(std::io::ErrorKind::AlreadyExists);
        assert!(matches!(
            classify_cgroup_claim_failure(&eexist, true, false, Path::new("/x/job_1")),
            CgroupClaim::Fatal(_)
        ));
        assert!(matches!(
            classify_cgroup_claim_failure(&eexist, false, true, Path::new("/x/job_1")),
            CgroupClaim::Fatal(_)
        ));
    }

    #[test]
    fn write_job_scratch_is_executable_and_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spur_job.sh");
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();
        write_job_scratch(&path, "#!/bin/bash\necho hi\n", uid, gid).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "#!/bin/bash\necho hi\n"
        );
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);
    }

    #[test]
    fn job_spool_dir_round_trips_create_and_cleanup() {
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();
        // A job id unlikely to collide with a real job on the test host; as a
        // non-root runner this resolves to the temp-dir fallback.
        let job_id: JobId = 987_654_321;
        // LaunchError has no Debug impl on purpose (it must not be convertible
        // back into an anyhow::Error), so report it through Display.
        let dir = create_job_spool_dir(job_id, uid, gid)
            .unwrap_or_else(|e| panic!("create spool dir: {e}"));
        assert!(dir.is_dir());
        write_job_scratch(&dir.join("spur_job.sh"), "x", uid, gid).unwrap();
        cleanup_job_spool(job_id);
        assert!(!dir.exists());
    }

    // send_fds/recv_fds are process-agnostic: they pass fds over any Unix
    // socket. Exercising the SCM_RIGHTS round-trip over an in-process socketpair
    // covers the fd-passing logic without needing root or a fork.
    #[test]
    fn send_recv_fds_round_trips_an_open_file() {
        use nix::sys::socket::{socketpair, AddressFamily, SockFlag, SockType};
        use std::io::{Read, Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("passed.txt");
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        file.write_all(b"from-sender").unwrap();

        let (a, b) = socketpair(
            AddressFamily::Unix,
            SockType::Datagram,
            None,
            SockFlag::empty(),
        )
        .unwrap();

        send_fds(a.as_raw_fd(), &[file.as_raw_fd()]).unwrap();
        let received = recv_fds(b.as_raw_fd()).unwrap();
        assert_eq!(received.len(), 1);

        // The received fd refers to the same open file description: writes made
        // through it land in the same file the sender opened.
        let mut got = std::fs::File::from(received.into_iter().next().unwrap());
        got.write_all(b"-and-more").unwrap();
        got.flush().unwrap();

        let mut contents = String::new();
        file.rewind().unwrap();
        file.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "from-sender-and-more");
    }

    #[test]
    fn recv_fds_returns_empty_when_no_fds_sent() {
        use nix::sys::socket::{sendmsg, socketpair, AddressFamily, MsgFlags, SockFlag, SockType};

        let (a, b) = socketpair(
            AddressFamily::Unix,
            SockType::Datagram,
            None,
            SockFlag::empty(),
        )
        .unwrap();

        // A payload with no ancillary data — mirrors a helper that reported
        // success framing but attached no descriptors.
        let iov = [std::io::IoSlice::new(b"F")];
        sendmsg::<()>(a.as_raw_fd(), &iov, &[], MsgFlags::empty(), None).unwrap();

        let received = recv_fds(b.as_raw_fd()).unwrap();
        assert!(received.is_empty());
    }

    fn launch_cfg_for_paths(job_id: JobId, name: &str, user: &str, node: &str) -> JobLaunchConfig {
        JobLaunchConfig {
            job_id,
            run_attempt: 1,
            script: String::new(),
            work_dir: String::new(),
            name: name.to_string(),
            user: user.to_string(),
            node: node.to_string(),
            array_job_id: None,
            array_task_id: None,
            environment: HashMap::new(),
            stdout_path: String::new(),
            stderr_path: String::new(),
            stdin_path: String::new(),
            cpus: 1,
            memory_mb: 0,
            gpu_devices: Vec::new(),
            cpu_ids: Vec::new(),
            open_mode: None,
            uid: 0,
            gid: 0,
            container: None,
            prolog_script: None,
            partition: String::new(),
            nodelist: String::new(),
            host_device_plan: None,
            memlock: MemlockLimit::Unlimited,
            cgroup: CgroupConfig::default(),
            io_mode: LaunchIo::File,
            pmix_multi_task: false,
        }
    }

    #[test]
    fn test_resolve_output_path() {
        let cfg = launch_cfg_for_paths(42, "train", "alice", "node7");
        assert_eq!(
            resolve_output_path(&cfg, "/home/user", "spur-%j.out"),
            "/home/user/spur-42.out"
        );
        assert_eq!(
            resolve_output_path(&cfg, "/home/user", "/var/log/job-%j.log"),
            "/var/log/job-42.log"
        );
        assert_eq!(resolve_output_path(&cfg, "/tmp", ""), "/tmp/spur-42.out");
        // Same codes as the controller (%x/%u/%N), so reported/computed never diverge.
        assert_eq!(
            resolve_output_path(&cfg, "/tmp", "out-%x-%u-%N.log"),
            "/tmp/out-train-alice-node7.log"
        );
    }

    #[test]
    fn cgroup_oom_killed_parses_memory_events() {
        let dir = tempfile::tempdir().unwrap();
        // Missing file (no cgroup isolation) -> not OOM.
        assert!(!cgroup_oom_killed(dir.path()));
        // oom_kill 0 -> not OOM.
        std::fs::write(
            dir.path().join("memory.events"),
            "low 0\nhigh 0\nmax 5\noom 0\noom_kill 0\n",
        )
        .unwrap();
        assert!(!cgroup_oom_killed(dir.path()));
        // oom_kill > 0 -> OOM.
        std::fs::write(
            dir.path().join("memory.events"),
            "low 0\nhigh 0\nmax 12\noom 1\noom_kill 1\n",
        )
        .unwrap();
        assert!(cgroup_oom_killed(dir.path()));
    }

    #[test]
    fn test_burst_buffer_wrap_stage_in_only() {
        let script = "#!/bin/bash\necho hello\n";
        let bb = "stage_in:cp /data/model.bin /tmp/";
        let wrapped = wrap_with_burst_buffer(script, bb);
        assert!(wrapped.contains("cp /data/model.bin /tmp/ || exit 1"));
        assert!(wrapped.contains("echo hello"));
        assert!(wrapped.contains("exit $SPUR_BB_EXIT"));
    }

    #[test]
    fn test_burst_buffer_wrap_stage_out_only() {
        let script = "#!/bin/bash\necho hello\n";
        let bb = "stage_out:cp /tmp/results /data/";
        let wrapped = wrap_with_burst_buffer(script, bb);
        assert!(wrapped.contains("cp /tmp/results /data/ || true"));
        assert!(wrapped.contains("echo hello"));
    }

    #[test]
    fn test_burst_buffer_wrap_both() {
        let script = "#!/bin/bash\necho hello\n";
        let bb = "stage_in:cp /data/in.bin /tmp/;stage_out:cp /tmp/out.bin /data/";
        let wrapped = wrap_with_burst_buffer(script, bb);
        assert!(wrapped.contains("cp /data/in.bin /tmp/ || exit 1"));
        assert!(wrapped.contains("cp /tmp/out.bin /data/ || true"));
        // Stage-in should come before user script, stage-out after
        let stage_in_pos = wrapped.find("stage-in").unwrap();
        let user_pos = wrapped.find("User script").unwrap();
        let stage_out_pos = wrapped.find("stage-out").unwrap();
        assert!(stage_in_pos < user_pos);
        assert!(user_pos < stage_out_pos);
    }

    #[test]
    fn test_burst_buffer_empty_passthrough() {
        let script = "#!/bin/bash\necho hello\n";
        let wrapped = wrap_with_burst_buffer(script, "");
        assert_eq!(wrapped, script);
    }

    #[test]
    fn test_burst_buffer_capacity_directive_ignored_by_wrapper() {
        // The controller consumes `capacity=NNN`; the agent's stage wrapper must
        // ignore it (it's not a stage_in/stage_out command) and only act on the
        // stage directive. The shared parser owns the capacity grammar.
        let script = "#!/bin/bash\necho run\n";
        let bb = "capacity=128;stage_in:cp /data /tmp";
        let wrapped = wrap_with_burst_buffer(script, bb);
        assert!(wrapped.contains("cp /data /tmp"));
        assert!(!wrapped.contains("capacity=128"));
        assert_eq!(spur_core::burst_buffer::parse_capacity_gb(bb), 128);
    }

    #[test]
    fn test_burst_buffer_capacity_only_is_passthrough() {
        // A BB spec with only a capacity reservation (no stage commands) leaves
        // the script unwrapped — there is nothing for the agent to run.
        let script = "#!/bin/bash\necho run\n";
        let wrapped = wrap_with_burst_buffer(script, "capacity=64");
        assert_eq!(wrapped, script);
    }

    /// Issue #128: when uid > 0, the wrapper must drop privilege via setpriv
    /// *after* the mounts (which need CAP_SYS_ADMIN). Dropping priv before
    /// unshare would cause unshare(2) to fail with EPERM.
    #[test]
    fn test_namespace_wrapper_drops_priv_via_setpriv() {
        let script = PathBuf::from("/work/.spur_job_42.sh");
        let wrapper = build_namespace_wrapper(1000, 1000, &[], &script);

        // setpriv must appear with both --reuid and --regid plus --init-groups
        // (so video/render supplementary groups are picked up for GPU access).
        assert!(
            wrapper.contains("setpriv --reuid=1000 --regid=1000 --init-groups"),
            "wrapper missing setpriv invocation: {wrapper}"
        );
        // The setpriv exec must be the *last* exec, after the mount commands.
        let mount_pos = wrapper.find("mount -t proc").expect("missing proc mount");
        let setpriv_pos = wrapper.find("setpriv").expect("missing setpriv");
        assert!(
            mount_pos < setpriv_pos,
            "mounts must run before priv drop:\n{wrapper}"
        );
        // No bare `exec /bin/bash` slip-through that would run as root.
        assert!(
            !wrapper.contains("exec /bin/bash /work"),
            "uid>0 wrapper must not exec bash directly as root:\n{wrapper}"
        );
    }

    /// When uid == 0 (root job), no priv drop is needed and the wrapper exec's
    /// bash directly.
    #[test]
    fn test_namespace_wrapper_root_no_setpriv() {
        let script = PathBuf::from("/work/.spur_job_7.sh");
        let wrapper = build_namespace_wrapper(0, 0, &[], &script);

        assert!(
            !wrapper.contains("setpriv"),
            "root job should not invoke setpriv:\n{wrapper}"
        );
        assert!(
            wrapper.contains("exec /bin/bash /work/.spur_job_7.sh"),
            "root wrapper should exec the job script directly:\n{wrapper}"
        );
    }

    /// GPU device restriction lines are emitted for each allocated DRI device.
    #[test]
    fn test_namespace_wrapper_gpu_mounts() {
        let script = PathBuf::from("/work/.spur_job_1.sh");
        let paths = vec!["/dev/dri/renderD128".into(), "/dev/dri/renderD130".into()];
        let wrapper = build_namespace_wrapper(1000, 1000, &paths, &script);

        assert!(wrapper.contains("renderD128"));
        assert!(wrapper.contains("renderD130"));
        assert!(!wrapper.contains("renderD129"));
        assert!(!wrapper.contains("renderD131"));
    }

    /// Non-DRI paths (e.g. /dev/nvidia*) are skipped — they can't be isolated
    /// via the /dev/dri tmpfs trick; env vars handle visibility instead.
    #[test]
    fn test_namespace_wrapper_ignores_non_dri_paths() {
        let script = PathBuf::from("/work/.spur_job_5.sh");
        let paths = vec![
            "/dev/nvidia0".into(),
            "/dev/nvidiactl".into(),
            "/dev/nvidia-uvm".into(),
            "/dev/dri/renderD128".into(),
        ];
        let wrapper = build_namespace_wrapper(1000, 1000, &paths, &script);

        assert!(wrapper.contains("renderD128"));
        assert!(!wrapper.contains("nvidia"));
    }

    /// A bulk `cp -a /dev/dri/.` recreates every host node with `mknod(2)`, which the
    /// filter denies — and that failure must not skip the tmpfs that hides them.
    #[test]
    fn test_namespace_wrapper_stages_only_allocated_dri_nodes() {
        let script = PathBuf::from("/work/.spur_job_9.sh");
        let paths = vec!["/dev/dri/renderD128".into()];
        let wrapper = build_namespace_wrapper(1000, 1000, &paths, &script);

        assert!(
            !wrapper.contains("/dev/dri/."),
            "no bulk copy of the host directory:\n{wrapper}"
        );
        assert!(
            wrapper.contains("cp -a /dev/dri/renderD128 $SPUR_HOST_DRI/renderD128"),
            "the job's own node is staged:\n{wrapper}"
        );
        assert!(
            wrapper.contains("cp -a $SPUR_HOST_DRI/renderD128 /dev/dri/renderD128"),
            "and restored onto the tmpfs:\n{wrapper}"
        );
        let mount = wrapper
            .find("mount -t tmpfs tmpfs /dev/dri")
            .expect("missing /dev/dri tmpfs mount");
        let restore = wrapper
            .find("cp -a $SPUR_HOST_DRI/")
            .expect("missing restore");
        assert!(
            mount < restore,
            "the tmpfs must be mounted before nodes are restored onto it:\n{wrapper}"
        );
        assert!(
            wrapper.contains("if mount -t tmpfs tmpfs /dev/dri 2>/dev/null; then"),
            "and the restore must be gated on that mount: against the host's real \
             /dev/dri, `cp` unlinks the node before recreating it:\n{wrapper}"
        );
    }

    /// A job allocated no render nodes must still get the empty tmpfs; skipping the
    /// mount would leave every GPU on the node visible in `/dev/dri`.
    #[test]
    fn test_namespace_wrapper_mounts_empty_dri_tmpfs_without_gpus() {
        let script = PathBuf::from("/work/.spur_job_10.sh");
        let wrapper = build_namespace_wrapper(1000, 1000, &[], &script);

        assert!(
            wrapper.contains("mount -t tmpfs tmpfs /dev/dri"),
            "zero-GPU job must still hide the host's /dev/dri:\n{wrapper}"
        );
        assert!(
            !wrapper.contains("cp -a"),
            "nothing is staged for a job with no allocated nodes:\n{wrapper}"
        );
    }

    /// The security-critical direction: a job launched without an injection plan
    /// was allocated nothing, so the filter must be built from an empty list.
    #[test]
    fn no_injection_plan_yields_no_device_paths() {
        assert!(allocated_device_paths(None).is_empty());

        let plan = spur_devices::inject::HostInjectionPlan {
            device_paths: vec!["/dev/dri/renderD128".to_string()],
            ..Default::default()
        };
        assert_eq!(
            allocated_device_paths(Some(&plan)),
            ["/dev/dri/renderD128"],
            "a plan's paths reach the filter unchanged"
        );
    }

    #[tokio::test]
    async fn jobio_wire_pty() {
        let (master, slave) = crate::pty::openpty_with_winsize(Some(&crate::pty::WindowSize {
            rows: 24,
            cols: 80,
            xpixel: 0,
            ypixel: 0,
        }))
        .expect("openpty");

        let job_io = JobIo::Pty { master, slave };
        let raw = job_io.raw();

        let mut cmd = Command::new("/bin/echo");
        cmd.arg("pty_test_output")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        unsafe {
            cmd.pre_exec(move || raw.wire());
        }

        let mut child = cmd.spawn().expect("spawn");
        let master_fd = job_io.into_master().expect("PTY must have master");

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let mut buf = [0u8; 256];
        let n = unsafe { libc::read(master_fd.as_raw_fd(), buf.as_mut_ptr() as *mut _, buf.len()) };
        assert!(n > 0, "expected output from PTY master");
        let output = String::from_utf8_lossy(&buf[..n as usize]);
        assert!(
            output.contains("pty_test_output"),
            "expected 'pty_test_output' in output, got: {output}"
        );

        let status = child.wait().await.expect("wait");
        assert!(status.success());
    }

    #[tokio::test]
    async fn jobio_wire_file() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("stdout");
        let err_path = dir.path().join("stderr");

        let out_file = std::fs::File::create(&out_path).unwrap();
        let err_file = std::fs::File::create(&err_path).unwrap();

        let job_io = JobIo::File {
            stdin: None,
            stdout: OwnedFd::from(out_file),
            stderr: OwnedFd::from(err_file),
        };
        let raw = job_io.raw();

        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c")
            .arg("echo file_stdout; echo file_stderr >&2")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        unsafe {
            cmd.pre_exec(move || raw.wire());
        }

        let mut child = cmd.spawn().expect("spawn");
        assert!(job_io.into_master().is_none(), "File mode has no master");

        let status = child.wait().await.expect("wait");
        assert!(status.success());

        let stdout = std::fs::read_to_string(&out_path).unwrap();
        let stderr = std::fs::read_to_string(&err_path).unwrap();
        assert!(
            stdout.contains("file_stdout"),
            "expected 'file_stdout' in stdout, got: {stdout}"
        );
        assert!(
            stderr.contains("file_stderr"),
            "expected 'file_stderr' in stderr, got: {stderr}"
        );
    }

    #[test]
    fn wire_file_closes_originals_gt_2() {
        // After wire(), originals > 2 should be closed. Verify by checking that
        // a write to the original fd fails with EBADF.
        use std::os::fd::AsRawFd;
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("out");
        let err_path = dir.path().join("err");

        let out_file = std::fs::File::create(&out_path).unwrap();
        let err_file = std::fs::File::create(&err_path).unwrap();
        let out_fd = out_file.as_raw_fd();
        let err_fd = err_file.as_raw_fd();

        // Both fds should be > 2 since 0/1/2 are taken.
        assert!(out_fd > 2);
        assert!(err_fd > 2);

        // Fork so we don't corrupt our own stdio.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed");

        if pid == 0 {
            let raw = JobIoRaw::File {
                stdin: None,
                stdout: out_fd,
                stderr: err_fd,
            };
            let result = unsafe { raw.wire() };
            // Exit with code 0 on success, 1 on failure.
            std::process::exit(if result.is_ok() { 0 } else { 1 });
        }

        // Parent: wait for child.
        let mut status = 0i32;
        unsafe { libc::waitpid(pid, &mut status, 0) };
        assert!(
            libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
            "child exited with non-zero status"
        );
    }

    #[test]
    fn wire_file_bad_fd_returns_error() {
        let raw = JobIoRaw::File {
            stdin: None,
            stdout: -1,
            stderr: -1,
        };
        // Fork to avoid clobbering test process stdio.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed");

        if pid == 0 {
            let result = unsafe { raw.wire() };
            std::process::exit(if result.is_err() { 0 } else { 1 });
        }

        let mut status = 0i32;
        unsafe { libc::waitpid(pid, &mut status, 0) };
        assert!(
            libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
            "wire() should have returned an error for bad fd"
        );
    }

    #[test]
    fn join_cgroup_self_writes_own_pid() {
        // cgroup.procs is a plain "write my decimal pid" interface, so a regular
        // file exercises the hand-rolled formatting and write without a cgroup.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cgroup.procs");
        // The real cgroup.procs is kernel-created; join_cgroup_self opens it
        // without O_CREAT, so the stand-in must exist first.
        std::fs::write(&path, "").unwrap();
        let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();

        assert!(
            join_cgroup_self(&c_path, -1),
            "a successful write must report joined"
        );

        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, std::process::id().to_string());
    }

    #[test]
    fn join_cgroup_self_reports_failure_when_the_path_is_missing() {
        // A non-existent cgroup.procs must not panic and must report the miss, so the
        // caller can degrade (best-effort) or abort (`required`) as it chooses.
        let c_path = CString::new("/nonexistent/spur-test/cgroup.procs").unwrap();
        assert!(
            !join_cgroup_self(&c_path, -1),
            "a missing cgroup must report not joined"
        );
    }

    #[test]
    fn join_cgroup_self_warns_to_log_fd_on_failure() {
        // A failed placement must surface on the log fd so an unenforced limit is
        // not silent; the child's own stderr points at the job, hence a dup here.
        let mut fds = [0 as libc::c_int; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let (read_fd, write_fd) = (fds[0], fds[1]);
        let c_path = CString::new("/nonexistent/spur-test/cgroup.procs").unwrap();

        assert!(
            !join_cgroup_self(&c_path, write_fd),
            "the failed join must be reported"
        );
        unsafe { libc::close(write_fd) };

        let mut buf = [0u8; 128];
        let n = unsafe { libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        unsafe { libc::close(read_fd) };
        assert!(n > 0, "expected a warning on the log fd");
        let msg = std::str::from_utf8(&buf[..n as usize]).unwrap();
        assert!(
            msg.contains("failed to join cgroup"),
            "unexpected message: {msg}"
        );
    }
}
