use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use anyhow::Context;
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;
use tokio::process::Command;
use tracing::{debug, error, info, warn};

use spur_core::job::JobId;
use spur_proto::proto::slurm_controller_client::SlurmControllerClient;

use crate::reporter::NodeReporter;

/// Cgroup root for slurmd-managed jobs.
const CGROUP_ROOT: &str = "/sys/fs/cgroup/spur";

/// Job execution loop: polls controller for assigned jobs and runs them.
pub async fn job_execution_loop(reporter: Arc<NodeReporter>) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
    let mut running: HashMap<JobId, RunningJob> = HashMap::new();

    loop {
        interval.tick().await;

        // Check status of running jobs
        let mut completed = Vec::new();
        for (job_id, rj) in running.iter_mut() {
            if let Some(status) = rj.try_wait().await {
                info!(job_id, exit_code = status, "job completed");
                completed.push((*job_id, status));
            }
        }

        // Report completions to controller
        for (job_id, exit_code) in &completed {
            running.remove(job_id);
            if let Err(e) = report_completion(&reporter.controller_addr, *job_id, *exit_code).await
            {
                warn!(job_id, error = %e, "failed to report job completion");
            }
        }

        // Poll for new jobs (via GetJobs filtered to RUNNING state for our node)
        // In a full implementation, the controller pushes jobs via streaming.
        // For now, we rely on the scheduler_loop in slurmctld to track assignments.
    }
}

/// Report job completion back to the controller.
async fn report_completion(
    controller_addr: &str,
    job_id: JobId,
    exit_code: i32,
) -> anyhow::Result<()> {
    // For now, the controller handles this via the heartbeat/status update path.
    // Full implementation in Phase 5 with streaming heartbeats.
    debug!(job_id, exit_code, "would report completion to controller");
    Ok(())
}

/// A running job process.
pub struct RunningJob {
    job_id: JobId,
    child: tokio::process::Child,
    cgroup_path: Option<PathBuf>,
}

impl RunningJob {
    /// Consume self and return the child process (for tracking by agent server).
    pub fn into_child(self) -> tokio::process::Child {
        self.child
    }
}

impl RunningJob {
    /// Check if the job has finished. Returns exit code if done.
    async fn try_wait(&mut self) -> Option<i32> {
        match self.child.try_wait() {
            Ok(Some(status)) => {
                // Clean up cgroup
                if let Some(ref path) = self.cgroup_path {
                    cleanup_cgroup(path);
                }
                Some(status.code().unwrap_or(-1))
            }
            Ok(None) => None, // Still running
            Err(e) => {
                warn!(job_id = self.job_id, error = %e, "failed to check job status");
                None
            }
        }
    }
}

/// Launch a job script on this node.
pub async fn launch_job(
    job_id: JobId,
    script: &str,
    work_dir: &str,
    environment: &HashMap<String, String>,
    stdout_path: &str,
    stderr_path: &str,
    cpus: u32,
    memory_mb: u64,
    gpu_devices: &[u32],
) -> anyhow::Result<RunningJob> {
    info!(job_id, work_dir, "launching job");

    // Run prolog if configured
    if let Ok(prolog) = std::env::var("SPUR_PROLOG") {
        if !prolog.is_empty() {
            run_hook(&prolog, job_id, work_dir, "prolog").await?;
        }
    }

    // Set up cgroup for isolation
    let cgroup_path = setup_cgroup(job_id, cpus, memory_mb)?;

    // Ensure work_dir exists on this node (the submitted path may only exist on the submitting
    // node). If creation fails (e.g. path is under another user's home), fall back to /tmp so
    // the job can still run; absolute output paths in the spec are unaffected.
    let effective_work_dir: String = if tokio::fs::create_dir_all(work_dir).await.is_ok() {
        work_dir.to_string()
    } else {
        warn!(
            job_id,
            work_dir, "work_dir unavailable on this node, using /tmp"
        );
        "/tmp".to_string()
    };
    let work_dir = effective_work_dir.as_str();

    // Write script to temp file
    let script_path = PathBuf::from(work_dir).join(format!(".spur_job_{}.sh", job_id));
    tokio::fs::write(&script_path, script)
        .await
        .context("failed to write job script")?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&script_path, perms)?;
    }

    // Resolve stdout/stderr paths
    let stdout_resolved = resolve_output_path(stdout_path, job_id, work_dir);
    let stderr_resolved = resolve_output_path(stderr_path, job_id, work_dir);

    // Ensure output directories exist
    if let Some(parent) = Path::new(&stdout_resolved).parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    if let Some(parent) = Path::new(&stderr_resolved).parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }

    let stdout_file = tokio::fs::File::create(&stdout_resolved)
        .await
        .context("failed to create stdout file")?;
    let stderr_file = tokio::fs::File::create(&stderr_resolved)
        .await
        .context("failed to create stderr file")?;

    // Build environment
    let mut env = environment.clone();

    // Set SLURM environment variables
    env.insert("SPUR_JOB_ID".into(), job_id.to_string());
    env.insert("SPUR_JOBID".into(), job_id.to_string());
    env.insert(
        "SPUR_JOB_NODELIST".into(),
        hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "localhost".into()),
    );
    env.insert("SPUR_CPUS_ON_NODE".into(), cpus.to_string());

    // GPU isolation
    if !gpu_devices.is_empty() {
        let gpu_list: String = gpu_devices
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        // Set for both AMD and NVIDIA
        env.insert("ROCR_VISIBLE_DEVICES".into(), gpu_list.clone());
        env.insert("CUDA_VISIBLE_DEVICES".into(), gpu_list.clone());
        env.insert("GPU_DEVICE_ORDINAL".into(), gpu_list);
        env.insert(
            "SPUR_JOB_GPUS".into(),
            gpu_devices
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    }

    // Launch the process
    let mut cmd = Command::new("/bin/bash");
    cmd.arg(&script_path)
        .current_dir(work_dir)
        .envs(&env)
        .stdout(stdout_file.into_std().await)
        .stderr(stderr_file.into_std().await)
        .stdin(Stdio::null());

    // If cgroup is set up, launch via cgexec or write PID to cgroup
    let child = cmd.spawn().context("failed to spawn job process")?;

    // Move process into cgroup
    if let Some(ref cgroup) = cgroup_path {
        if let Some(pid) = child.id() {
            move_to_cgroup(cgroup, pid);
        }
    }

    debug!(
        job_id,
        pid = child.id(),
        script = %script_path.display(),
        "job process spawned"
    );

    Ok(RunningJob {
        job_id,
        child,
        cgroup_path,
    })
}

/// Set up a cgroups v2 hierarchy for a job.
fn setup_cgroup(job_id: JobId, cpus: u32, memory_mb: u64) -> anyhow::Result<Option<PathBuf>> {
    let cgroup_path = PathBuf::from(CGROUP_ROOT).join(format!("job_{}", job_id));

    // Try to create cgroup (may fail without root)
    if std::fs::create_dir_all(&cgroup_path).is_err() {
        debug!(
            job_id,
            "cgroup creation failed (not root?), running without isolation"
        );
        return Ok(None);
    }

    // Set CPU limit (cpu.max: quota period)
    // e.g., 4 CPUs → "400000 100000" (400ms out of 100ms period)
    let quota = cpus as u64 * 100_000;
    let cpu_max = format!("{} 100000", quota);
    if let Err(e) = std::fs::write(cgroup_path.join("cpu.max"), &cpu_max) {
        debug!(error = %e, "failed to set cpu.max");
    }

    // Set memory limit
    if memory_mb > 0 {
        let memory_bytes = memory_mb * 1024 * 1024;
        if let Err(e) = std::fs::write(cgroup_path.join("memory.max"), memory_bytes.to_string()) {
            debug!(error = %e, "failed to set memory.max");
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

/// Move a process into a cgroup.
fn move_to_cgroup(cgroup_path: &Path, pid: u32) {
    let procs_file = cgroup_path.join("cgroup.procs");
    if let Err(e) = std::fs::write(&procs_file, pid.to_string()) {
        debug!(
            pid,
            error = %e,
            "failed to move process to cgroup"
        );
    }
}

/// Clean up a job's cgroup.
fn cleanup_cgroup(cgroup_path: &Path) {
    // Kill any remaining processes
    if let Ok(pids) = std::fs::read_to_string(cgroup_path.join("cgroup.procs")) {
        for pid_str in pids.lines() {
            if let Ok(pid) = pid_str.trim().parse::<i32>() {
                let _ = signal::kill(Pid::from_raw(pid), Signal::SIGKILL);
            }
        }
    }

    // Remove cgroup directory
    if let Err(e) = std::fs::remove_dir(cgroup_path) {
        debug!(error = %e, path = %cgroup_path.display(), "failed to remove cgroup");
    }
}

/// Send a signal to a running job.
pub fn signal_job(job: &RunningJob, sig: Signal) -> anyhow::Result<()> {
    if let Some(pid) = job.child.id() {
        signal::kill(Pid::from_raw(pid as i32), sig).context("failed to signal job process")?;
    }
    Ok(())
}

/// Run a prolog/epilog hook script.
async fn run_hook(
    script_path: &str,
    job_id: JobId,
    work_dir: &str,
    hook_name: &str,
) -> anyhow::Result<()> {
    info!(
        job_id,
        hook = hook_name,
        script = script_path,
        "running hook"
    );

    let status = Command::new(script_path)
        .env("SPUR_JOB_ID", job_id.to_string())
        .env("SPUR_JOB_WORK_DIR", work_dir)
        .current_dir(work_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .with_context(|| format!("{} script failed to execute", hook_name))?;

    if !status.success() {
        anyhow::bail!("{} script exited with {}", hook_name, status);
    }
    Ok(())
}

/// Run epilog after job completion (best-effort).
pub async fn run_epilog(job_id: JobId, work_dir: &str) {
    if let Ok(epilog) = std::env::var("SPUR_EPILOG") {
        if !epilog.is_empty() {
            if let Err(e) = run_hook(&epilog, job_id, work_dir, "epilog").await {
                warn!(job_id, error = %e, "epilog failed");
            }
        }
    }
}

/// Resolve output path patterns (%j → job_id, etc.)
fn resolve_output_path(pattern: &str, job_id: JobId, work_dir: &str) -> String {
    let resolved = if pattern.is_empty() {
        format!("spur-{}.out", job_id)
    } else {
        pattern
            .replace("%j", &job_id.to_string())
            .replace("%J", &job_id.to_string())
    };

    if Path::new(&resolved).is_absolute() {
        resolved
    } else {
        PathBuf::from(work_dir)
            .join(resolved)
            .to_string_lossy()
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_output_path() {
        assert_eq!(
            resolve_output_path("spur-%j.out", 42, "/home/user"),
            "/home/user/spur-42.out"
        );
        assert_eq!(
            resolve_output_path("/var/log/job-%j.log", 42, "/home/user"),
            "/var/log/job-42.log"
        );
        assert_eq!(resolve_output_path("", 42, "/tmp"), "/tmp/spur-42.out");
    }
}
