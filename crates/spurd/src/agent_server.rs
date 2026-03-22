//! gRPC server implementing the SlurmAgent service.
//! Receives job launch/cancel requests from spurctld.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};

use tokio_stream::wrappers::ReceiverStream;

use spur_proto::proto::slurm_agent_server::{SlurmAgent, SlurmAgentServer};
use spur_proto::proto::*;

use spur_sched::cons_tres::{AllocationResult, NodeAllocation};

use spur_spank::{SpankHook, SpankHost};

use crate::executor;
use crate::reporter::NodeReporter;

/// Running job handle for tracking.
struct TrackedJob {
    child: tokio::process::Child,
    /// PID of the container init process (for nsenter/exec).
    pid: Option<u32>,
    /// How the container rootfs was set up (for cleanup).
    rootfs_mode: crate::container::RootfsMode,
    /// GPU/CPU allocation result for release on completion.
    allocation: Option<AllocationResult>,
    /// Stdout path for output streaming.
    stdout_path: String,
    /// Stderr path for output streaming.
    stderr_path: String,
}

pub struct AgentService {
    pub reporter: Arc<NodeReporter>,
    running: Arc<Mutex<HashMap<u32, TrackedJob>>>,
    allocation: Arc<Mutex<NodeAllocation>>,
    spank: Arc<Option<SpankHost>>,
}

impl AgentService {
    pub fn new(reporter: Arc<NodeReporter>) -> Self {
        let allocation = NodeAllocation::new(
            hostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|_| "unknown".into()),
            &reporter.resources,
        );

        // Load SPANK plugins from plugstack.conf if available
        let plugstack_path = std::env::var("SPUR_PLUGSTACK")
            .unwrap_or_else(|_| "/etc/spur/plugstack.conf".to_string());
        let spank = if std::path::Path::new(&plugstack_path).exists() {
            match spur_spank::parse_plugstack(std::path::Path::new(&plugstack_path)) {
                Ok(entries) => {
                    let mut host = SpankHost::new();
                    for entry in &entries {
                        if let Err(e) = host.load_plugin(&entry.path) {
                            if entry.required {
                                warn!(
                                    plugin = %entry.path.display(),
                                    error = %e,
                                    "required SPANK plugin failed to load"
                                );
                            } else {
                                info!(
                                    plugin = %entry.path.display(),
                                    error = %e,
                                    "optional SPANK plugin failed to load, skipping"
                                );
                            }
                        }
                    }
                    if host.plugin_count() > 0 {
                        info!(count = host.plugin_count(), "SPANK plugins loaded");
                        Some(host)
                    } else {
                        None
                    }
                }
                Err(e) => {
                    warn!(
                        path = %plugstack_path,
                        error = %e,
                        "failed to parse plugstack.conf"
                    );
                    None
                }
            }
        } else {
            None
        };

        Self {
            reporter,
            running: Arc::new(Mutex::new(HashMap::new())),
            allocation: Arc::new(Mutex::new(allocation)),
            spank: Arc::new(spank),
        }
    }

    /// Spawn a background task to monitor running jobs and report completions.
    pub fn start_monitor(&self, controller_addr: String) {
        let running = self.running.clone();
        let allocation = self.allocation.clone();
        let spank = self.spank.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(2));
            loop {
                interval.tick().await;
                let mut jobs = running.lock().await;
                let mut completed: Vec<(
                    u32,
                    i32,
                    crate::container::RootfsMode,
                    Option<AllocationResult>,
                )> = Vec::new();

                for (job_id, tracked) in jobs.iter_mut() {
                    match tracked.child.try_wait() {
                        Ok(Some(status)) => {
                            let exit_code = status.code().unwrap_or(-1);
                            info!(job_id, exit_code, "job finished");
                            completed.push((
                                *job_id,
                                exit_code,
                                tracked.rootfs_mode.clone(),
                                tracked.allocation.take(),
                            ));
                        }
                        Ok(None) => {} // Still running
                        Err(e) => {
                            warn!(job_id, error = %e, "failed to check job status");
                        }
                    }
                }

                for (job_id, _exit_code, mode, alloc) in &completed {
                    jobs.remove(job_id);
                    crate::container::cleanup_rootfs(*job_id, mode);
                    // Release GPU/CPU allocation
                    if let Some(alloc) = alloc {
                        allocation.lock().await.release(alloc);
                    }
                }

                // Release lock BEFORE network I/O — holding the lock during
                // report_completion blocks new job launches and can lose
                // completions if the RPC times out.
                drop(jobs);

                // Invoke SPANK TaskExit and JobEpilog hooks for completed jobs
                if let Some(ref spank_host) = *spank {
                    for (job_id, _exit_code, _mode, _alloc) in &completed {
                        if let Err(e) = spank_host.invoke_hook(SpankHook::TaskExit) {
                            warn!(job_id, error = %e, "SPANK TaskExit hook failed");
                        }
                        if let Err(e) = spank_host.invoke_hook(SpankHook::JobEpilog) {
                            warn!(job_id, error = %e, "SPANK JobEpilog hook failed");
                        }
                    }
                }

                for (job_id, exit_code, _mode, _alloc) in &completed {
                    report_completion(&controller_addr, *job_id, *exit_code).await;
                }
            }
        });
    }
}

async fn report_completion(controller_addr: &str, job_id: u32, exit_code: i32) {
    use spur_proto::proto::slurm_controller_client::SlurmControllerClient;

    let state = if exit_code == 0 {
        JobState::JobCompleted as i32
    } else {
        JobState::JobFailed as i32
    };

    let url = if controller_addr.starts_with("http") {
        controller_addr.to_string()
    } else {
        format!("http://{}", controller_addr)
    };

    // Retry up to 3 times with 1-second backoff — a single transient failure
    // must not permanently lose a job completion.
    for attempt in 1..=3 {
        match SlurmControllerClient::connect(url.clone()).await {
            Ok(mut client) => {
                let req = ReportJobStatusRequest {
                    job_id,
                    state,
                    exit_code,
                    message: format!("exit_code={}", exit_code),
                };
                match client.report_job_status(req).await {
                    Ok(_) => {
                        info!(job_id, exit_code, "reported completion to controller");
                        return;
                    }
                    Err(e) => {
                        warn!(
                            job_id,
                            attempt,
                            error = %e,
                            "ReportJobStatus RPC failed"
                        );
                    }
                }
            }
            Err(e) => {
                warn!(
                    job_id,
                    attempt,
                    error = %e,
                    "failed to connect to controller for completion report"
                );
            }
        }
        if attempt < 3 {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }
    error!(
        job_id,
        exit_code, "gave up reporting completion after 3 attempts"
    );
}

#[tonic::async_trait]
impl SlurmAgent for AgentService {
    type StreamJobOutputStream = ReceiverStream<Result<StreamJobOutputChunk, Status>>;

    async fn launch_job(
        &self,
        request: Request<LaunchJobRequest>,
    ) -> Result<Response<LaunchJobResponse>, Status> {
        let req = request.into_inner();
        let job_id = req.job_id;
        let peer_nodes = req.peer_nodes;
        let task_offset = req.task_offset;
        let spec = req
            .spec
            .ok_or_else(|| Status::invalid_argument("missing job spec"))?;

        info!(
            job_id,
            name = %spec.name,
            task_offset,
            num_peers = peer_nodes.len(),
            "received job launch request"
        );

        let work_dir = if spec.work_dir.is_empty() {
            "/tmp".to_string()
        } else {
            spec.work_dir.clone()
        };

        let script = if spec.script.is_empty() {
            if spec.argv.is_empty() {
                return Err(Status::invalid_argument("no script or argv"));
            }
            // Build a script from argv
            let mut s = String::from("#!/bin/bash\n");
            s.push_str(&spec.argv.join(" "));
            s.push('\n');
            s
        } else {
            spec.script.clone()
        };

        // Inject peer node info as environment variables for MPI/distributed apps
        let mut env = spec.environment.clone();
        env.insert("SPUR_JOB_ID".into(), job_id.to_string());
        env.insert("SPUR_TASK_OFFSET".into(), task_offset.to_string());
        env.insert("SPUR_NUM_NODES".into(), peer_nodes.len().to_string());
        if !peer_nodes.is_empty() {
            env.insert("SPUR_PEER_NODES".into(), peer_nodes.join(","));
        }
        if !req.target_node.is_empty() {
            env.insert("SPUR_TARGET_NODE".into(), req.target_node.clone());
        }

        // Compute tasks_per_node for both single- and multi-node jobs
        let tasks_per_node = if spec.tasks_per_node > 0 {
            spec.tasks_per_node
        } else {
            (spec.num_tasks / spec.num_nodes.max(1)).max(1)
        };
        let node_rank = task_offset / tasks_per_node.max(1);

        // LOCAL_RANK / LOCAL_WORLD_SIZE — always set, even for single-node jobs
        env.insert("LOCAL_RANK".into(), "0".to_string()); // Single process per node (multi-task wrapper overrides per-process)
        env.insert("LOCAL_WORLD_SIZE".into(), tasks_per_node.to_string());
        env.insert("NPROC_PER_NODE".into(), tasks_per_node.to_string());
        env.insert("NODE_RANK".into(), node_rank.to_string());

        // PyTorch/NCCL/RCCL distributed training env vars
        if peer_nodes.len() > 1 {
            // MASTER_ADDR: first peer node's address (strip port)
            if let Some(first_peer) = peer_nodes.first() {
                let master_addr = first_peer
                    .rsplit(':')
                    .nth(1)
                    .or_else(|| first_peer.split(':').next())
                    .unwrap_or(first_peer);
                env.insert("MASTER_ADDR".into(), master_addr.to_string());
            }
            env.insert("MASTER_PORT".into(), "29500".to_string());
            env.insert("WORLD_SIZE".into(), peer_nodes.len().to_string());

            // RANK = node index within peer list
            env.insert("RANK".into(), node_rank.to_string());
            env.insert("SPUR_NODE_RANK".into(), node_rank.to_string());
        }

        // If container image is specified, wrap the job in a container
        let (launch_script, rootfs_mode) = if !spec.container_image.is_empty() {
            info!(job_id, image = %spec.container_image, "launching containerized job");

            let mounts: Vec<crate::container::BindMount> = spec
                .container_mounts
                .iter()
                .filter_map(|m| crate::container::parse_mount(m).ok())
                .collect();

            // Resolve user info for shadow hook
            let username = spec.user.clone();
            let uid = spec.uid;
            let gid = spec.gid;
            let home_dir = std::env::var("HOME").unwrap_or_else(|_| format!("/home/{}", username));

            let container_config = crate::container::ContainerConfig {
                image: spec.container_image.clone(),
                mounts,
                workdir: if spec.container_workdir.is_empty() {
                    None
                } else {
                    Some(spec.container_workdir.clone())
                },
                name: if spec.container_name.is_empty() {
                    None
                } else {
                    Some(spec.container_name.clone())
                },
                readonly: spec.container_readonly,
                mount_home: spec.container_mount_home,
                remap_root: spec.container_remap_root,
                gpu_devices: vec![], // TODO: from GRES allocation
                environment: env.clone(),
                container_env: spec.container_env.clone(),
                entrypoint: if spec.container_entrypoint.is_empty() {
                    None
                } else {
                    Some(spec.container_entrypoint.clone())
                },
                uid,
                gid,
                username: if username.is_empty() {
                    "spur".to_string()
                } else {
                    username
                },
                home_dir,
            };

            let image_path = crate::container::resolve_image(&spec.container_image)
                .map_err(|e| Status::failed_precondition(e.to_string()))?;

            let (rootfs, rootfs_mode) = crate::container::setup_rootfs(
                &image_path,
                job_id,
                container_config.name.as_deref(),
            )
            .map_err(|e| Status::internal(format!("container setup failed: {}", e)))?;

            // Write the user's actual script to a separate file
            // (the executor will write the *wrapper* as .spur_job_{id}.sh)
            let inner_script_path = format!("{}/.spur_inner_{}.sh", work_dir, job_id);
            std::fs::write(&inner_script_path, &script)
                .map_err(|e| Status::internal(format!("failed to write inner script: {}", e)))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(
                    &inner_script_path,
                    std::fs::Permissions::from_mode(0o755),
                );
            }

            let wrapper = crate::container::build_container_launch_script(
                &container_config,
                &rootfs,
                &inner_script_path,
                job_id,
            )
            .map_err(|e| Status::internal(format!("container script failed: {}", e)))?;

            (wrapper, rootfs_mode)
        } else {
            (script, crate::container::RootfsMode::Extracted)
        };

        // Multi-task per-node: wrap the user script so it forks N processes,
        // each with a distinct LOCAL_RANK. The wrapper backgrounds N copies and
        // waits for all to finish, so TrackedJob only tracks a single PID (the
        // wrapper shell). GPU devices are partitioned across tasks via
        // ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES overrides in each fork.
        let launch_script = if tasks_per_node > 1 {
            // Write the user script to disk first so the wrapper can reference it
            let user_script_path = format!("{}/.spur_user_{}.sh", work_dir, job_id);
            std::fs::write(&user_script_path, &launch_script)
                .map_err(|e| Status::internal(format!("failed to write user script: {}", e)))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(
                    &user_script_path,
                    std::fs::Permissions::from_mode(0o755),
                );
            }

            // Build the wrapper that launches N tasks with GPU partitioning
            let mut wrapper = String::from("#!/bin/bash\n");
            wrapper.push_str(&format!(
                "SPUR_NTASKS={}\nSPUR_TASK_OFFSET=${{SPUR_TASK_OFFSET:-0}}\n",
                tasks_per_node
            ));
            wrapper.push_str("for LOCAL_RANK in $(seq 0 $((SPUR_NTASKS - 1))); do\n");
            wrapper.push_str("  export LOCAL_RANK\n");
            wrapper.push_str("  export SPUR_LOCALID=$LOCAL_RANK\n");
            wrapper.push_str(
                "  export SPUR_PROCID=$((SPUR_TASK_OFFSET + LOCAL_RANK))\n",
            );
            wrapper.push_str("  export PMI_RANK=$SPUR_PROCID\n");

            // Partition GPUs across tasks if GPUs are allocated
            wrapper.push_str("  if [ -n \"$SPUR_JOB_GPUS\" ]; then\n");
            wrapper.push_str("    IFS=',' read -ra _ALL_GPUS <<< \"$SPUR_JOB_GPUS\"\n");
            wrapper.push_str(
                "    _GPUS_PER_TASK=$(( ${#_ALL_GPUS[@]} / SPUR_NTASKS ))\n",
            );
            wrapper.push_str("    if [ $_GPUS_PER_TASK -gt 0 ]; then\n");
            wrapper.push_str(
                "      _START=$((LOCAL_RANK * _GPUS_PER_TASK))\n",
            );
            wrapper.push_str(
                "      _TASK_GPUS=$(echo \"${_ALL_GPUS[@]:$_START:$_GPUS_PER_TASK}\" | tr ' ' ',')\n",
            );
            wrapper.push_str("      export ROCR_VISIBLE_DEVICES=$_TASK_GPUS\n");
            wrapper.push_str("      export CUDA_VISIBLE_DEVICES=$_TASK_GPUS\n");
            wrapper.push_str("      export GPU_DEVICE_ORDINAL=$_TASK_GPUS\n");
            wrapper.push_str("    fi\n");
            wrapper.push_str("  fi\n");

            wrapper.push_str(&format!(
                "  bash \"{}\" &\n",
                user_script_path.replace('"', "\\\"")
            ));
            wrapper.push_str("done\nwait\n");
            wrapper
        } else {
            launch_script
        };

        // Allocate GPU devices from the node's pool
        let mut gpu_count = 0u32;
        let mut gpu_type: Option<String> = None;
        for gres in &spec.gres {
            if let Some((name, gtype, count)) = spur_core::resource::parse_gres(gres) {
                if name == "gpu" {
                    gpu_count += count;
                    if let Some(t) = gtype {
                        gpu_type = Some(t);
                    }
                }
            }
        }

        let alloc_result = if gpu_count > 0 || spec.cpus_per_task > 0 {
            let mut alloc = self.allocation.lock().await;
            alloc.try_allocate(
                spec.cpus_per_task.max(1),
                spec.memory_per_node_mb,
                gpu_count,
                gpu_type.as_deref(),
            )
        } else {
            None
        };

        let gpu_devices: Vec<u32> = alloc_result
            .as_ref()
            .map(|a| a.gpu_ids.clone())
            .unwrap_or_default();

        let cpu_ids: Vec<u32> = alloc_result
            .as_ref()
            .map(|a| a.cpu_ids.clone())
            .unwrap_or_default();

        // Resolve stdout/stderr paths
        let stdout_path = if spec.stdout_path.is_empty() {
            format!("{}/spur-{}.out", work_dir, job_id)
        } else {
            spec.stdout_path.clone()
        };
        let stderr_path = if spec.stderr_path.is_empty() {
            format!("{}/spur-{}.out", work_dir, job_id)
        } else {
            spec.stderr_path.clone()
        };

        // Launch the job
        match executor::launch_job(
            job_id,
            &launch_script,
            &work_dir,
            &env,
            &stdout_path,
            &stderr_path,
            spec.cpus_per_task.max(1),
            spec.memory_per_node_mb,
            &gpu_devices,
            &cpu_ids,
            (*self.spank).as_ref(),
        )
        .await
        {
            Ok(running_job) => {
                let child = running_job.into_child();
                let pid = child.id();
                let mut jobs = self.running.lock().await;
                jobs.insert(
                    job_id,
                    TrackedJob {
                        child,
                        pid,
                        rootfs_mode: rootfs_mode.clone(),
                        allocation: alloc_result,
                        stdout_path,
                        stderr_path,
                    },
                );
                info!(job_id, gpus = ?gpu_devices, "job launched successfully");
                Ok(Response::new(LaunchJobResponse {
                    success: true,
                    error: String::new(),
                }))
            }
            Err(e) => {
                // Release allocation on launch failure
                if let Some(ref alloc) = alloc_result {
                    self.allocation.lock().await.release(alloc);
                }
                error!(job_id, error = %e, "failed to launch job");
                Ok(Response::new(LaunchJobResponse {
                    success: false,
                    error: e.to_string(),
                }))
            }
        }
    }

    async fn cancel_job(
        &self,
        request: Request<AgentCancelJobRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let job_id = req.job_id;
        let signal = req.signal;

        let pid = {
            let jobs = self.running.lock().await;
            jobs.get(&job_id).and_then(|t| t.pid)
        };

        let Some(pid) = pid else {
            return Ok(Response::new(()));
        };

        if signal > 0 {
            // Send the specific signal requested (e.g., SIGTERM=15, SIGUSR1=10)
            info!(job_id, signal, "sending signal to job");
            let sig = nix::sys::signal::Signal::try_from(signal)
                .unwrap_or(nix::sys::signal::Signal::SIGTERM);
            let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), sig);
        } else {
            // Default: graceful shutdown — SIGTERM, wait 5s, then SIGKILL
            info!(job_id, "graceful cancel: SIGTERM → 5s grace → SIGKILL");
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGTERM,
            );

            let running = self.running.clone();
            tokio::spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                let mut jobs = running.lock().await;
                if let Some(tracked) = jobs.get_mut(&job_id) {
                    // Still running after grace period — force kill
                    info!(job_id, "grace period expired, sending SIGKILL");
                    let _ = tracked.child.kill().await;
                    jobs.remove(&job_id);
                }
            });
        }

        Ok(Response::new(()))
    }

    async fn get_node_resources(
        &self,
        _request: Request<()>,
    ) -> Result<Response<NodeResourcesResponse>, Status> {
        let resources = &self.reporter.resources;
        Ok(Response::new(NodeResourcesResponse {
            total: Some(crate::reporter::resource_to_proto(resources)),
            used: Some(ResourceSet::default()),
        }))
    }

    async fn exec_in_job(
        &self,
        request: Request<ExecInJobRequest>,
    ) -> Result<Response<ExecInJobResponse>, Status> {
        let req = request.into_inner();
        let jobs = self.running.lock().await;

        let tracked = jobs.get(&req.job_id).ok_or_else(|| {
            Status::not_found(format!("job {} not running on this node", req.job_id))
        })?;

        let pid = tracked.pid.ok_or_else(|| {
            Status::failed_precondition(format!("job {} has no tracked PID", req.job_id))
        })?;

        if req.command.is_empty() {
            return Err(Status::invalid_argument("no command specified"));
        }

        info!(
            job_id = req.job_id,
            pid,
            command = ?req.command,
            "exec into running job"
        );

        // Use nsenter to enter the job's mount namespace and run the command
        let mut cmd = tokio::process::Command::new("nsenter");
        cmd.args(["--target", &pid.to_string(), "--mount", "--"]);
        cmd.arg(&req.command[0]);
        for arg in &req.command[1..] {
            cmd.arg(arg);
        }

        let output = cmd
            .output()
            .await
            .map_err(|e| Status::internal(format!("nsenter failed: {}", e)))?;

        Ok(Response::new(ExecInJobResponse {
            success: output.status.success(),
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        }))
    }

    async fn stream_job_output(
        &self,
        request: Request<StreamJobOutputRequest>,
    ) -> Result<Response<Self::StreamJobOutputStream>, Status> {
        let req = request.into_inner();
        let job_id = req.job_id;

        // Look up the output file path from the tracked job
        let file_path = {
            let jobs = self.running.lock().await;
            match jobs.get(&job_id) {
                Some(tracked) => {
                    if req.stream == "stderr" {
                        tracked.stderr_path.clone()
                    } else {
                        tracked.stdout_path.clone()
                    }
                }
                None => {
                    return Err(Status::not_found(format!(
                        "job {} not running on this node",
                        job_id
                    )));
                }
            }
        };

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let running = self.running.clone();

        tokio::spawn(async move {
            // Wait for the output file to appear
            let mut waited = 0;
            while !std::path::Path::new(&file_path).exists() && waited < 30 {
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                waited += 1;
            }

            let mut offset = 0u64;
            loop {
                // Read new data from the file
                match tokio::fs::read(&file_path).await {
                    Ok(data) => {
                        if data.len() as u64 > offset {
                            let new_data = data[offset as usize..].to_vec();
                            offset = data.len() as u64;
                            if tx
                                .send(Ok(StreamJobOutputChunk {
                                    data: new_data,
                                    eof: false,
                                }))
                                .await
                                .is_err()
                            {
                                break; // Client disconnected
                            }
                        }
                    }
                    Err(_) => {} // File not ready yet
                }

                // Check if job is still running
                let still_running = running.lock().await.contains_key(&job_id);
                if !still_running {
                    // Final read to get any remaining output
                    if let Ok(data) = tokio::fs::read(&file_path).await {
                        if data.len() as u64 > offset {
                            let _ = tx
                                .send(Ok(StreamJobOutputChunk {
                                    data: data[offset as usize..].to_vec(),
                                    eof: false,
                                }))
                                .await;
                        }
                    }
                    let _ = tx
                        .send(Ok(StreamJobOutputChunk {
                            data: Vec::new(),
                            eof: true,
                        }))
                        .await;
                    break;
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

pub fn create_server(reporter: Arc<NodeReporter>) -> SlurmAgentServer<AgentService> {
    let service = AgentService::new(reporter);
    SlurmAgentServer::new(service)
}
