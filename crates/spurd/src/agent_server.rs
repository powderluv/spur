//! gRPC server implementing the SlurmAgent service.
//! Receives job launch/cancel requests from spurctld.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};

use spur_proto::proto::slurm_agent_server::{SlurmAgent, SlurmAgentServer};
use spur_proto::proto::*;

use crate::executor;
use crate::reporter::NodeReporter;

/// Running job handle for tracking.
struct TrackedJob {
    child: tokio::process::Child,
    /// PID of the container init process (for nsenter/exec).
    pid: Option<u32>,
    /// How the container rootfs was set up (for cleanup).
    rootfs_mode: crate::container::RootfsMode,
}

pub struct AgentService {
    pub reporter: Arc<NodeReporter>,
    running: Arc<Mutex<HashMap<u32, TrackedJob>>>,
}

impl AgentService {
    pub fn new(reporter: Arc<NodeReporter>) -> Self {
        Self {
            reporter,
            running: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Spawn a background task to monitor running jobs and report completions.
    pub fn start_monitor(&self, controller_addr: String) {
        let running = self.running.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(2));
            loop {
                interval.tick().await;
                let mut jobs = running.lock().await;
                let mut completed = Vec::new();

                for (job_id, tracked) in jobs.iter_mut() {
                    match tracked.child.try_wait() {
                        Ok(Some(status)) => {
                            let exit_code = status.code().unwrap_or(-1);
                            info!(job_id, exit_code, "job finished");
                            completed.push((*job_id, exit_code, tracked.rootfs_mode.clone()));
                        }
                        Ok(None) => {} // Still running
                        Err(e) => {
                            warn!(job_id, error = %e, "failed to check job status");
                        }
                    }
                }

                for (job_id, exit_code, mode) in &completed {
                    jobs.remove(job_id);
                    crate::container::cleanup_rootfs(*job_id, mode);
                }

                // Release lock BEFORE network I/O — holding the lock during
                // report_completion blocks new job launches and can lose
                // completions if the RPC times out.
                drop(jobs);

                for (job_id, exit_code, _mode) in &completed {
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

            // RANK = node index within peer list (match by task_offset)
            let tasks_per_node = if spec.tasks_per_node > 0 {
                spec.tasks_per_node
            } else {
                (spec.num_tasks / spec.num_nodes.max(1)).max(1)
            };
            let node_rank = task_offset / tasks_per_node;
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

        // Launch the job
        match executor::launch_job(
            job_id,
            &launch_script,
            &work_dir,
            &env,
            &spec.stdout_path,
            &spec.stderr_path,
            spec.cpus_per_task.max(1),
            spec.memory_per_node_mb,
            &[], // GPU devices — TODO: allocate from GRES
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
                    },
                );
                info!(job_id, "job launched successfully");
                Ok(Response::new(LaunchJobResponse {
                    success: true,
                    error: String::new(),
                }))
            }
            Err(e) => {
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
        let mut jobs = self.running.lock().await;

        if let Some(tracked) = jobs.get_mut(&req.job_id) {
            info!(job_id = req.job_id, "cancelling job");
            let _ = tracked.child.kill().await;
            jobs.remove(&req.job_id);
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
}

pub fn create_server(reporter: Arc<NodeReporter>) -> SlurmAgentServer<AgentService> {
    let service = AgentService::new(reporter);
    SlurmAgentServer::new(service)
}
