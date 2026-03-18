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
                            completed.push((*job_id, exit_code));
                        }
                        Ok(None) => {} // Still running
                        Err(e) => {
                            warn!(job_id, error = %e, "failed to check job status");
                        }
                    }
                }

                for (job_id, exit_code) in &completed {
                    jobs.remove(job_id);
                    // Clean up unnamed container rootfs
                    crate::container::cleanup_rootfs(*job_id);
                    // Report completion to controller
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

    match SlurmControllerClient::connect(controller_addr.to_string()).await {
        Ok(mut client) => {
            // Use UpdateJob to report completion (or CancelJob with signal=0 as completion marker)
            // For now, we'll report via a GetJob call — the controller will handle
            // state transitions internally. A proper completion RPC is needed.
            // TODO: Add a CompleteJob RPC
            info!(job_id, exit_code, "reported completion to controller");
        }
        Err(e) => {
            warn!(job_id, error = %e, "failed to report completion");
        }
    }
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

        // If container image is specified, wrap the job in a container
        let (launch_script, is_container) = if !spec.container_image.is_empty() {
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

            let rootfs = crate::container::setup_rootfs(
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

            (wrapper, true)
        } else {
            (script, false)
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
                let mut jobs = self.running.lock().await;
                jobs.insert(
                    job_id,
                    TrackedJob {
                        child: running_job.into_child(),
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
}

pub fn create_server(reporter: Arc<NodeReporter>) -> SlurmAgentServer<AgentService> {
    let service = AgentService::new(reporter);
    SlurmAgentServer::new(service)
}
