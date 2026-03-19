use std::net::SocketAddr;
use std::sync::Arc;

use tonic::{Request, Response, Status};

use spur_proto::proto::slurm_controller_server::{SlurmController, SlurmControllerServer};
use spur_proto::proto::*;

use crate::cluster::ClusterManager;

pub struct ControllerService {
    cluster: Arc<ClusterManager>,
}

#[tonic::async_trait]
impl SlurmController for ControllerService {
    async fn submit_job(
        &self,
        request: Request<SubmitJobRequest>,
    ) -> Result<Response<SubmitJobResponse>, Status> {
        let spec = request
            .into_inner()
            .spec
            .ok_or_else(|| Status::invalid_argument("missing job spec"))?;

        let core_spec = proto_to_job_spec(spec)?;
        let job_id = self
            .cluster
            .submit_job(core_spec)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(SubmitJobResponse { job_id }))
    }

    async fn get_jobs(
        &self,
        request: Request<GetJobsRequest>,
    ) -> Result<Response<GetJobsResponse>, Status> {
        let req = request.into_inner();

        let states: Vec<spur_core::job::JobState> = req
            .states
            .iter()
            .filter_map(|s| proto_to_job_state(*s))
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

        let jobs = self
            .cluster
            .get_jobs(&states, user, partition, account, &req.job_ids);

        let proto_jobs: Vec<JobInfo> = jobs.iter().map(job_to_proto).collect();

        Ok(Response::new(GetJobsResponse { jobs: proto_jobs }))
    }

    async fn get_job(&self, request: Request<GetJobRequest>) -> Result<Response<JobInfo>, Status> {
        let job_id = request.into_inner().job_id;
        let job = self
            .cluster
            .get_job(job_id)
            .ok_or_else(|| Status::not_found(format!("job {} not found", job_id)))?;

        Ok(Response::new(job_to_proto(&job)))
    }

    async fn cancel_job(&self, request: Request<CancelJobRequest>) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        self.cluster
            .cancel_job(req.job_id, &req.user)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(()))
    }

    async fn update_job(&self, request: Request<UpdateJobRequest>) -> Result<Response<()>, Status> {
        let req = request.into_inner();

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

        let time_limit = req.time_limit.map(|d| chrono::Duration::seconds(d.seconds));

        self.cluster
            .update_job(
                req.job_id,
                time_limit,
                req.priority,
                req.partition,
                req.comment,
            )
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(()))
    }

    async fn get_nodes(
        &self,
        request: Request<GetNodesRequest>,
    ) -> Result<Response<GetNodesResponse>, Status> {
        let _req = request.into_inner();
        let nodes = self.cluster.get_nodes();
        let proto_nodes: Vec<NodeInfo> = nodes.iter().map(node_to_proto).collect();
        Ok(Response::new(GetNodesResponse { nodes: proto_nodes }))
    }

    async fn get_node(
        &self,
        request: Request<GetNodeRequest>,
    ) -> Result<Response<NodeInfo>, Status> {
        let name = request.into_inner().name;
        // spurd uses GetNode as a lightweight heartbeat — update the
        // last_heartbeat timestamp so the health checker doesn't mark
        // the node DOWN.
        self.cluster.update_heartbeat(&name, 0, 0);
        let node = self
            .cluster
            .get_node(&name)
            .ok_or_else(|| Status::not_found(format!("node {} not found", name)))?;
        Ok(Response::new(node_to_proto(&node)))
    }

    async fn update_node(
        &self,
        request: Request<UpdateNodeRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        if let Some(state) = req.state {
            let node_state = proto_to_node_state(state)
                .ok_or_else(|| Status::invalid_argument("invalid node state"))?;
            self.cluster
                .update_node_state(&req.name, node_state, req.reason)
                .map_err(|e| Status::internal(e.to_string()))?;
        }
        Ok(Response::new(()))
    }

    async fn get_partitions(
        &self,
        _request: Request<GetPartitionsRequest>,
    ) -> Result<Response<GetPartitionsResponse>, Status> {
        let partitions = self.cluster.get_partitions();
        let proto: Vec<PartitionInfo> = partitions.iter().map(partition_to_proto).collect();
        Ok(Response::new(GetPartitionsResponse { partitions: proto }))
    }

    async fn ping(&self, _request: Request<()>) -> Result<Response<PingResponse>, Status> {
        let hostname: String = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".into());

        Ok(Response::new(PingResponse {
            hostname,
            server_time: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            version: env!("CARGO_PKG_VERSION").into(),
        }))
    }

    async fn register_agent(
        &self,
        request: Request<RegisterAgentRequest>,
    ) -> Result<Response<RegisterAgentResponse>, Status> {
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

        let req = request.into_inner();
        let resources = req.resources.map(proto_to_resource_set).unwrap_or_default();

        // Prefer agent's self-reported address (e.g. WireGuard IP),
        // fall back to remote TCP address
        let agent_addr = if !req.address.is_empty() {
            req.address.clone()
        } else {
            let is_loopback =
                remote_addr.is_empty() || remote_addr == "127.0.0.1" || remote_addr == "::1";
            if is_loopback {
                "127.0.0.1".to_string()
            } else {
                remote_addr
            }
        };

        let agent_port = if req.port > 0 { req.port as u16 } else { 6818 };

        self.cluster.register_node(
            req.hostname.clone(),
            resources,
            agent_addr,
            agent_port,
            req.wg_pubkey,
            req.version,
            spur_core::node::NodeSource::BareMetal,
        );

        Ok(Response::new(RegisterAgentResponse {
            accepted: true,
            message: "registered".into(),
        }))
    }

    type HeartbeatStream =
        tokio_stream::wrappers::ReceiverStream<Result<HeartbeatResponse, Status>>;

    async fn report_job_status(
        &self,
        request: Request<ReportJobStatusRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let state = proto_to_job_state(req.state)
            .ok_or_else(|| Status::invalid_argument("invalid job state"))?;

        if state.is_terminal() {
            let exit_code = req.exit_code;
            self.cluster
                .complete_job(req.job_id, exit_code, state)
                .map_err(|e| Status::internal(e.to_string()))?;
        }
        Ok(Response::new(()))
    }

    async fn heartbeat(
        &self,
        _request: Request<tonic::Streaming<HeartbeatRequest>>,
    ) -> Result<Response<Self::HeartbeatStream>, Status> {
        // MVP: simple request/response, streaming deferred
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let _ = tx; // Will implement streaming in Phase 5
        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
            rx,
        )))
    }
}

pub async fn serve(addr: SocketAddr, cluster: Arc<ClusterManager>) -> anyhow::Result<()> {
    let service = ControllerService { cluster };

    tonic::transport::Server::builder()
        .add_service(SlurmControllerServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

// ---- Proto conversion helpers ----

fn proto_to_job_spec(spec: JobSpec) -> Result<spur_core::job::JobSpec, Status> {
    Ok(spur_core::job::JobSpec {
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
        num_tasks: spec.num_tasks.max(1),
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
        gres: spec.gres,
        script: if spec.script.is_empty() {
            None
        } else {
            Some(spec.script)
        },
        argv: spec.argv,
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
        array_spec: if spec.array_spec.is_empty() {
            None
        } else {
            Some(spec.array_spec)
        },
        requeue: spec.requeue,
        exclusive: spec.exclusive,
        hold: spec.hold,
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
    })
}

fn proto_to_job_state(s: i32) -> Option<spur_core::job::JobState> {
    match s {
        0 => Some(spur_core::job::JobState::Pending),
        1 => Some(spur_core::job::JobState::Running),
        2 => Some(spur_core::job::JobState::Completing),
        3 => Some(spur_core::job::JobState::Completed),
        4 => Some(spur_core::job::JobState::Failed),
        5 => Some(spur_core::job::JobState::Cancelled),
        6 => Some(spur_core::job::JobState::Timeout),
        7 => Some(spur_core::job::JobState::NodeFail),
        8 => Some(spur_core::job::JobState::Preempted),
        9 => Some(spur_core::job::JobState::Suspended),
        _ => None,
    }
}

fn proto_to_node_state(s: i32) -> Option<spur_core::node::NodeState> {
    match s {
        0 => Some(spur_core::node::NodeState::Idle),
        1 => Some(spur_core::node::NodeState::Allocated),
        2 => Some(spur_core::node::NodeState::Mixed),
        3 => Some(spur_core::node::NodeState::Down),
        4 => Some(spur_core::node::NodeState::Drain),
        5 => Some(spur_core::node::NodeState::Draining),
        6 => Some(spur_core::node::NodeState::Error),
        7 => Some(spur_core::node::NodeState::Unknown),
        _ => None,
    }
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

fn job_to_proto(job: &spur_core::job::Job) -> JobInfo {
    use spur_core::hostlist;

    JobInfo {
        job_id: job.job_id,
        name: job.spec.name.clone(),
        user: job.spec.user.clone(),
        uid: job.spec.uid,
        partition: job.spec.partition.clone().unwrap_or_default(),
        account: job.spec.account.clone().unwrap_or_default(),
        state: job_state_to_proto(job.state) as i32,
        state_reason: job.pending_reason.display().to_string(),
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
        stdout_path: job.resolved_stdout(),
        stderr_path: job.resolved_stderr(),
        resources: job.allocated_resources.as_ref().map(resource_to_proto),
        priority: job.priority,
        qos: job.spec.qos.clone().unwrap_or_default(),
        array_job_id: job.array_job_id.unwrap_or(0),
        array_task_id: job.array_task_id.unwrap_or(0),
    }
}

fn node_to_proto(node: &spur_core::node::Node) -> NodeInfo {
    NodeInfo {
        name: node.name.clone(),
        state: node_state_to_proto(node.state) as i32,
        state_reason: node.state_reason.clone().unwrap_or_default(),
        partition: node.partitions.first().cloned().unwrap_or_default(),
        total_resources: Some(resource_to_proto(&node.total_resources)),
        alloc_resources: Some(resource_to_proto(&node.alloc_resources)),
        arch: node.arch.clone(),
        os: node.os.clone(),
        cpu_load: node.cpu_load,
        free_memory_mb: node.free_memory_mb,
        boot_time: node.boot_time.map(datetime_to_proto),
        last_busy: node.last_busy.map(datetime_to_proto),
        slurmd_start_time: node.agent_start_time.map(datetime_to_proto),
    }
}

fn partition_to_proto(part: &spur_core::partition::Partition) -> PartitionInfo {
    PartitionInfo {
        name: part.name.clone(),
        state: part.state.display().to_string(),
        is_default: part.is_default,
        total_nodes: 0, // Computed at query time
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
        preempt_mode: format!("{:?}", part.preempt_mode),
        priority_tier: part.priority_tier,
    }
}

fn resource_to_proto(r: &spur_core::resource::ResourceSet) -> spur_proto::proto::ResourceSet {
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

fn job_state_to_proto(s: spur_core::job::JobState) -> spur_proto::proto::JobState {
    match s {
        spur_core::job::JobState::Pending => spur_proto::proto::JobState::JobPending,
        spur_core::job::JobState::Running => spur_proto::proto::JobState::JobRunning,
        spur_core::job::JobState::Completing => spur_proto::proto::JobState::JobCompleting,
        spur_core::job::JobState::Completed => spur_proto::proto::JobState::JobCompleted,
        spur_core::job::JobState::Failed => spur_proto::proto::JobState::JobFailed,
        spur_core::job::JobState::Cancelled => spur_proto::proto::JobState::JobCancelled,
        spur_core::job::JobState::Timeout => spur_proto::proto::JobState::JobTimeout,
        spur_core::job::JobState::NodeFail => spur_proto::proto::JobState::JobNodeFail,
        spur_core::job::JobState::Preempted => spur_proto::proto::JobState::JobPreempted,
        spur_core::job::JobState::Suspended => spur_proto::proto::JobState::JobSuspended,
    }
}

fn node_state_to_proto(s: spur_core::node::NodeState) -> spur_proto::proto::NodeState {
    match s {
        spur_core::node::NodeState::Idle => spur_proto::proto::NodeState::NodeIdle,
        spur_core::node::NodeState::Allocated => spur_proto::proto::NodeState::NodeAllocated,
        spur_core::node::NodeState::Mixed => spur_proto::proto::NodeState::NodeMixed,
        spur_core::node::NodeState::Down => spur_proto::proto::NodeState::NodeDown,
        spur_core::node::NodeState::Drain => spur_proto::proto::NodeState::NodeDrain,
        spur_core::node::NodeState::Draining => spur_proto::proto::NodeState::NodeDraining,
        spur_core::node::NodeState::Error => spur_proto::proto::NodeState::NodeError,
        spur_core::node::NodeState::Unknown => spur_proto::proto::NodeState::NodeUnknown,
    }
}

fn datetime_to_proto(dt: chrono::DateTime<chrono::Utc>) -> prost_types::Timestamp {
    prost_types::Timestamp {
        seconds: dt.timestamp(),
        nanos: dt.timestamp_subsec_nanos() as i32,
    }
}
