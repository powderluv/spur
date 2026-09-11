// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::time::Duration;

use backon::{ExponentialBuilder, Retryable};
use k8s_openapi::api::core::v1::{
    Container, EnvVar, HostPathVolumeSource, Pod, PodSpec, ResourceRequirements, Service,
    ServicePort, ServiceSpec, Volume, VolumeMount,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;
use kube::api::{Api, AttachParams, DeleteParams, ListParams, ObjectMeta, PostParams};
use kube::Client;
use tokio::io::AsyncReadExt;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

use crate::crd::SpurJob;
use spur_core::spur_env::SpurEnv;
use spur_proto::proto::slurm_agent_server::SlurmAgent;
use spur_proto::proto::*;

const NS_LOOKUP_BUDGET: Duration = Duration::from_secs(5);

/// The placement facts the agent reads off a job's SpurJob before creating pods: which namespace it
/// belongs to, and which nodes the controller recorded as its allocation.
struct ResolvedJob {
    namespace: String,
    /// `status.assignedNodes` — the controller's recorded allocation, projected onto the CRD. Empty
    /// until the operator's job controller has patched it (see [`validate_target_node`]).
    assigned_nodes: Vec<String>,
}

/// Refuses a `target_node` (which bypasses the scheduler) outside the job's allocation. Before the
/// allocation is projected — the normal state for any not-yet-scheduled job, not a brief race — the
/// pin is allowed but logged, unvalidated.
fn validate_target_node(
    target_node: &str,
    assigned_nodes: &[String],
    job_id: u32,
) -> Result<(), Status> {
    if target_node.is_empty() {
        return Ok(());
    }
    if assigned_nodes.is_empty() {
        warn!(
            job_id,
            target_node,
            "launch pins a node before the job's allocation is visible; placement not validated"
        );
        return Ok(());
    }
    if assigned_nodes.iter().any(|n| n == target_node) {
        Ok(())
    } else {
        Err(Status::permission_denied(format!(
            "target node '{target_node}' is not in job {job_id}'s allocation {assigned_nodes:?}: \
             refusing to place a pod outside the recorded allocation"
        )))
    }
}

/// Virtual SlurmAgent that creates K8s Pods instead of fork/exec.
pub struct VirtualAgent {
    client: Client,
}

impl VirtualAgent {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// Look up the SpurJob labeled `spur.amd.com/job-id=<id>` and return its placement facts.
    /// Fails loudly if not found so pods are never placed in the wrong namespace.
    async fn resolve_job(&self, job_id: u32) -> Result<ResolvedJob, Status> {
        let api: Api<SpurJob> = Api::all(self.client.clone());
        let lp = ListParams::default().labels(&format!("spur.amd.com/job-id={}", job_id));

        let result = tokio::time::timeout(
            NS_LOOKUP_BUDGET,
            (|| async {
                let list = tokio::time::timeout(Duration::from_millis(300), api.list(&lp))
                    .await
                    .map_err(|_| Status::unavailable("k8s API timeout"))?
                    .map_err(|e| Status::internal(e.to_string()))?;

                // Fail closed on ambiguity: a non-unique label could match >1 SpurJob. Zero is a
                // not-yet-visible race (retried); >1 is a real conflict, never guessed.
                let mut items = list.items.into_iter();
                match (items.next(), items.next()) {
                    (None, _) => Err(Status::not_found(format!(
                        "no SpurJob carries the label spur.amd.com/job-id={job_id}. In Pod mode \
                         the operator makes Pods only for a SpurJob custom resource, so a job \
                         submitted with the CLI (sbatch, spur submit) has nothing to launch. \
                         Submit it with `kubectl apply` of a SpurJob instead. If this job DID \
                         come from a SpurJob, the label is not visible yet and this is retried."
                    ))),
                    (Some(job), None) => Ok(job),
                    (Some(_), Some(_)) => Err(Status::failed_precondition(format!(
                        "multiple SpurJobs carry spur.amd.com/job-id={job_id}; refusing to guess \
                         which one to launch"
                    ))),
                }
            })
            .retry(
                ExponentialBuilder::default()
                    .with_min_delay(Duration::from_millis(200))
                    .with_max_delay(Duration::from_secs(2))
                    .without_max_times(),
            )
            .when(|e: &Status| {
                matches!(e.code(), tonic::Code::Unavailable | tonic::Code::NotFound)
            }),
        )
        .await;

        let job = match result {
            Ok(Ok(job)) => job,
            Ok(Err(status)) => return Err(status),
            Err(_elapsed) => {
                return Err(Status::deadline_exceeded(format!(
                    "namespace lookup for spur.amd.com/job-id={job_id} timed out after {}s",
                    NS_LOOKUP_BUDGET.as_secs()
                )))
            }
        };

        let namespace = job.metadata.namespace.ok_or_else(|| {
            Status::not_found(format!(
                "SpurJob for spur.amd.com/job-id={job_id} has no namespace"
            ))
        })?;
        let assigned_nodes = job.status.map(|s| s.assigned_nodes).unwrap_or_default();
        Ok(ResolvedJob {
            namespace,
            assigned_nodes,
        })
    }

    /// Look up the namespace of the SpurJob labeled `spur.amd.com/job-id=<id>`.
    async fn resolve_namespace(&self, job_id: u32) -> Result<String, Status> {
        Ok(self.resolve_job(job_id).await?.namespace)
    }
}

#[tonic::async_trait]
impl SlurmAgent for VirtualAgent {
    type StreamJobOutputStream =
        tokio_stream::wrappers::ReceiverStream<Result<StreamJobOutputChunk, Status>>;
    type InteractiveSessionStream =
        tokio_stream::wrappers::ReceiverStream<Result<InteractiveOutput, Status>>;

    async fn launch_job(
        &self,
        request: Request<LaunchJobRequest>,
    ) -> Result<Response<LaunchJobResponse>, Status> {
        let req = request.into_inner();
        let job_id = req.job_id;
        let job = self.resolve_job(job_id).await?;
        let ns = job.namespace;
        let target_node = req.target_node.clone();
        // `target_node` becomes the pod's spec.nodeName (scheduler-bypassing). Refuse to pin a node
        // the controller did not allocate to this job so a caller can't place work on another
        // tenant's node.
        validate_target_node(&target_node, &job.assigned_nodes, job_id)?;
        let peer_nodes = &req.peer_nodes;
        let num_peers = peer_nodes.len();

        let spec = req
            .spec
            .ok_or_else(|| Status::invalid_argument("missing job spec"))?;

        // Pod name includes target_node to avoid conflicts for multi-node jobs
        let pod_name = if target_node.is_empty() {
            format!("spur-job-{}", job_id)
        } else {
            // Sanitize node name for K8s naming (lowercase, alphanumeric + dashes)
            let sanitized = sanitize_k8s_name(&target_node);
            format!("spur-job-{}-{}", job_id, sanitized)
        };

        let image = if spec.container_image.is_empty() {
            "busybox:latest".to_string()
        } else {
            spec.container_image.clone()
        };

        // Build resource requests
        let mut resource_requests = BTreeMap::new();
        let mut resource_limits = BTreeMap::new();

        if let Some(ref alloc) = req.allocated {
            if alloc.cpus > 0 {
                let cpu_str = alloc.cpus.to_string();
                resource_requests.insert("cpu".to_string(), Quantity(cpu_str.clone()));
                resource_limits.insert("cpu".to_string(), Quantity(cpu_str));
            }
            if alloc.memory_mb > 0 {
                let mem_str = format!("{}Mi", alloc.memory_mb);
                resource_requests.insert("memory".to_string(), Quantity(mem_str.clone()));
                resource_limits.insert("memory".to_string(), Quantity(mem_str));
            }
            let gpu_count = alloc
                .devices
                .get("gpu")
                .map(|d| d.devices.len() as u32)
                .unwrap_or(0);
            if gpu_count > 0 {
                let gpu_str = gpu_count.to_string();
                let gpu_type = spec
                    .gres
                    .iter()
                    .find_map(|g| spur_core::resource::parse_gres(g))
                    .and_then(|(_, t, _)| t);
                let gpu_resource_key = gpu_vendor_resource_key(gpu_type.as_deref());
                resource_limits.insert(gpu_resource_key.to_string(), Quantity(gpu_str.clone()));
                resource_requests.insert(gpu_resource_key.to_string(), Quantity(gpu_str));
            }
        }

        // Compute node rank from task_offset.
        // Issue #69: peer_nodes contains addr:port strings (e.g., "10.0.0.1:6818")
        // while target_node is a hostname — starts_with matching never worked,
        // causing all pods to get rank 0. Instead, derive rank from task_offset
        // which is incremented per-node by the dispatcher.
        let tasks_per_node = spec.tasks_per_node.max(1);
        let node_rank = req.task_offset / tasks_per_node;

        // Build env vars via SpurEnv accumulator
        let mut senv = SpurEnv::new();
        senv.set_with_slurm_twin("SPUR_JOB_ID", job_id);
        senv.set_with_slurm_twin("SPUR_JOBID", job_id);
        senv.set_with_slurm_twin("SPUR_JOB_NAME", &spec.name);
        senv.set_with_slurm_twin("SPUR_JOB_PARTITION", &spec.partition);
        senv.set_with_slurm_twin("SPUR_JOB_ACCOUNT", &spec.account);
        senv.set_with_slurm_twin("SPUR_JOB_QOS", &spec.qos);
        senv.set_with_slurm_twin("SPUR_NNODES", num_peers);
        senv.set_with_slurm_twin("SPUR_JOB_NUM_NODES", num_peers);
        senv.set_with_slurm_twin("SPUR_NTASKS", spec.num_tasks);
        senv.set_with_slurm_twin("SPUR_NPROCS", spec.num_tasks);
        senv.set_with_slurm_twin("SPUR_CPUS_PER_TASK", spec.cpus_per_task);
        senv.set_with_slurm_twin(
            "SPUR_CPUS_ON_NODE",
            tasks_per_node * spec.cpus_per_task.max(1),
        );
        senv.set_with_slurm_twin("SPUR_TASKS_PER_NODE", tasks_per_node);
        senv.set_with_slurm_twin("SPUR_NODEID", node_rank);
        senv.set_with_slurm_twin("SPUR_NODELIST", &spec.nodelist);
        senv.set_with_slurm_twin("SPUR_JOB_NODELIST", &spec.nodelist);
        senv.set_with_slurm_twin("SPURD_NODENAME", &target_node);

        senv.set("SPUR_TASK_OFFSET", req.task_offset);
        senv.set("SPUR_NODE_RANK", node_rank);
        // The controller sends agent addresses in `peer_nodes`. In Pod mode every
        // node answers on the one operator address, so that list is the same
        // address repeated and no workload can reach a peer with it. The Pods of a
        // multi-node job are reachable under the headless Service instead, because
        // each Pod takes its target node as its hostname.
        let peer_dns = headless_peer_dns(&spec.nodelist, job_id, &ns);
        if !peer_dns.is_empty() {
            senv.set("SPUR_PEER_NODES", peer_dns.join(","));
        }
        if !target_node.is_empty() {
            senv.set("SPUR_TARGET_NODE", &target_node);
        }

        let mut env_vars: Vec<EnvVar> = senv
            .into_map()
            .into_iter()
            .map(|(name, value)| EnvVar {
                name,
                value: Some(value),
                ..Default::default()
            })
            .collect();

        // Set GPU vendor-specific env vars for the runtime
        let gpu_count = req
            .allocated
            .as_ref()
            .and_then(|a| a.devices.get("gpu"))
            .map(|d| d.devices.len())
            .unwrap_or(0);
        if gpu_count > 0 {
            let gpu_type = spec
                .gres
                .iter()
                .find_map(|g| spur_core::resource::parse_gres(g))
                .and_then(|(_, t, _)| t);
            if gpu_type.as_deref().is_none_or(|t| !is_nvidia_gpu(t)) {
                env_vars.push(EnvVar {
                    name: "GPU_ENABLE_PAL".into(),
                    value: Some("0".into()),
                    ..Default::default()
                });
                if num_peers > 1 {
                    env_vars.push(EnvVar {
                        name: "NCCL_SOCKET_IFNAME".into(),
                        value: Some("eth0".into()),
                        ..Default::default()
                    });
                }
            } else if num_peers > 1 {
                env_vars.push(EnvVar {
                    name: "NCCL_SOCKET_IFNAME".into(),
                    value: Some("eth0".into()),
                    ..Default::default()
                });
            }
        }

        for (k, v) in &spec.environment {
            env_vars.push(EnvVar {
                name: k.clone(),
                value: Some(v.clone()),
                ..Default::default()
            });
        }

        // Issue #117: Inject secret env vars from SpurJob CRD's secretEnv field.
        // These reference K8s Secrets and are injected as secretKeyRef, keeping
        // secret values out of the SpurJob spec and Raft log.
        {
            let api: kube::Api<crate::crd::SpurJob> = kube::Api::all(self.client.clone());
            let lp =
                kube::api::ListParams::default().labels(&format!("spur.amd.com/job-id={}", job_id));
            if let Ok(list) = api.list(&lp).await {
                if let Some(spurjob) = list.items.into_iter().next() {
                    for (env_name, secret_ref) in &spurjob.spec.secret_env {
                        if let Some((secret_name, secret_key)) = secret_ref.split_once('/') {
                            env_vars.push(EnvVar {
                                name: env_name.clone(),
                                value_from: Some(k8s_openapi::api::core::v1::EnvVarSource {
                                    secret_key_ref: Some(
                                        k8s_openapi::api::core::v1::SecretKeySelector {
                                            name: secret_name.to_string(),
                                            key: secret_key.to_string(),
                                            optional: Some(true),
                                        },
                                    ),
                                    ..Default::default()
                                }),
                                ..Default::default()
                            });
                        }
                    }
                }
            }
        }

        // Build command
        let command = if !spec.argv.is_empty() {
            Some(spec.argv.clone())
        } else if !spec.script.is_empty() {
            Some(vec!["sh".into(), "-c".into(), spec.script.clone()])
        } else {
            // Interactive session: keep pod alive so kube exec can attach a terminal
            Some(vec!["sleep".into(), "infinity".into()])
        };

        // Parse container_mounts → volumes + volume_mounts
        let (mut volumes, mut volume_mounts) = parse_mounts(&spec.container_mounts);

        // Set working_dir from work_dir or container_workdir
        let working_dir = if !spec.container_workdir.is_empty() {
            Some(spec.container_workdir.clone())
        } else if !spec.work_dir.is_empty() {
            Some(spec.work_dir.clone())
        } else {
            None
        };

        // Add extra device plugin resources (RDMA, MIG, etc.) — Issue #88
        for (key, val) in &spec.extra_resources {
            resource_requests.insert(key.clone(), Quantity(val.clone()));
            resource_limits.insert(key.clone(), Quantity(val.clone()));
        }

        // Shared memory volume mount — Issue #87
        if !spec.shm_size.is_empty() {
            volume_mounts.push(k8s_openapi::api::core::v1::VolumeMount {
                name: "dshm".into(),
                mount_path: "/dev/shm".into(),
                ..Default::default()
            });
        }

        // Privileged mode / SecurityContext — Issue #86
        let security_context = if spec.privileged {
            Some(k8s_openapi::api::core::v1::SecurityContext {
                privileged: Some(true),
                ..Default::default()
            })
        } else {
            None
        };

        let container = Container {
            name: "spur-job".into(),
            image: Some(image),
            command,
            env: Some(env_vars),
            working_dir,
            volume_mounts: if volume_mounts.is_empty() {
                None
            } else {
                Some(volume_mounts)
            },
            resources: Some(ResourceRequirements {
                requests: Some(resource_requests),
                limits: Some(resource_limits),
                ..Default::default()
            }),
            security_context,
            ..Default::default()
        };

        // Build labels
        let mut labels = BTreeMap::new();
        labels.insert("spur.amd.com/job-id".to_string(), job_id.to_string());
        labels.insert(
            "spur.amd.com/managed-by".to_string(),
            "spur-k8s-operator".to_string(),
        );
        if !spec.name.is_empty() {
            labels.insert("spur.amd.com/job-name".to_string(), spec.name.clone());
        }
        if !target_node.is_empty() {
            labels.insert("spur.amd.com/target-node".to_string(), target_node.clone());
        }

        // For multi-node jobs, create headless Service for DNS discovery
        if num_peers > 1 {
            if let Err(e) = self.ensure_headless_service(job_id, &labels, &ns).await {
                warn!(job_id, error = %e, "failed to create headless service");
            }
        }

        // Pin to target K8s node
        let node_name = if !target_node.is_empty() {
            Some(target_node.clone())
        } else {
            peer_nodes.first().cloned()
        };

        // For headless service DNS: set hostname and subdomain
        let (hostname, subdomain) = if num_peers > 1 && !target_node.is_empty() {
            (
                Some(sanitize_k8s_name(&target_node)),
                Some(format!("spur-job-{}", job_id)),
            )
        } else {
            (None, None)
        };

        let pod = Pod {
            metadata: ObjectMeta {
                name: Some(pod_name.clone()),
                namespace: Some(ns.clone()),
                labels: Some(labels),
                ..Default::default()
            },
            spec: Some({
                // Shared memory emptyDir volume — Issue #87
                if !spec.shm_size.is_empty() {
                    volumes.push(k8s_openapi::api::core::v1::Volume {
                        name: "dshm".into(),
                        empty_dir: Some(k8s_openapi::api::core::v1::EmptyDirVolumeSource {
                            medium: Some("Memory".into()),
                            size_limit: Some(Quantity(spec.shm_size.clone())),
                        }),
                        ..Default::default()
                    });
                }

                PodSpec {
                    containers: vec![container],
                    restart_policy: Some("Never".into()),
                    node_name,
                    hostname,
                    subdomain,
                    volumes: if volumes.is_empty() {
                        None
                    } else {
                        Some(volumes)
                    },
                    // Issue #85: host_network
                    host_network: if spec.host_network { Some(true) } else { None },
                    // Issue #87: host_ipc
                    host_ipc: if spec.host_ipc { Some(true) } else { None },
                    ..Default::default()
                }
            }),
            ..Default::default()
        };

        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &ns);
        match pods.create(&PostParams::default(), &pod).await {
            Ok(_) => {
                info!(job_id, pod = %pod_name, namespace = %ns, target = %req.target_node, "K8s Pod created");
                Ok(Response::new(LaunchJobResponse {
                    success: true,
                    error: String::new(),
                    ..Default::default()
                }))
            }
            Err(kube::Error::Api(e)) if e.code == 409 => {
                info!(job_id, pod = %pod_name, namespace = %ns, target = %req.target_node, "K8s Pod already exists, treating as success");
                Ok(Response::new(LaunchJobResponse {
                    success: true,
                    error: String::new(),
                    ..Default::default()
                }))
            }
            Err(e) => {
                error!(job_id, error = %e, "failed to create K8s Pod");
                Ok(Response::new(LaunchJobResponse {
                    success: false,
                    error: e.to_string(),
                    ..Default::default()
                }))
            }
        }
    }

    async fn prepare_pmix(
        &self,
        _request: Request<PreparePmixRequest>,
    ) -> Result<Response<PreparePmixResponse>, Status> {
        Err(Status::unimplemented(
            "PMIx prepare is not supported on the K8s virtual agent",
        ))
    }

    async fn release_pmix(
        &self,
        _request: Request<ReleasePmixRequest>,
    ) -> Result<Response<ReleasePmixResponse>, Status> {
        Ok(Response::new(ReleasePmixResponse {}))
    }

    async fn cancel_job(
        &self,
        request: Request<AgentCancelJobRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let job_id = req.job_id;
        let ns = self.resolve_namespace(job_id).await?;

        // Delete all pods for this job by label selector
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &ns);
        let lp = ListParams::default().labels(&format!("spur.amd.com/job-id={}", job_id));

        match pods.list(&lp).await {
            Ok(pod_list) => {
                for pod in pod_list {
                    let name = pod.metadata.name.unwrap_or_default();
                    match pods.delete(&name, &DeleteParams::default()).await {
                        Ok(_) => info!(job_id, pod = %name, "deleted Pod"),
                        Err(kube::Error::Api(e)) if e.code == 404 => {
                            debug!(job_id, pod = %name, "Pod already gone");
                        }
                        Err(e) => {
                            error!(job_id, pod = %name, error = %e, "failed to delete Pod");
                        }
                    }
                }
            }
            Err(e) => {
                error!(job_id, error = %e, "failed to list Pods for cancellation");
            }
        }

        // Also clean up the headless service if it exists
        let services: Api<Service> = Api::namespaced(self.client.clone(), &ns);
        let svc_name = format!("spur-job-{}", job_id);
        match services.delete(&svc_name, &DeleteParams::default()).await {
            Ok(_) => debug!(job_id, "deleted headless Service"),
            Err(kube::Error::Api(e)) if e.code == 404 => {}
            Err(e) => {
                debug!(job_id, error = %e, "failed to delete headless Service");
            }
        }

        Ok(Response::new(()))
    }

    async fn suspend_job(
        &self,
        request: Request<AgentSuspendJobRequest>,
    ) -> Result<Response<()>, Status> {
        // Pod-level SIGSTOP/SIGCONT is not modeled for the k8s backend; the
        // controller-side state change still applies. Accept as a no-op.
        let req = request.into_inner();
        debug!(
            job_id = req.job_id,
            resume = req.resume,
            "k8s backend: suspend/resume is a no-op"
        );
        Ok(Response::new(()))
    }

    async fn get_node_resources(
        &self,
        _request: Request<()>,
    ) -> Result<Response<NodeResourcesResponse>, Status> {
        Ok(Response::new(NodeResourcesResponse {
            total: Some(ResourceSet::default()),
            used: Some(spur_proto::proto::ResourceAllocations::default()),
        }))
    }

    async fn exec_in_job(
        &self,
        request: Request<ExecInJobRequest>,
    ) -> Result<Response<ExecInJobResponse>, Status> {
        let req = request.into_inner();
        let job_id = req.job_id;
        let ns = self.resolve_namespace(job_id).await?;
        let pod_name = format!("spur-job-{}", job_id);
        let command: Vec<String> = if req.command.is_empty() {
            vec!["bash".into(), "-c".into(), "echo ok".into()]
        } else {
            req.command
        };

        debug!(pod = %pod_name, cmd = ?command, "exec in K8s pod");

        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &ns);

        let attach = AttachParams {
            stdin: false,
            stdout: true,
            stderr: true,
            tty: false,
            container: None,
            max_stdin_buf_size: None,
            max_stdout_buf_size: Some(1024 * 1024),
            max_stderr_buf_size: Some(1024 * 1024),
        };

        let mut exec = pods
            .exec(&pod_name, command, &attach)
            .await
            .map_err(|e| Status::internal(format!("exec failed: {e}")))?;

        let mut stdout_data = Vec::new();
        let mut stderr_data = Vec::new();

        if let Some(mut stdout) = exec.stdout() {
            let _ = stdout.read_to_end(&mut stdout_data).await;
        }
        if let Some(mut stderr) = exec.stderr() {
            let _ = stderr.read_to_end(&mut stderr_data).await;
        }

        let status = exec
            .take_status()
            .ok_or_else(|| Status::internal("no exit status"))?
            .await
            .ok_or_else(|| Status::internal("status channel closed"))?;

        let exit_code = status
            .status
            .as_deref()
            .map(|s| if s == "Success" { 0 } else { 1 })
            .unwrap_or(1);

        Ok(Response::new(ExecInJobResponse {
            success: exit_code == 0,
            exit_code,
            stdout: String::from_utf8_lossy(&stdout_data).into_owned(),
            stderr: String::from_utf8_lossy(&stderr_data).into_owned(),
        }))
    }

    async fn run_command(
        &self,
        _request: Request<RunCommandRequest>,
    ) -> Result<Response<RunCommandResponse>, Status> {
        // Srun step dispatch. The K8s virtual agent does not currently support
        // one-shot commands outside the job pod's lifecycle — salloc plus
        // srun-in-allocation is not a common K8s workflow.
        Err(Status::unimplemented(
            "RunCommand is not yet supported by the K8s virtual agent",
        ))
    }

    async fn cancel_step(
        &self,
        _request: Request<CancelStepRequest>,
    ) -> Result<Response<()>, Status> {
        Err(Status::unimplemented(
            "CancelStep is not yet supported by the K8s virtual agent",
        ))
    }

    async fn register_job_allocation(
        &self,
        _request: Request<RegisterJobAllocationRequest>,
    ) -> Result<Response<RegisterJobAllocationResponse>, Status> {
        Err(Status::unimplemented(
            "RegisterJobAllocation is not yet supported by the K8s virtual agent",
        ))
    }

    async fn stream_job_output(
        &self,
        request: Request<StreamJobOutputRequest>,
    ) -> Result<Response<Self::StreamJobOutputStream>, Status> {
        let req = request.into_inner();
        let job_id = req.job_id;
        let ns = self.resolve_namespace(job_id).await?;
        let pod_name = format!("spur-job-{}", job_id);

        debug!(pod = %pod_name, "streaming logs from K8s pod");

        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &ns);
        let log_params = kube::api::LogParams {
            follow: true,
            tail_lines: Some(100),
            ..Default::default()
        };

        let log_stream = pods
            .log_stream(&pod_name, &log_params)
            .await
            .map_err(|e| Status::internal(format!("log stream failed: {e}")))?;

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::spawn(async move {
            use futures_util::AsyncReadExt;
            let mut reader = log_stream;
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => {
                        if tx
                            .send(Ok(StreamJobOutputChunk {
                                data: buf[..n].to_vec(),
                                eof: false,
                            }))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            let _ = tx
                .send(Ok(StreamJobOutputChunk {
                    data: Vec::new(),
                    eof: true,
                }))
                .await;
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
            rx,
        )))
    }

    async fn interactive_session(
        &self,
        _request: Request<tonic::Streaming<InteractiveInput>>,
    ) -> Result<Response<Self::InteractiveSessionStream>, Status> {
        Err(Status::unimplemented(
            "interactive session not supported for K8s agent",
        ))
    }

    // -- Native cluster component control. The virtual K8s agent does not run k0s
    //    systemd units, so these are permanently unsupported here. --
    async fn start_cluster_component(
        &self,
        _request: Request<StartClusterComponentRequest>,
    ) -> Result<Response<StartClusterComponentResponse>, Status> {
        Err(Status::unimplemented(
            "cluster components not supported for K8s agent",
        ))
    }

    async fn stop_cluster_component(
        &self,
        _request: Request<StopClusterComponentRequest>,
    ) -> Result<Response<StopClusterComponentResponse>, Status> {
        Err(Status::unimplemented(
            "cluster components not supported for K8s agent",
        ))
    }

    async fn get_cluster_component_status(
        &self,
        _request: Request<GetClusterComponentStatusRequest>,
    ) -> Result<Response<GetClusterComponentStatusResponse>, Status> {
        Err(Status::unimplemented(
            "cluster components not supported for K8s agent",
        ))
    }

    async fn create_k0s_join_token(
        &self,
        _request: Request<CreateK0sJoinTokenRequest>,
    ) -> Result<Response<CreateK0sJoinTokenResponse>, Status> {
        Err(Status::unimplemented(
            "cluster components not supported for K8s agent",
        ))
    }

    async fn drain_k8s_node(
        &self,
        _request: Request<DrainK8sNodeRequest>,
    ) -> Result<Response<DrainK8sNodeResponse>, Status> {
        Err(Status::unimplemented(
            "cluster components not supported for K8s agent",
        ))
    }

    async fn delete_k8s_node(
        &self,
        _request: Request<DeleteK8sNodeRequest>,
    ) -> Result<Response<DeleteK8sNodeResponse>, Status> {
        Err(Status::unimplemented(
            "cluster components not supported for K8s agent",
        ))
    }

    async fn get_kubeconfig(
        &self,
        _request: Request<GetKubeconfigRequest>,
    ) -> Result<Response<GetKubeconfigResponse>, Status> {
        Err(Status::unimplemented(
            "cluster components not supported for K8s agent",
        ))
    }

    async fn apply_mesh(
        &self,
        _request: Request<MeshMembership>,
    ) -> Result<Response<ApplyMeshResponse>, Status> {
        Err(Status::unimplemented("mesh not supported for K8s agent"))
    }
}

impl VirtualAgent {
    /// Create a headless Service for inter-pod DNS discovery in multi-node jobs.
    async fn ensure_headless_service(
        &self,
        job_id: u32,
        labels: &BTreeMap<String, String>,
        namespace: &str,
    ) -> Result<(), kube::Error> {
        let services: Api<Service> = Api::namespaced(self.client.clone(), namespace);
        let svc_name = format!("spur-job-{}", job_id);

        let selector = BTreeMap::from([("spur.amd.com/job-id".to_string(), job_id.to_string())]);

        let svc = Service {
            metadata: ObjectMeta {
                name: Some(svc_name.clone()),
                namespace: Some(namespace.to_string()),
                labels: Some(labels.clone()),
                ..Default::default()
            },
            spec: Some(ServiceSpec {
                cluster_ip: Some("None".into()), // headless
                selector: Some(selector),
                ports: Some(vec![ServicePort {
                    name: Some("nccl".into()),
                    port: 29500,
                    target_port: Some(IntOrString::Int(29500)),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };

        match services.create(&PostParams::default(), &svc).await {
            Ok(_) => {
                info!(job_id, svc = %svc_name, "headless Service created");
                Ok(())
            }
            Err(kube::Error::Api(e)) if e.code == 409 => {
                debug!(job_id, "headless Service already exists");
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

/// Parse container_mounts ("/src:/dst:ro" or "pvc:name:/dst") into K8s volumes + mounts.
fn parse_mounts(mounts: &[String]) -> (Vec<Volume>, Vec<VolumeMount>) {
    let mut volumes = Vec::new();
    let mut volume_mounts = Vec::new();

    for (i, mount_str) in mounts.iter().enumerate() {
        let parts: Vec<&str> = mount_str.split(':').collect();

        if parts.len() >= 2 && parts[0] == "pvc" {
            // PVC mount: "pvc:claim-name:/dst"
            if parts.len() >= 3 {
                let vol_name = format!("pvc-{}", i);
                volumes.push(Volume {
                    name: vol_name.clone(),
                    persistent_volume_claim: Some(
                        k8s_openapi::api::core::v1::PersistentVolumeClaimVolumeSource {
                            claim_name: parts[1].to_string(),
                            read_only: Some(parts.get(3).is_some_and(|&v| v == "ro")),
                        },
                    ),
                    ..Default::default()
                });
                volume_mounts.push(VolumeMount {
                    name: vol_name,
                    mount_path: parts[2].to_string(),
                    read_only: Some(parts.get(3).is_some_and(|&v| v == "ro")),
                    ..Default::default()
                });
            }
        } else if parts.len() >= 2 {
            // hostPath mount: "/src:/dst[:ro]"
            let vol_name = format!("hostpath-{}", i);
            let read_only = parts.get(2).is_some_and(|&v| v == "ro");
            volumes.push(Volume {
                name: vol_name.clone(),
                host_path: Some(HostPathVolumeSource {
                    path: parts[0].to_string(),
                    type_: Some("DirectoryOrCreate".into()),
                }),
                ..Default::default()
            });
            volume_mounts.push(VolumeMount {
                name: vol_name,
                mount_path: parts[1].to_string(),
                read_only: Some(read_only),
                ..Default::default()
            });
        }
    }

    (volumes, volume_mounts)
}

/// Sanitize a string for use in K8s resource names.
fn sanitize_k8s_name(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

/// The DNS names under which the Pods of a multi-node job reach each other.
///
/// `launch_job` gives each Pod `hostname = sanitize_k8s_name(target_node)` and
/// `subdomain = spur-job-<id>`, and `ensure_headless_service` publishes that
/// Service, so every peer answers at
/// `<hostname>.spur-job-<id>.<namespace>.svc.cluster.local`. The order follows
/// the nodelist, so index N is the peer whose SPUR_NODE_RANK is N.
///
/// A single node job gets no headless Service and therefore no name to return.
fn headless_peer_dns(nodelist: &str, job_id: u32, namespace: &str) -> Vec<String> {
    let nodes: Vec<&str> = nodelist
        .split(',')
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .collect();
    if nodes.len() < 2 {
        return Vec::new();
    }
    nodes
        .iter()
        .map(|n| {
            format!(
                "{}.spur-job-{}.{}.svc.cluster.local",
                sanitize_k8s_name(n),
                job_id,
                namespace
            )
        })
        .collect()
}

/// Determine the K8s device plugin resource key based on GPU type.
///
/// AMD GPUs (mi300x, mi250x, gfx*, etc.) → "amd.com/gpu"
/// NVIDIA GPUs (h100, a100, etc.) → "nvidia.com/gpu"
/// Unknown/generic → "amd.com/gpu" (AMD-first default for ROCm project)
fn gpu_vendor_resource_key(gpu_type: Option<&str>) -> &'static str {
    match gpu_type {
        Some(t) if is_nvidia_gpu(t) => "nvidia.com/gpu",
        _ => "amd.com/gpu",
    }
}

/// Check if a GPU type string refers to an NVIDIA GPU.
fn is_nvidia_gpu(gpu_type: &str) -> bool {
    let lower = gpu_type.to_lowercase();
    // NVIDIA product families
    lower.starts_with("h100")
        || lower.starts_with("h200")
        || lower.starts_with("a100")
        || lower.starts_with("a10g")
        || lower.starts_with("a30")
        || lower.starts_with("v100")
        || lower.starts_with("t4")
        || lower.starts_with("l4")
        || lower.starts_with("l40")
        || lower.starts_with("b100")
        || lower.starts_with("b200")
        || lower.starts_with("gb200")
        || lower.starts_with("rtx")
        || lower == "nvidia"
}

/// Build a gres string from GPU count and type.
pub fn gpu_request_to_gres(count: u32, gpu_type: Option<&str>) -> String {
    let t = gpu_type.unwrap_or("any");
    let t = if t.is_empty() { "any" } else { t };
    format!("gpu:{}:{}", t, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- validate_target_node ---

    #[test]
    fn target_node_in_allocation_is_allowed() {
        let alloc = vec!["gpu-a-01".to_string(), "gpu-a-02".to_string()];
        assert!(validate_target_node("gpu-a-02", &alloc, 7).is_ok());
    }

    #[test]
    fn target_node_outside_allocation_is_rejected() {
        // The core hardening: a caller pinning another tenant's node is refused.
        let alloc = vec!["gpu-a-01".to_string()];
        let err = validate_target_node("gpu-b-07", &alloc, 7).unwrap_err();
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
    }

    #[test]
    fn empty_target_node_is_allowed() {
        // No pin: the scheduler (or a peer node) places the pod; nothing to validate.
        assert!(validate_target_node("", &["gpu-a-01".to_string()], 7).is_ok());
    }

    #[test]
    fn target_node_allowed_when_allocation_not_yet_visible() {
        // Races the status projection: allowed but (in the real path) logged.
        assert!(validate_target_node("gpu-a-01", &[], 7).is_ok());
    }

    // --- gpu_request_to_gres ---

    #[test]
    fn test_gpu_request_to_gres() {
        assert_eq!(gpu_request_to_gres(8, Some("mi300x")), "gpu:mi300x:8");
        assert_eq!(gpu_request_to_gres(4, None), "gpu:any:4");
        assert_eq!(gpu_request_to_gres(2, Some("")), "gpu:any:2");
    }

    #[test]
    fn test_gpu_request_to_gres_single() {
        assert_eq!(gpu_request_to_gres(1, Some("h100")), "gpu:h100:1");
    }

    // --- sanitize_k8s_name ---

    #[test]
    fn test_sanitize_k8s_name() {
        assert_eq!(sanitize_k8s_name("gpu-node-01"), "gpu-node-01");
        assert_eq!(sanitize_k8s_name("NODE_WITH.DOTS"), "node-with-dots");
        assert_eq!(sanitize_k8s_name("--leading--"), "leading");
    }

    #[test]
    fn test_sanitize_k8s_name_uppercase() {
        assert_eq!(sanitize_k8s_name("GPU-NODE-01"), "gpu-node-01");
    }

    #[test]
    fn test_sanitize_k8s_name_spaces_and_special() {
        assert_eq!(sanitize_k8s_name("my node@#$123"), "my-node---123");
    }

    #[test]
    fn test_sanitize_k8s_name_all_special() {
        assert_eq!(sanitize_k8s_name("@@@"), "");
    }

    #[test]
    fn test_sanitize_k8s_name_already_clean() {
        assert_eq!(sanitize_k8s_name("worker-3"), "worker-3");
    }

    // --- parse_mounts: hostPath ---

    #[test]
    fn test_parse_mounts_hostpath() {
        let mounts = vec!["/data:/mnt/data:ro".to_string(), "/tmp:/tmp".to_string()];
        let (vols, vmounts) = parse_mounts(&mounts);
        assert_eq!(vols.len(), 2);
        assert_eq!(vmounts.len(), 2);
        assert_eq!(vmounts[0].mount_path, "/mnt/data");
        assert_eq!(vmounts[0].read_only, Some(true));
        assert_eq!(vmounts[1].mount_path, "/tmp");
        assert_eq!(vmounts[1].read_only, Some(false));
    }

    #[test]
    fn test_parse_mounts_hostpath_source_path() {
        let mounts = vec!["/host/data:/container/data".to_string()];
        let (vols, _) = parse_mounts(&mounts);
        assert_eq!(vols.len(), 1);
        let hp = vols[0].host_path.as_ref().unwrap();
        assert_eq!(hp.path, "/host/data");
        assert_eq!(hp.type_.as_deref(), Some("DirectoryOrCreate"));
    }

    #[test]
    fn test_parse_mounts_hostpath_volume_naming() {
        let mounts = vec![
            "/a:/b".to_string(),
            "/c:/d".to_string(),
            "/e:/f".to_string(),
        ];
        let (vols, vmounts) = parse_mounts(&mounts);
        assert_eq!(vols[0].name, "hostpath-0");
        assert_eq!(vols[1].name, "hostpath-1");
        assert_eq!(vols[2].name, "hostpath-2");
        // Volume mount names must match volume names
        assert_eq!(vmounts[0].name, "hostpath-0");
        assert_eq!(vmounts[1].name, "hostpath-1");
        assert_eq!(vmounts[2].name, "hostpath-2");
    }

    // --- parse_mounts: PVC ---

    #[test]
    fn test_parse_mounts_pvc() {
        let mounts = vec!["pvc:my-claim:/data".to_string()];
        let (vols, vmounts) = parse_mounts(&mounts);
        assert_eq!(vols.len(), 1);
        assert_eq!(vmounts.len(), 1);
        assert_eq!(vmounts[0].mount_path, "/data");
        assert!(vols[0].persistent_volume_claim.is_some());
    }

    #[test]
    fn test_parse_mounts_pvc_claim_name() {
        let mounts = vec!["pvc:training-data:/mnt/data".to_string()];
        let (vols, _) = parse_mounts(&mounts);
        let pvc = vols[0].persistent_volume_claim.as_ref().unwrap();
        assert_eq!(pvc.claim_name, "training-data");
    }

    #[test]
    fn test_parse_mounts_pvc_readonly() {
        let mounts = vec!["pvc:datasets:/data:ro".to_string()];
        let (vols, vmounts) = parse_mounts(&mounts);
        let pvc = vols[0].persistent_volume_claim.as_ref().unwrap();
        assert_eq!(pvc.read_only, Some(true));
        assert_eq!(vmounts[0].read_only, Some(true));
    }

    #[test]
    fn test_parse_mounts_pvc_readwrite() {
        let mounts = vec!["pvc:output:/results".to_string()];
        let (vols, vmounts) = parse_mounts(&mounts);
        let pvc = vols[0].persistent_volume_claim.as_ref().unwrap();
        assert_eq!(pvc.read_only, Some(false));
        assert_eq!(vmounts[0].read_only, Some(false));
    }

    #[test]
    fn test_parse_mounts_pvc_naming() {
        let mounts = vec!["pvc:a:/x".to_string(), "pvc:b:/y".to_string()];
        let (vols, _) = parse_mounts(&mounts);
        assert_eq!(vols[0].name, "pvc-0");
        assert_eq!(vols[1].name, "pvc-1");
    }

    // --- parse_mounts: mixed and edge cases ---

    #[test]
    fn test_parse_mounts_mixed_hostpath_and_pvc() {
        let mounts = vec![
            "/data:/mnt/data:ro".to_string(),
            "pvc:checkpoints:/checkpoints".to_string(),
            "/logs:/var/log".to_string(),
        ];
        let (vols, vmounts) = parse_mounts(&mounts);
        assert_eq!(vols.len(), 3);
        assert_eq!(vmounts.len(), 3);
        // First is hostPath
        assert!(vols[0].host_path.is_some());
        assert!(vols[0].persistent_volume_claim.is_none());
        // Second is PVC
        assert!(vols[1].persistent_volume_claim.is_some());
        assert!(vols[1].host_path.is_none());
        // Third is hostPath
        assert!(vols[2].host_path.is_some());
    }

    #[test]
    fn test_parse_mounts_empty() {
        let mounts: Vec<String> = vec![];
        let (vols, vmounts) = parse_mounts(&mounts);
        assert!(vols.is_empty());
        assert!(vmounts.is_empty());
    }

    #[test]
    fn test_parse_mounts_single_component_ignored() {
        // A single component (no colon) should be skipped
        let mounts = vec!["just-a-path".to_string()];
        let (vols, vmounts) = parse_mounts(&mounts);
        assert!(vols.is_empty());
        assert!(vmounts.is_empty());
    }

    #[test]
    fn test_parse_mounts_pvc_missing_mount_path_ignored() {
        // "pvc:name" without mount path should be skipped (parts.len() < 3)
        let mounts = vec!["pvc:my-claim".to_string()];
        let (vols, vmounts) = parse_mounts(&mounts);
        assert!(vols.is_empty());
        assert!(vmounts.is_empty());
    }

    // --- GPU vendor detection ---

    #[test]
    fn test_is_nvidia_gpu_positive() {
        assert!(is_nvidia_gpu("h100"));
        assert!(is_nvidia_gpu("H100"));
        assert!(is_nvidia_gpu("h200"));
        assert!(is_nvidia_gpu("a100"));
        assert!(is_nvidia_gpu("A100"));
        assert!(is_nvidia_gpu("a10g"));
        assert!(is_nvidia_gpu("a30"));
        assert!(is_nvidia_gpu("v100"));
        assert!(is_nvidia_gpu("t4"));
        assert!(is_nvidia_gpu("T4"));
        assert!(is_nvidia_gpu("l4"));
        assert!(is_nvidia_gpu("l40s"));
        assert!(is_nvidia_gpu("L40"));
        assert!(is_nvidia_gpu("b100"));
        assert!(is_nvidia_gpu("b200"));
        assert!(is_nvidia_gpu("gb200"));
        assert!(is_nvidia_gpu("GB200"));
        assert!(is_nvidia_gpu("rtx4090"));
        assert!(is_nvidia_gpu("RTX3090"));
        assert!(is_nvidia_gpu("nvidia"));
        assert!(is_nvidia_gpu("NVIDIA"));
    }

    #[test]
    fn test_is_nvidia_gpu_negative_amd() {
        assert!(!is_nvidia_gpu("mi300x"));
        assert!(!is_nvidia_gpu("MI300X"));
        assert!(!is_nvidia_gpu("mi250x"));
        assert!(!is_nvidia_gpu("mi210"));
        assert!(!is_nvidia_gpu("mi100"));
        assert!(!is_nvidia_gpu("gfx942"));
        assert!(!is_nvidia_gpu("gfx1201"));
        assert!(!is_nvidia_gpu("gfx90a"));
        assert!(!is_nvidia_gpu("rx7900xtx"));
        assert!(!is_nvidia_gpu("w7900"));
        assert!(!is_nvidia_gpu("amd"));
        assert!(!is_nvidia_gpu("gpu"));
        assert!(!is_nvidia_gpu("any"));
        assert!(!is_nvidia_gpu(""));
    }

    #[test]
    fn test_gpu_vendor_resource_key_amd() {
        assert_eq!(gpu_vendor_resource_key(Some("mi300x")), "amd.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("mi250x")), "amd.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("gfx942")), "amd.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("gfx90a")), "amd.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("w7900")), "amd.com/gpu");
    }

    #[test]
    fn test_gpu_vendor_resource_key_nvidia() {
        assert_eq!(gpu_vendor_resource_key(Some("h100")), "nvidia.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("a100")), "nvidia.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("v100")), "nvidia.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("t4")), "nvidia.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("l40s")), "nvidia.com/gpu");
    }

    #[test]
    fn test_gpu_vendor_resource_key_defaults_amd() {
        // Unknown or generic GPU types default to AMD (ROCm project)
        assert_eq!(gpu_vendor_resource_key(None), "amd.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("gpu")), "amd.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("any")), "amd.com/gpu");
        assert_eq!(gpu_vendor_resource_key(Some("")), "amd.com/gpu");
    }

    #[test]
    fn test_gpu_request_to_gres_amd_types() {
        assert_eq!(gpu_request_to_gres(8, Some("mi300x")), "gpu:mi300x:8");
        assert_eq!(gpu_request_to_gres(4, Some("mi250x")), "gpu:mi250x:4");
        assert_eq!(gpu_request_to_gres(1, Some("gfx942")), "gpu:gfx942:1");
    }
}

#[cfg(test)]
mod peer_dns_tests {
    use super::headless_peer_dns;

    #[test]
    fn multi_node_job_gets_one_resolvable_name_per_node_in_nodelist_order() {
        let out = headless_peer_dns("node-a,node-b,node-c", 7, "spur");
        assert_eq!(
            out,
            vec![
                "node-a.spur-job-7.spur.svc.cluster.local".to_string(),
                "node-b.spur-job-7.spur.svc.cluster.local".to_string(),
                "node-c.spur-job-7.spur.svc.cluster.local".to_string(),
            ]
        );
    }

    #[test]
    fn single_node_job_gets_nothing_because_it_has_no_headless_service() {
        assert!(headless_peer_dns("node-a", 7, "spur").is_empty());
        assert!(headless_peer_dns("", 7, "spur").is_empty());
    }

    #[test]
    fn node_names_are_sanitized_the_same_way_as_the_pod_hostname() {
        let out = headless_peer_dns("Node_A.example,node-b", 3, "ns");
        assert_eq!(out[0], "node-a-example.spur-job-3.ns.svc.cluster.local");
    }
}
