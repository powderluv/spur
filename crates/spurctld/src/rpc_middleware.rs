// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tower middleware that records controller RPC handler durations on the leader.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use http::{Request, Response};
use tower::{Layer, Service};

use crate::raft::RaftHandle;
use crate::rpc_stats::RpcStatsCollector;

#[derive(Clone)]
pub struct RpcStatsLayer {
    stats: Arc<RpcStatsCollector>,
    raft: Arc<RaftHandle>,
}

impl RpcStatsLayer {
    pub fn new(stats: Arc<RpcStatsCollector>, raft: Arc<RaftHandle>) -> Self {
        Self { stats, raft }
    }
}

impl<S> Layer<S> for RpcStatsLayer {
    type Service = RpcStatsMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RpcStatsMiddleware {
            inner,
            stats: self.stats.clone(),
            raft: self.raft.clone(),
        }
    }
}

#[derive(Clone)]
pub struct RpcStatsMiddleware<S> {
    inner: S,
    stats: Arc<RpcStatsCollector>,
    raft: Arc<RaftHandle>,
}

impl<S, B> Service<Request<B>> for RpcStatsMiddleware<S>
where
    S: Service<Request<B>, Response = Response<tonic::body::Body>> + Clone + Send + 'static,
    S::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    S::Future: Send + 'static,
    B: Send + 'static,
{
    type Response = Response<tonic::body::Body>;
    type Error = Box<dyn std::error::Error + Send + Sync>;
    type Future =
        Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send + 'static>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, req: Request<B>) -> Self::Future {
        let operation = grpc_operation_name(req.uri().path());
        let record = self.raft.is_leader() && should_record_operation(&operation);
        let stats = self.stats.clone();
        let start = Instant::now();
        let mut inner = self.inner.clone();

        Box::pin(async move {
            let result = inner.call(req).await;
            if record {
                stats.record(&operation, start.elapsed());
            }
            result.map_err(Into::into)
        })
    }
}

/// Diagnostic RPCs excluded so reads and `sdiag` polling do not skew accumulators.
fn should_record_operation(operation: &str) -> bool {
    !matches!(
        operation,
        "GetRpcStats"
            | "GetSchedStats"
            | "ResetDiagStats"
            | "GetJobMetrics"
            | "GetNodeMetrics"
            | "Ping"
    )
}

/// Extract the gRPC method name from an HTTP/2 `:path` value.
pub(crate) fn grpc_operation_name(path: &str) -> String {
    path.rsplit('/')
        .next()
        .filter(|segment| !segment.is_empty())
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grpc_operation_name_parses_method_from_path() {
        assert_eq!(
            grpc_operation_name("/slurm.SlurmController/SubmitJob"),
            "SubmitJob"
        );
        assert_eq!(grpc_operation_name("/unknown"), "unknown");
    }

    #[test]
    fn should_record_operation_skips_diagnostic_rpcs() {
        assert!(should_record_operation("SubmitJob"));
        assert!(should_record_operation("GetJobs"));
        assert!(!should_record_operation("GetRpcStats"));
        assert!(!should_record_operation("GetSchedStats"));
        assert!(!should_record_operation("ResetDiagStats"));
        assert!(!should_record_operation("GetJobMetrics"));
        assert!(!should_record_operation("GetNodeMetrics"));
        assert!(!should_record_operation("Ping"));
    }

    #[tokio::test]
    async fn middleware_records_leader_rpc_and_skips_diagnostic_names() {
        use std::collections::HashMap;
        use std::convert::Infallible;

        use http::Request;
        use tempfile::TempDir;
        use tower::ServiceExt;

        use crate::cluster::ClusterManager;
        use crate::raft::start_raft;
        use spur_core::config::SlurmConfig;

        #[derive(Clone, Default)]
        struct OkService;

        impl<B> Service<Request<B>> for OkService
        where
            B: Send + 'static,
        {
            type Response = Response<tonic::body::Body>;
            type Error = Infallible;
            type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, _req: Request<B>) -> Self::Future {
                std::future::ready(Ok(Response::new(tonic::body::Body::empty())))
            }
        }

        fn test_config() -> SlurmConfig {
            SlurmConfig {
                cluster_name: "test".into(),
                controller: spur_core::config::ControllerConfig {
                    first_job_id: 1,
                    ..Default::default()
                },
                accounting: Default::default(),
                scheduler: Default::default(),
                auth: Default::default(),
                partitions: vec![spur_core::config::PartitionConfig {
                    name: "default".into(),
                    default: true,
                    state: "UP".into(),
                    nodes: "ALL".into(),
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
                licenses: HashMap::new(),
                update: Default::default(),
                probes: Default::default(),
                metrics: Default::default(),
                rest_api: Default::default(),
                hooks: Default::default(),
                devices: Default::default(),
                admission: Default::default(),
                burst_buffer: Default::default(),
                rlimits: Default::default(),
                cgroup: Default::default(),
                mpi: Default::default(),
            }
        }

        let dir = TempDir::new().unwrap();
        let cm = Arc::new(ClusterManager::new(test_config(), dir.path()).unwrap());
        let handle = start_raft(1, &["[::1]:0".into()], dir.path(), cm)
            .await
            .unwrap();
        handle
            .raft
            .wait(Some(std::time::Duration::from_secs(5)))
            .metrics(|m| m.current_leader == Some(1), "leader elected")
            .await
            .expect("single-node raft did not self-elect within 5s");

        let stats = Arc::new(RpcStatsCollector::new());
        let raft = Arc::new(handle);

        let submit = Request::builder()
            .uri("/slurm.SlurmController/SubmitJob")
            .body(())
            .unwrap();
        RpcStatsMiddleware {
            inner: OkService,
            stats: stats.clone(),
            raft: raft.clone(),
        }
        .oneshot(submit)
        .await
        .unwrap();

        let snap = stats.snapshot();
        assert_eq!(snap.by_operation.len(), 1);
        assert_eq!(snap.by_operation[0].operation, "SubmitJob");
        assert_eq!(snap.by_operation[0].count, 1);

        let ping = Request::builder()
            .uri("/slurm.SlurmController/Ping")
            .body(())
            .unwrap();
        RpcStatsMiddleware {
            inner: OkService,
            stats: stats.clone(),
            raft,
        }
        .oneshot(ping)
        .await
        .unwrap();
        assert_eq!(stats.snapshot().by_operation.len(), 1);
    }
}
