// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP health, readiness, and metrics server for the K8s operator.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use kube::Client;
use tracing::{debug, info};

struct HealthState {
    k8s_client: Client,
    controller_addr: String,
}

/// Start the health/readiness HTTP server.
pub async fn serve(
    addr: SocketAddr,
    k8s_client: Client,
    controller_addr: String,
) -> anyhow::Result<()> {
    let state = Arc::new(HealthState {
        k8s_client,
        controller_addr,
    });

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .with_state(state);

    info!(%addr, "health server listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Liveness probe — always OK if process is running.
async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Readiness probe — checks K8s API and spurctld reachability.
async fn readyz(State(state): State<Arc<HealthState>>) -> impl IntoResponse {
    // Check K8s API
    let k8s_ok = state.k8s_client.apiserver_version().await.is_ok();

    // Check spurctld reachability
    let ctrl_ok = crate::controller::connect(&state.controller_addr)
        .await
        .is_ok();

    if k8s_ok && ctrl_ok {
        debug!("readyz: ok");
        (StatusCode::OK, "ok".to_string())
    } else {
        let mut reasons = Vec::new();
        if !k8s_ok {
            reasons.push("k8s-api-unreachable");
        }
        if !ctrl_ok {
            reasons.push("spurctld-unreachable");
        }
        (StatusCode::SERVICE_UNAVAILABLE, reasons.join(","))
    }
}

/// Prometheus metrics endpoint (minimal for now).
async fn metrics() -> impl IntoResponse {
    // Basic process metrics
    let uptime = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let body = format!(
        "# HELP spur_k8s_operator_up Whether the operator is running\n\
         # TYPE spur_k8s_operator_up gauge\n\
         spur_k8s_operator_up 1\n\
         # HELP spur_k8s_operator_timestamp_seconds Current unix timestamp\n\
         # TYPE spur_k8s_operator_timestamp_seconds gauge\n\
         spur_k8s_operator_timestamp_seconds {}\n",
        uptime
    );

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        body,
    )
}
