// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use tokio::sync::RwLock;
use tracing::{debug, warn};

use spur_proto::proto::{HeartbeatRequest, RegisterAgentRequest};

// Matches spurd's 30 s interval, well inside `controller.heartbeat_timeout_secs`.
const INTERVAL_SECS: u64 = 30;

/// Tracks the set of active K8s nodes and sends periodic `Heartbeat` RPCs
/// to spurctld on their behalf, mirroring what `spurd`'s `reporter::heartbeat_loop`
/// does for native-host nodes.
///
/// `node_watcher` holds an `Arc<HeartbeatManager>` and calls `track`/`untrack`
/// as nodes appear and disappear; the heartbeat task calls `run` under
/// `retry::run_with_retry`.
pub struct HeartbeatManager {
    registry: RwLock<HashMap<String, RegisterAgentRequest>>,
    controller_addr: String,
}

impl HeartbeatManager {
    pub fn new(controller_addr: String) -> Self {
        Self {
            registry: RwLock::new(HashMap::new()),
            controller_addr,
        }
    }

    /// Add or update a node in the tracked set.
    pub async fn track(&self, name: String, req: RegisterAgentRequest) {
        self.registry.write().await.insert(name, req);
    }

    /// Remove a node from the tracked set. Safe to call for unknown names.
    pub async fn untrack(&self, name: &str) {
        self.registry.write().await.remove(name);
    }

    /// Send `Heartbeat` RPCs to spurctld for every tracked node.
    pub async fn run(&self) -> anyhow::Result<()> {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(INTERVAL_SECS));
        loop {
            interval.tick().await;

            let names: Vec<String> = self.registry.read().await.keys().cloned().collect();
            if names.is_empty() {
                continue;
            }

            match crate::controller::connect(&self.controller_addr).await {
                Ok(mut client) => {
                    for name in &names {
                        let req = HeartbeatRequest {
                            hostname: name.clone(),
                            cpu_load: 0,
                            free_memory_mb: 0,
                            running_jobs: vec![],
                            node_token: String::new(),
                            wg_pubkey: String::new(), // virtual agents are not on the mesh
                            k0s_status: None,         // virtual agents run no k0s unit
                        };
                        match client.heartbeat(req).await {
                            Ok(_) => debug!(node = %name, "heartbeat sent"),
                            Err(e) => warn!(node = %name, error = %e, "heartbeat failed"),
                        }
                    }
                }
                Err(e) => warn!(error = %e, "heartbeat: failed to connect to spurctld"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_req(hostname: &str) -> RegisterAgentRequest {
        RegisterAgentRequest {
            hostname: hostname.into(),
            resources: None,
            version: "test".into(),
            address: "127.0.0.1".into(),
            port: 6818,
            wg_pubkey: String::new(),
            labels: std::collections::HashMap::new(),
            join_token: String::new(),
        }
    }

    #[tokio::test]
    async fn test_new_registry_is_empty() {
        let hb = HeartbeatManager::new("http://localhost:6817".into());
        assert!(hb.registry.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_track_adds_node() {
        let hb = HeartbeatManager::new("http://localhost:6817".into());
        hb.track("node-1".into(), make_req("node-1")).await;
        assert!(hb.registry.read().await.contains_key("node-1"));
    }

    #[tokio::test]
    async fn test_untrack_removes_node() {
        let hb = HeartbeatManager::new("http://localhost:6817".into());
        hb.track("node-1".into(), make_req("node-1")).await;
        hb.untrack("node-1").await;
        assert!(hb.registry.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_untrack_unknown_name_is_safe() {
        let hb = HeartbeatManager::new("http://localhost:6817".into());
        hb.untrack("does-not-exist").await;
        assert!(hb.registry.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_track_idempotent_updates_entry() {
        let hb = HeartbeatManager::new("http://localhost:6817".into());
        hb.track("node-1".into(), make_req("node-1")).await;

        let mut updated = make_req("node-1");
        updated.address = "10.0.0.1".into();
        hb.track("node-1".into(), updated).await;

        let guard = hb.registry.read().await;
        assert_eq!(guard.len(), 1);
        assert_eq!(guard["node-1"].address, "10.0.0.1");
    }

    #[tokio::test]
    async fn test_multiple_nodes_tracked_independently() {
        let hb = HeartbeatManager::new("http://localhost:6817".into());
        hb.track("node-1".into(), make_req("node-1")).await;
        hb.track("node-2".into(), make_req("node-2")).await;
        hb.track("node-3".into(), make_req("node-3")).await;
        assert_eq!(hb.registry.read().await.len(), 3);
    }

    #[tokio::test]
    async fn test_untrack_one_of_many_leaves_others() {
        let hb = HeartbeatManager::new("http://localhost:6817".into());
        hb.track("node-1".into(), make_req("node-1")).await;
        hb.track("node-2".into(), make_req("node-2")).await;
        hb.untrack("node-1").await;

        let guard = hb.registry.read().await;
        assert_eq!(guard.len(), 1);
        assert!(!guard.contains_key("node-1"));
        assert!(guard.contains_key("node-2"));
    }

    #[tokio::test]
    async fn test_track_after_untrack_re_adds_node() {
        let hb = HeartbeatManager::new("http://localhost:6817".into());
        hb.track("node-1".into(), make_req("node-1")).await;
        hb.untrack("node-1").await;
        hb.track("node-1".into(), make_req("node-1")).await;

        assert_eq!(hb.registry.read().await.len(), 1);
    }

    #[tokio::test]
    async fn test_untrack_all_leaves_empty_registry() {
        let hb = HeartbeatManager::new("http://localhost:6817".into());
        hb.track("node-1".into(), make_req("node-1")).await;
        hb.track("node-2".into(), make_req("node-2")).await;
        hb.untrack("node-1").await;
        hb.untrack("node-2").await;
        assert!(hb.registry.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_register_req_preserved_for_reregistration() {
        let hb = HeartbeatManager::new("http://localhost:6817".into());
        let req = make_req("node-1");
        hb.track("node-1".into(), req.clone()).await;
        hb.track("node-2".into(), make_req("node-2")).await;
        hb.untrack("node-2").await;

        let guard = hb.registry.read().await;
        let stored = guard.get("node-1").expect("node-1 must still be tracked");
        assert_eq!(stored.hostname, req.hostname);
        assert_eq!(stored.address, req.address);
        assert_eq!(stored.port, req.port);
    }
}
