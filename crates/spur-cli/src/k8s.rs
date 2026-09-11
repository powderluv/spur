// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `spur k8s` subcommands: drive the SPUR-managed k0s cluster.

use anyhow::Result;
use clap::{Parser, Subcommand};

use spur_proto::proto::slurm_controller_client::SlurmControllerClient;
use spur_proto::proto::{
    ClusterAddNodesRequest, ClusterDownRequest, ClusterKubeconfigRequest,
    ClusterRemoveNodesRequest, ClusterStatusRequest, ClusterUpRequest,
};

/// Manage the SPUR-provisioned k0s cluster.
#[derive(Parser, Debug)]
#[command(name = "k8s", about = "Manage the SPUR-provisioned k0s cluster")]
pub struct K8sArgs {
    /// Controller address
    #[arg(
        long,
        env = "SPUR_CONTROLLER_ADDR",
        default_value = "http://localhost:6817",
        global = true
    )]
    controller: String,

    #[command(subcommand)]
    pub command: K8sCommand,
}

#[derive(Subcommand, Debug)]
pub enum K8sCommand {
    /// Bring the k0s cluster up (assign roles/IPs, then start each node's component).
    Up {
        /// Control-plane node (default: picked from inventory / [cluster] config).
        #[arg(long)]
        control_plane_node: Option<String>,
        /// HA control-plane count: 1, 3, or 5 (default: [cluster] config, else 1).
        #[arg(long)]
        replicas: Option<u32>,
        /// Explicit control-plane nodes (repeatable; 1, 3, or 5). First is the etcd bootstrap.
        /// Overrides --replicas.
        #[arg(long = "control-plane-nodes", value_delimiter = ',')]
        control_plane_nodes: Vec<String>,
        /// Scope the cluster to a subset of nodes (hostlist, e.g. "gpu[01-08]"), unioned with
        /// --partition/--selector; empty = whole inventory. Resolved once at up time (not re-evaluated).
        #[arg(long)]
        nodes: Option<String>,
        /// Scope the cluster to a partition's nodes.
        #[arg(long)]
        partition: Option<String>,
        /// Scope the cluster to nodes matching every key=value label (repeatable).
        #[arg(long = "selector", value_parser = parse_key_val)]
        selector: Vec<(String, String)>,
    },
    /// Add worker nodes to a running cluster (scoped clusters only; no down/reset needed).
    AddNodes {
        /// Nodes to add (hostlist, e.g. "gpu[09-12]"), unioned with --partition/--selector.
        #[arg(long)]
        nodes: Option<String>,
        /// Add a partition's nodes.
        #[arg(long)]
        partition: Option<String>,
        /// Add nodes matching every key=value label (repeatable).
        #[arg(long = "selector", value_parser = parse_key_val)]
        selector: Vec<(String, String)>,
    },
    /// Remove worker nodes from a running cluster: cordon + drain, then `k0s reset` the node
    /// (destructive — the node re-downloads/re-seeds on a later add). Use `spur node drain` instead
    /// for a temporary "stop scheduling here".
    RemoveNodes {
        /// Worker nodes to remove (hostlist, e.g. "gpu[09-12]").
        #[arg(long)]
        nodes: String,
        /// Max seconds to wait for the k8s drain per node (0 = server default).
        #[arg(long)]
        drain_timeout: Option<u32>,
        /// Proceed even if a node has running jobs (they are left running — this only skips the
        /// busy-node check) or its drain does not complete (bypasses PodDisruptionBudgets).
        #[arg(long)]
        force: bool,
    },
    /// Tear the k0s cluster down.
    Down {
        /// Also `k0s reset` each node (destructive: wipes cluster state).
        #[arg(long)]
        reset: bool,
    },
    /// Show cluster phase + per-node component status.
    Status,
    /// Print a kubeconfig to stdout. Default: your own scope. `--admin`: cluster-admin (admins only).
    /// `--user X`: another user's scope (admins only).
    Kubeconfig {
        /// Mint a scoped kubeconfig for this SPUR user; targeting anyone but yourself needs admin.
        #[arg(long)]
        user: Option<String>,
        /// Fetch the cluster-admin kubeconfig instead of a scoped one (requires cluster admin).
        #[arg(long, conflicts_with = "user")]
        admin: bool,
    },
    /// Download + install the k0s binary on THIS node (local; no controller needed).
    /// Run as root for the default /usr/local/bin path.
    InstallK0s {
        /// k0s release tag to install, or "latest". Defaults to spur's pinned version.
        #[arg(long, default_value_t = String::from(spur_core::k0s::K0S_PINNED_VERSION))]
        version: String,
        /// Install path for the k0s binary.
        #[arg(long, default_value_t = String::from(spur_core::k0s::K0S_DEFAULT_BINARY))]
        path: String,
        /// Reinstall even if a k0s binary already exists at --path.
        #[arg(long)]
        force: bool,
    },
}

pub async fn main() -> Result<()> {
    main_with_args(std::env::args().collect()).await
}

pub async fn main_with_args(args: Vec<String>) -> Result<()> {
    main_with_args_and_user_resolver(args, crate::interactive::current_user).await
}

async fn main_with_args_and_user_resolver(
    args: Vec<String>,
    current_user: impl Fn() -> Result<String>,
) -> Result<()> {
    let parsed = K8sArgs::try_parse_from(args)?;
    let controller = parsed.controller;
    match parsed.command {
        K8sCommand::Up {
            control_plane_node,
            replicas,
            control_plane_nodes,
            nodes,
            partition,
            selector,
        } => {
            let selector = selector_map(selector)?;
            let caller = current_user()?;
            cmd_up(
                &controller,
                caller,
                control_plane_node,
                replicas,
                control_plane_nodes,
                nodes,
                partition,
                selector,
            )
            .await
        }
        K8sCommand::AddNodes {
            nodes,
            partition,
            selector,
        } => cmd_add_nodes(&controller, nodes, partition, selector).await,
        K8sCommand::RemoveNodes {
            nodes,
            drain_timeout,
            force,
        } => cmd_remove_nodes(&controller, nodes, drain_timeout, force).await,
        K8sCommand::Down { reset } => cmd_down(&controller, current_user()?, reset).await,
        K8sCommand::Status => cmd_status(&controller).await,
        K8sCommand::Kubeconfig { user, admin } => {
            cmd_kubeconfig(&controller, current_user()?, user, admin).await
        }
        K8sCommand::InstallK0s {
            version,
            path,
            force,
        } => cmd_install_k0s(&version, &path, force).await,
    }
}

fn effective_user() -> String {
    whoami::username().unwrap_or_else(|_| "unknown".into())
}

fn parse_key_val(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected key=value, got {s}"))?;
    if k.is_empty() {
        return Err(format!("empty selector key in {s}"));
    }
    if v.is_empty() {
        return Err(format!("empty selector value in {s}"));
    }
    Ok((k.to_string(), v.to_string()))
}

/// Fold repeated `--selector key=val` into a map, rejecting a duplicate key rather than silently
/// dropping the earlier value (last-wins would change the intended AND scope).
fn selector_map(
    selector: Vec<(String, String)>,
) -> Result<std::collections::HashMap<String, String>, anyhow::Error> {
    let mut map = std::collections::HashMap::new();
    for (k, v) in selector {
        if map.insert(k.clone(), v).is_some() {
            anyhow::bail!("duplicate --selector key {k}");
        }
    }
    Ok(map)
}

async fn cmd_install_k0s(version: &str, path: &str, force: bool) -> Result<()> {
    let dest = std::path::Path::new(path);
    if dest.exists() && !force {
        eprintln!("k0s already present at {path} (use --force to reinstall)");
        return Ok(());
    }
    eprintln!("Installing k0s {version} -> {path} ...");
    let info = spur_update::k0s::install_k0s(version, dest).await?;
    let short = &info.sha256[..info.sha256.len().min(16)];
    eprintln!(
        "Installed k0s {} to {} (sha256 {}…)",
        info.version,
        info.path.display(),
        short
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn cmd_up(
    controller: &str,
    caller: String,
    control_plane_node: Option<String>,
    replicas: Option<u32>,
    control_plane_nodes: Vec<String>,
    nodes: Option<String>,
    partition: Option<String>,
    selector: std::collections::HashMap<String, String>,
) -> Result<()> {
    let mut client = SlurmControllerClient::new(crate::authclient::connect(controller).await?);
    let resp = client
        .cluster_up(ClusterUpRequest {
            control_plane_node,
            control_plane_replicas: replicas,
            control_plane_nodes,
            caller,
            nodes: nodes.unwrap_or_default(),
            partition: partition.unwrap_or_default(),
            selector,
        })
        .await?
        .into_inner();
    if resp.accepted {
        println!("k0s cluster up requested: {}", resp.message);
    } else {
        eprintln!("k0s cluster up NOT accepted: {}", resp.message);
    }
    for n in resp.nodes {
        println!("  {} [{}] {}", n.node, n.role, n.component_state);
    }
    Ok(())
}

async fn cmd_add_nodes(
    controller: &str,
    nodes: Option<String>,
    partition: Option<String>,
    selector: Vec<(String, String)>,
) -> Result<()> {
    let selector = selector_map(selector)?;
    let mut client = SlurmControllerClient::new(spur_client::connect_channel(controller).await?);
    let resp = client
        .cluster_add_nodes(ClusterAddNodesRequest {
            nodes: nodes.unwrap_or_default(),
            partition: partition.unwrap_or_default(),
            selector,
            caller: effective_user(),
        })
        .await?
        .into_inner();
    if resp.accepted {
        println!("k0s add-nodes requested: {}", resp.message);
    } else {
        eprintln!("k0s add-nodes NOT accepted: {}", resp.message);
    }
    for n in resp.nodes {
        println!("  {} [{}] {}", n.node, n.role, n.component_state);
    }
    Ok(())
}

async fn cmd_remove_nodes(
    controller: &str,
    nodes: String,
    drain_timeout: Option<u32>,
    force: bool,
) -> Result<()> {
    let mut client = SlurmControllerClient::new(spur_client::connect_channel(controller).await?);
    let resp = client
        .cluster_remove_nodes(ClusterRemoveNodesRequest {
            nodes,
            caller: effective_user(),
            drain_timeout_secs: drain_timeout,
            force: Some(force),
        })
        .await?
        .into_inner();
    if resp.accepted {
        println!("k0s remove-nodes requested: {}", resp.message);
    } else {
        eprintln!("k0s remove-nodes NOT accepted: {}", resp.message);
    }
    for n in resp.nodes {
        println!("  {} [{}] {}", n.node, n.role, n.component_state);
    }
    Ok(())
}

async fn cmd_down(controller: &str, caller: String, reset: bool) -> Result<()> {
    let mut client = SlurmControllerClient::new(crate::authclient::connect(controller).await?);
    let resp = client
        .cluster_down(ClusterDownRequest { reset, caller })
        .await?
        .into_inner();
    if resp.accepted {
        println!("k0s cluster down requested: {}", resp.message);
    } else {
        eprintln!("k0s cluster down NOT accepted: {}", resp.message);
    }
    Ok(())
}

async fn cmd_status(controller: &str) -> Result<()> {
    let mut client = SlurmControllerClient::new(crate::authclient::connect(controller).await?);
    let resp = client
        .cluster_status(ClusterStatusRequest {})
        .await?
        .into_inner();
    println!("phase: {}", resp.phase);
    if !resp.control_plane_nodes.is_empty() {
        println!("control-plane: {}", resp.control_plane_nodes.join(", "));
    } else if !resp.control_plane_node.is_empty() {
        println!("control-plane: {}", resp.control_plane_node);
    }
    // Teardown clears the recorded scope immediately, before node roles finish draining; showing
    // "all nodes" here would misrepresent a cluster that's mid-teardown, not freshly scoped.
    if resp.phase != "down" {
        if resp.member_nodes.is_empty() {
            println!("members: all nodes");
        } else {
            println!("members: {}", resp.member_nodes.join(", "));
        }
    }
    for n in resp.nodes {
        print!(
            "  {:<24} {:<11} {:<11} enabled={}",
            n.node, n.role, n.component_state, n.enabled
        );
        if !n.reason.is_empty() {
            print!("  reason: {}", n.reason);
        }
        println!();
    }
    Ok(())
}

async fn cmd_kubeconfig(
    controller: &str,
    caller: String,
    user: Option<String>,
    admin: bool,
) -> Result<()> {
    let mut client = SlurmControllerClient::new(crate::authclient::connect(controller).await?);
    let resp = client
        .cluster_kubeconfig(ClusterKubeconfigRequest {
            user: user.unwrap_or_default(),
            caller,
            admin,
        })
        .await?
        .into_inner();
    // stdout = data (the YAML), so it can be redirected to a kubeconfig file.
    print!("{}", resp.kubeconfig);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_up_with_control_plane() {
        let args =
            K8sArgs::try_parse_from(["k8s", "up", "--control-plane-node", "head-node"]).unwrap();
        match args.command {
            K8sCommand::Up {
                control_plane_node,
                replicas,
                control_plane_nodes,
                ..
            } => {
                assert_eq!(control_plane_node.as_deref(), Some("head-node"));
                assert_eq!(replicas, None);
                assert!(control_plane_nodes.is_empty());
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn parses_up_with_node_scope_flags() {
        let args = K8sArgs::try_parse_from([
            "k8s",
            "up",
            "--nodes",
            "gpu[01-08]",
            "--partition",
            "batch",
            "--selector",
            "zone=z1",
            "--selector",
            "gpu=mi300",
        ])
        .unwrap();
        match args.command {
            K8sCommand::Up {
                nodes,
                partition,
                selector,
                ..
            } => {
                assert_eq!(nodes.as_deref(), Some("gpu[01-08]"));
                assert_eq!(partition.as_deref(), Some("batch"));
                assert_eq!(
                    selector,
                    vec![
                        ("zone".to_string(), "z1".to_string()),
                        ("gpu".to_string(), "mi300".to_string())
                    ]
                );
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn selector_without_equals_is_rejected() {
        assert!(K8sArgs::try_parse_from(["k8s", "up", "--selector", "bogus"]).is_err());
    }

    #[test]
    fn selector_with_empty_value_is_rejected() {
        assert!(K8sArgs::try_parse_from(["k8s", "up", "--selector", "gpu="]).is_err());
    }

    #[test]
    fn selector_map_rejects_duplicate_key() {
        let dup = vec![
            ("zone".to_string(), "z1".to_string()),
            ("zone".to_string(), "z2".to_string()),
        ];
        let err = selector_map(dup).unwrap_err().to_string();
        assert!(err.contains("duplicate --selector key zone"), "got: {err}");
        let ok = selector_map(vec![
            ("zone".to_string(), "z1".to_string()),
            ("gpu".to_string(), "mi300".to_string()),
        ])
        .unwrap();
        assert_eq!(ok.len(), 2);
    }

    #[test]
    fn parses_up_with_replicas_and_node_set() {
        let args = K8sArgs::try_parse_from(["k8s", "up", "--replicas", "3"]).unwrap();
        match args.command {
            K8sCommand::Up { replicas, .. } => assert_eq!(replicas, Some(3)),
            _ => panic!("wrong command"),
        }
        let args =
            K8sArgs::try_parse_from(["k8s", "up", "--control-plane-nodes", "cp-1,cp-2,cp-3"])
                .unwrap();
        match args.command {
            K8sCommand::Up {
                control_plane_nodes,
                ..
            } => assert_eq!(control_plane_nodes, vec!["cp-1", "cp-2", "cp-3"]),
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn parses_down_reset_and_status() {
        let args = K8sArgs::try_parse_from(["k8s", "down", "--reset"]).unwrap();
        assert!(matches!(args.command, K8sCommand::Down { reset: true }));
        let args = K8sArgs::try_parse_from(["k8s", "status"]).unwrap();
        assert!(matches!(args.command, K8sCommand::Status));
    }

    #[test]
    fn parses_add_nodes_scope_flags() {
        let args = K8sArgs::try_parse_from(["k8s", "add-nodes", "--nodes", "gpu[09-12]"]).unwrap();
        match args.command {
            K8sCommand::AddNodes { nodes, .. } => assert_eq!(nodes.as_deref(), Some("gpu[09-12]")),
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn parses_remove_nodes_with_flags() {
        let args = K8sArgs::try_parse_from([
            "k8s",
            "remove-nodes",
            "--nodes",
            "gpu[09-12]",
            "--drain-timeout",
            "300",
            "--force",
        ])
        .unwrap();
        match args.command {
            K8sCommand::RemoveNodes {
                nodes,
                drain_timeout,
                force,
            } => {
                assert_eq!(nodes, "gpu[09-12]");
                assert_eq!(drain_timeout, Some(300));
                assert!(force);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn remove_nodes_requires_nodes() {
        // --nodes is mandatory for remove-nodes.
        assert!(K8sArgs::try_parse_from(["k8s", "remove-nodes"]).is_err());
    }

    #[test]
    fn controller_defaults_and_env() {
        let args = K8sArgs::try_parse_from(["k8s", "status"]).unwrap();
        assert_eq!(args.controller, "http://localhost:6817");
    }

    #[test]
    fn kubeconfig_bare_is_self_scoped() {
        let args = K8sArgs::try_parse_from(["k8s", "kubeconfig"]).unwrap();
        match args.command {
            K8sCommand::Kubeconfig { user, admin } => {
                assert_eq!(user, None);
                assert!(!admin);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn kubeconfig_admin_flag_parses() {
        let args = K8sArgs::try_parse_from(["k8s", "kubeconfig", "--admin"]).unwrap();
        match args.command {
            K8sCommand::Kubeconfig { admin, .. } => assert!(admin),
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn kubeconfig_admin_and_user_conflict() {
        let err =
            K8sArgs::try_parse_from(["k8s", "kubeconfig", "--admin", "--user", "bob"]).unwrap_err();
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::ArgumentConflict,
            "--admin and --user must be mutually exclusive"
        );
    }

    #[tokio::test]
    async fn authenticated_commands_fail_before_dispatch_when_user_lookup_fails() {
        let (addr, capture) = crate::mock_controller::spawn().await;
        let controller = format!("http://{addr}");

        for command in [vec!["up"], vec!["down"], vec!["kubeconfig"]] {
            let mut args = vec!["k8s".to_string(), "--controller".into(), controller.clone()];
            args.extend(command.into_iter().map(str::to_string));

            let error = main_with_args_and_user_resolver(args, || {
                Err(anyhow::anyhow!("failed to determine current username"))
            })
            .await
            .unwrap_err();

            assert_eq!(error.to_string(), "failed to determine current username");
        }

        assert!(capture.k8s_requests().is_empty());
    }

    #[tokio::test]
    async fn duplicate_selectors_fail_before_user_resolution_and_dispatch() {
        let (addr, capture) = crate::mock_controller::spawn().await;
        let resolutions = std::cell::Cell::new(0);

        let error = main_with_args_and_user_resolver(
            vec![
                "k8s".into(),
                "--controller".into(),
                format!("http://{addr}"),
                "up".into(),
                "--selector".into(),
                "zone=z1".into(),
                "--selector".into(),
                "zone=z2".into(),
            ],
            || {
                resolutions.set(resolutions.get() + 1);
                Ok("fixture-user".into())
            },
        )
        .await
        .unwrap_err();

        assert_eq!(error.to_string(), "duplicate --selector key zone");
        assert_eq!(resolutions.get(), 0);
        assert!(capture.k8s_requests().is_empty());
    }

    #[tokio::test]
    async fn authenticated_commands_send_the_resolved_caller() {
        use crate::mock_controller::K8sRequest;

        let (addr, capture) = crate::mock_controller::spawn().await;
        let controller = format!("http://{addr}");

        main_with_args_and_user_resolver(
            vec![
                "k8s".into(),
                "--controller".into(),
                controller.clone(),
                "up".into(),
                "--replicas".into(),
                "3".into(),
            ],
            || Ok("fixture-user".into()),
        )
        .await
        .unwrap();
        main_with_args_and_user_resolver(
            vec![
                "k8s".into(),
                "--controller".into(),
                controller.clone(),
                "down".into(),
                "--reset".into(),
            ],
            || Ok("fixture-user".into()),
        )
        .await
        .unwrap();
        main_with_args_and_user_resolver(
            vec![
                "k8s".into(),
                "--controller".into(),
                controller,
                "kubeconfig".into(),
                "--user".into(),
                "target-user".into(),
            ],
            || Ok("fixture-user".into()),
        )
        .await
        .unwrap();

        let requests = capture.k8s_requests();
        match requests.as_slice() {
            [K8sRequest::Up(up), K8sRequest::Down(down), K8sRequest::Kubeconfig(kubeconfig)] => {
                assert_eq!(up.caller, "fixture-user");
                assert_eq!(up.control_plane_replicas, Some(3));
                assert_eq!(down.caller, "fixture-user");
                assert!(down.reset);
                assert_eq!(kubeconfig.caller, "fixture-user");
                assert_eq!(kubeconfig.user, "target-user");
            }
            requests => panic!("unexpected k8s requests: {requests:?}"),
        }
    }

    #[tokio::test]
    async fn unauthenticated_commands_do_not_resolve_a_caller() {
        let (addr, capture) = crate::mock_controller::spawn().await;
        let controller = format!("http://{addr}");
        let resolutions = std::cell::Cell::new(0);
        let unresolved = || {
            resolutions.set(resolutions.get() + 1);
            Err(anyhow::anyhow!("failed to determine current username"))
        };

        main_with_args_and_user_resolver(
            vec![
                "k8s".into(),
                "--controller".into(),
                controller,
                "status".into(),
            ],
            &unresolved,
        )
        .await
        .unwrap();

        let temp = tempfile::tempdir().unwrap();
        let existing_k0s = temp.path().join("k0s");
        std::fs::write(&existing_k0s, b"fixture").unwrap();
        main_with_args_and_user_resolver(
            vec![
                "k8s".into(),
                "install-k0s".into(),
                "--path".into(),
                existing_k0s.to_string_lossy().into_owned(),
            ],
            &unresolved,
        )
        .await
        .unwrap();

        assert_eq!(resolutions.get(), 0);
        assert_eq!(
            capture.k8s_requests(),
            vec![crate::mock_controller::K8sRequest::Status]
        );
    }
}
