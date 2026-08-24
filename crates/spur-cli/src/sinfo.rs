// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use clap::Parser;
use spur_proto::proto::{GetNodesRequest, GetPartitionsRequest, NodeInfo, PartitionInfo};

use crate::format_engine;

/// View information about nodes and partitions.
#[derive(Parser, Debug)]
#[command(
    name = "sinfo",
    about = "View cluster information",
    disable_help_flag = true
)]
pub struct SinfoArgs {
    /// Show only this partition
    #[arg(short = 'p', long)]
    pub partition: Option<String>,

    /// Show only nodes in these states
    #[arg(short = 't', long)]
    pub states: Option<String>,

    /// Show only these nodes (hostlist)
    #[arg(short = 'n', long)]
    pub nodes: Option<String>,

    /// Output format
    #[arg(short = 'o', long)]
    pub format: Option<String>,

    /// Long format
    #[arg(short = 'l', long)]
    pub long: bool,

    /// Node-oriented (one line per node)
    #[arg(short = 'N', long)]
    pub node_oriented: bool,

    /// Don't print header
    #[arg(short = 'h', long)]
    pub noheader: bool,

    /// Accepted for Slurm compatibility; has no effect
    #[arg(long)]
    pub noconvert: bool,

    /// Controller address
    #[arg(
        long,
        env = "SPUR_CONTROLLER_ADDR",
        default_value = "http://localhost:6817"
    )]
    pub controller: String,

    /// Print help
    #[arg(long, action = clap::ArgAction::Help)]
    pub help: Option<bool>,
}

pub async fn main() -> Result<()> {
    main_with_args(std::env::args().collect()).await
}

pub async fn main_with_args(args: Vec<String>) -> Result<()> {
    let args = SinfoArgs::try_parse_from(&args)?;

    let fmt = if let Some(ref f) = args.format {
        f.clone()
    } else if args.long {
        "%#P %5a %.10l %.4D %.6t %.8c %.8m %N".to_string()
    } else if args.node_oriented {
        "%#N %.6D %#P %.11T %.4c %.8m %G".to_string()
    } else {
        format_engine::SINFO_DEFAULT_FORMAT.to_string()
    };

    let fields = format_engine::parse_format(&fmt, &format_engine::sinfo_header);

    // Built before connecting so an invalid `-t` fails without a round-trip.
    let nodes_req = build_get_nodes_request(&args)?;

    let channel = crate::authclient::connect(&args.controller)
        .await
        .context("failed to connect to spurctld")?;
    let mut client = spur_proto::controller_client(channel);

    // Get partitions
    let partitions_resp = client
        .get_partitions(GetPartitionsRequest {
            name: args.partition.clone().unwrap_or_default(),
        })
        .await
        .context("failed to get partitions")?;

    let partitions = partitions_resp.into_inner().partitions;

    // Get nodes
    let nodes_resp = client
        .get_nodes(nodes_req)
        .await
        .context("failed to get nodes")?;

    let nodes = nodes_resp.into_inner().nodes;

    // Print header
    if !args.noheader {
        println!("{}", format_engine::format_header(&fields));
    }
    for line in render_sinfo_output(&fields, &partitions, &nodes, args.node_oriented) {
        println!("{}", line);
    }

    Ok(())
}

fn build_get_nodes_request(args: &SinfoArgs) -> Result<GetNodesRequest> {
    let states = match args.states.as_deref() {
        Some(s) => parse_states_arg(s)?,
        None => Vec::new(),
    };

    Ok(GetNodesRequest {
        states: states.iter().map(|s| *s as i32).collect(),
        partition: args.partition.clone().unwrap_or_default(),
        nodelist: args.nodes.clone().unwrap_or_default(),
    })
}

/// Parse `-t` / `--states` (comma-separated). Whole-string `all` means no state filter.
/// Unknown tokens are rejected (Slurm exits with an error rather than showing all nodes).
fn parse_states_arg(s: &str) -> Result<Vec<spur_proto::proto::NodeState>> {
    use spur_core::node::NodeState;

    let trimmed = s.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        return Ok(Vec::new());
    }

    let tokens: Vec<&str> = trimmed
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();

    if tokens.is_empty() {
        anyhow::bail!("Invalid node state specified: (empty)");
    }

    let mut states = Vec::with_capacity(tokens.len());
    for token in tokens {
        let core = NodeState::from_short_or_name(token)
            .ok_or_else(|| anyhow::anyhow!("Invalid node state specified: {token}"))?;
        states.push(core.to_proto());
    }
    Ok(states)
}

fn group_nodes_by_display_state<'a>(nodes: &[&'a NodeInfo]) -> Vec<(String, Vec<&'a NodeInfo>)> {
    let mut groups: BTreeMap<String, Vec<&'a NodeInfo>> = BTreeMap::new();
    for node in nodes {
        let key = effective_state_str(node);
        groups.entry(key).or_default().push(node);
    }
    groups.into_iter().collect()
}

fn render_sinfo_output(
    fields: &[format_engine::FormatToken],
    partitions: &[PartitionInfo],
    nodes: &[NodeInfo],
    node_oriented: bool,
) -> Vec<String> {
    let mut lines = Vec::new();

    if node_oriented {
        for node in nodes {
            let no_partition = String::new();
            let parts: &[String] = if node.partitions.is_empty() {
                std::slice::from_ref(&no_partition)
            } else {
                &node.partitions
            };
            for part_name in parts {
                let row = format_engine::format_row(fields, &|spec| {
                    resolve_node_field(node, part_name, partitions, spec)
                });
                lines.push(row);
            }
        }
    } else {
        for part in partitions {
            let part_nodes: Vec<_> = nodes
                .iter()
                .filter(|n| n.partitions.contains(&part.name))
                .collect();
            let state_groups = group_nodes_by_display_state(&part_nodes);

            if state_groups.is_empty() {
                let row = format_engine::format_row(fields, &|spec| {
                    resolve_partition_field(part, &[], spec)
                });
                lines.push(row);
            } else {
                for (_, group_nodes) in &state_groups {
                    let row = format_engine::format_row(fields, &|spec| {
                        resolve_partition_field(part, group_nodes, spec)
                    });
                    lines.push(row);
                }
            }
        }
    }

    lines
}

fn resolve_node_field(
    node: &spur_proto::proto::NodeInfo,
    partition_name: &str,
    _partitions: &[spur_proto::proto::PartitionInfo],
    spec: char,
) -> String {
    match spec {
        'N' | 'n' => node.name.clone(),
        'P' | 'R' => partition_name.to_string(),
        't' | 'T' => effective_state_str(node),
        'c' => {
            if let Some(ref r) = node.total_resources {
                r.cpus.to_string()
            } else {
                "0".into()
            }
        }
        'm' => {
            if let Some(ref r) = node.total_resources {
                r.memory_mb.to_string()
            } else {
                "0".into()
            }
        }
        'G' => {
            if let Some(ref r) = node.total_resources {
                if r.gpus.is_empty() {
                    "(null)".into()
                } else {
                    r.gpus
                        .iter()
                        .map(|g| format!("gpu:{}:{}", g.gpu_type, 1))
                        .collect::<Vec<_>>()
                        .join(",")
                }
            } else {
                "(null)".into()
            }
        }
        'D' => "1".into(),
        'a' => {
            if node.state == spur_proto::proto::NodeState::NodeDown as i32 {
                "down".into()
            } else {
                "up".into()
            }
        }
        'O' => node.cpu_load.to_string(),
        'e' => node.free_memory_mb.to_string(),
        'f' => {
            if node.features.is_empty() {
                "(null)".into()
            } else {
                node.features.join(",")
            }
        }
        'l' => "UNLIMITED".into(), // Would need partition context
        _ => "?".into(),
    }
}

fn resolve_partition_field(
    part: &spur_proto::proto::PartitionInfo,
    nodes: &[&spur_proto::proto::NodeInfo],
    spec: char,
) -> String {
    match spec {
        'P' | 'R' => {
            if part.is_default {
                format!("{}*", part.name)
            } else {
                part.name.clone()
            }
        }
        'a' => part.state.clone(),
        'l' => {
            if let Some(ref mt) = part.max_time {
                spur_core::config::format_time(Some((mt.seconds / 60) as u32))
            } else {
                "infinite".into()
            }
        }
        'D' => nodes.len().to_string(),
        't' | 'T' => {
            if nodes.is_empty() {
                "n/a".into()
            } else {
                effective_state_str(nodes[0])
            }
        }
        'N' => {
            let names: Vec<String> = nodes.iter().map(|n| n.name.clone()).collect();
            spur_core::hostlist::compress(&names)
        }
        'n' => {
            // Expanded form of the same sorted hostlist as `%N`, so `%n` and
            // `%N` stay consistent (Slurm derives both from one sorted list).
            let names: Vec<String> = nodes.iter().map(|n| n.name.clone()).collect();
            let compressed = spur_core::hostlist::compress(&names);
            spur_core::hostlist::expand(&compressed)
                .unwrap_or_else(|_| {
                    let mut fallback = names;
                    fallback.sort();
                    fallback.dedup();
                    fallback
                })
                .join(",")
        }
        'c' => part.total_cpus.to_string(),
        _ => "?".into(),
    }
}

fn effective_state_str(node: &NodeInfo) -> String {
    match spur_core::node::node_overlay(node) {
        Some(spur_core::node::NodeOverlay::Reserved { maint: true }) => "maint".into(),
        Some(spur_core::node::NodeOverlay::Reserved { maint: false }) => "resv".into(),
        Some(spur_core::node::NodeOverlay::Planned) => "plnd".into(),
        None => spur_core::node::NodeState::from_proto_i32(node.state)
            .map(|s| s.short().to_string())
            .unwrap_or_else(|| "unk".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spur_proto::proto as pb;
    use spur_proto::proto::slurm_controller_server::SlurmController;
    use spur_proto::proto::{NodeState, ResourceSet};
    use std::sync::{Arc, Mutex};
    use tonic::{Request, Response, Status};

    fn make_node(name: &str, state: NodeState, partition: &str) -> NodeInfo {
        NodeInfo {
            name: name.into(),
            state: state as i32,
            partitions: vec![partition.into()],
            ..Default::default()
        }
    }

    fn make_partition(name: &str, is_default: bool) -> PartitionInfo {
        PartitionInfo {
            name: name.into(),
            state: "up".into(),
            is_default,
            ..Default::default()
        }
    }

    fn default_fields() -> Vec<format_engine::FormatToken> {
        format_engine::parse_format(
            format_engine::SINFO_DEFAULT_FORMAT,
            &format_engine::sinfo_header,
        )
    }

    // --- state filter plumbing tests ---

    fn parse_sinfo_args(argv: &[&str]) -> SinfoArgs {
        SinfoArgs::try_parse_from(argv).unwrap()
    }

    #[test]
    fn state_filter_reaches_the_request() {
        let args = parse_sinfo_args(&["sinfo", "-t", "idle"]);
        let req = build_get_nodes_request(&args).unwrap();
        assert_eq!(req.states, vec![NodeState::NodeIdle as i32]);
    }

    #[test]
    fn noconvert_is_accepted() {
        // Slurm scripts pass --noconvert to keep output machine-parseable.
        let args = parse_sinfo_args(&["sinfo", "--noconvert"]);
        assert!(args.noconvert);
    }

    #[test]
    fn memory_columns_render_as_raw_megabytes_with_and_without_noconvert() {
        // Pins that %m/%e are raw MB today; humanizing them later turns this red.
        let mut node = make_node("gpu001", NodeState::NodeIdle, "gpu");
        node.total_resources = Some(ResourceSet {
            memory_mb: 2_321_924,
            ..Default::default()
        });
        node.free_memory_mb = 11_077;
        let partitions = vec![make_partition("gpu", true)];

        for argv in [
            ["sinfo", "-N", "-h", "-o", "%m %e"].as_slice(),
            ["sinfo", "-N", "-h", "-o", "%m %e", "--noconvert"].as_slice(),
        ] {
            let args = parse_sinfo_args(argv);
            let fields = format_engine::parse_format(
                args.format.as_deref().expect("-o sets format"),
                &format_engine::sinfo_header,
            );

            let lines = render_sinfo_output(
                &fields,
                &partitions,
                std::slice::from_ref(&node),
                args.node_oriented,
            );

            assert_eq!(lines, ["2321924 11077"], "argv: {argv:?}");
        }
    }

    #[test]
    fn state_filter_accepts_comma_separated_short_and_long_names() {
        let args = parse_sinfo_args(&["sinfo", "--states", "alloc,DOWN,draining"]);
        let req = build_get_nodes_request(&args).unwrap();
        assert_eq!(
            req.states,
            vec![
                NodeState::NodeAllocated as i32,
                NodeState::NodeDown as i32,
                NodeState::NodeDraining as i32,
            ]
        );
    }

    #[test]
    fn no_state_filter_leaves_request_states_empty() {
        let args = parse_sinfo_args(&["sinfo"]);
        let req = build_get_nodes_request(&args).unwrap();
        assert!(req.states.is_empty());
    }

    #[test]
    fn state_filter_all_means_no_filter() {
        for spec in ["all", "ALL"] {
            let args = parse_sinfo_args(&["sinfo", "-t", spec]);
            assert!(build_get_nodes_request(&args).unwrap().states.is_empty());
        }
    }

    #[test]
    fn state_filter_rejects_unknown_state() {
        let args = parse_sinfo_args(&["sinfo", "-t", "BOGUS"]);
        let err = build_get_nodes_request(&args).unwrap_err();
        assert!(err.to_string().contains("BOGUS"), "{err}");
    }

    #[test]
    fn state_filter_rejects_empty_list() {
        let args = parse_sinfo_args(&["sinfo", "-t", "  ,  "]);
        assert!(build_get_nodes_request(&args).is_err());
    }

    #[test]
    fn partition_and_nodelist_still_reach_the_request() {
        let args = parse_sinfo_args(&["sinfo", "-p", "batch", "-n", "n[1-2]"]);
        let req = build_get_nodes_request(&args).unwrap();
        assert_eq!(req.partition, "batch");
        assert_eq!(req.nodelist, "n[1-2]");
    }

    #[test]
    fn short_h_is_noheader_not_help() {
        assert!(parse_sinfo_args(&["sinfo", "-h"]).noheader);
    }

    #[tokio::test]
    async fn invalid_state_fails_before_connecting() {
        // Errors while building the request, so no server/network is needed here.
        let err = main_with_args(vec!["sinfo".into(), "-t".into(), "BOGUS".into()])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("BOGUS"), "{err}");
    }

    /// Serves the two RPCs sinfo issues and records the `GetNodesRequest` it
    /// saw, so a test can assert what actually went over the wire.
    #[derive(Default)]
    struct StubController {
        nodes_req: Arc<Mutex<Option<pb::GetNodesRequest>>>,
    }

    /// Generates the impl so `async_trait` runs after expansion. Only the two
    /// RPCs sinfo calls need behavior; the rest just have to exist.
    macro_rules! stub_controller {
        ($($name:ident($req:ty) -> $resp:ty;)*) => {
            #[tonic::async_trait]
            impl SlurmController for StubController {
                async fn get_partitions(
                    &self,
                    _: Request<pb::GetPartitionsRequest>,
                ) -> Result<Response<pb::GetPartitionsResponse>, Status> {
                    Ok(Response::new(pb::GetPartitionsResponse {
                        partitions: vec![make_partition("batch", true)],
                    }))
                }

                async fn get_nodes(
                    &self,
                    request: Request<pb::GetNodesRequest>,
                ) -> Result<Response<pb::GetNodesResponse>, Status> {
                    *self.nodes_req.lock().unwrap() = Some(request.into_inner());
                    Ok(Response::new(pb::GetNodesResponse {
                        nodes: vec![make_node("n1", NodeState::NodeIdle, "batch")],
                    }))
                }

                $(
                    async fn $name(&self, _: Request<$req>) -> Result<Response<$resp>, Status> {
                        Err(Status::unimplemented(stringify!($name)))
                    }
                )*
            }
        };
    }

    stub_controller! {
        submit_job(pb::SubmitJobRequest) -> pb::SubmitJobResponse;
        get_jobs(pb::GetJobsRequest) -> pb::GetJobsResponse;
        get_job(pb::GetJobRequest) -> pb::JobInfo;
        cancel_job(pb::CancelJobRequest) -> ();
        complete_job(pb::CompleteJobRequest) -> ();
        job_keepalive(pb::JobKeepaliveRequest) -> pb::JobKeepaliveResponse;
        suspend_job(pb::SuspendJobRequest) -> ();
        resume_job(pb::ResumeJobRequest) -> ();
        update_job(pb::UpdateJobRequest) -> ();
        requeue_job(pb::RequeueJobRequest) -> pb::RequeueJobResponse;
        get_node(pb::GetNodeRequest) -> pb::NodeInfo;
        update_node(pb::UpdateNodeRequest) -> ();
        drain_node(pb::DrainNodeRequest) -> pb::DrainNodeResponse;
        deregister_node(pb::DeregisterNodeRequest) -> pb::DeregisterNodeResponse;
        deregister_agent(pb::DeregisterAgentRequest) -> ();
        get_job_steps(pb::GetJobStepsRequest) -> pb::GetJobStepsResponse;
        create_job_step(pb::CreateJobStepRequest) -> pb::CreateJobStepResponse;
        complete_job_step(pb::CompleteJobStepRequest) -> ();
        create_partition(pb::CreatePartitionRequest) -> ();
        update_partition(pb::UpdatePartitionRequest) -> ();
        delete_partition(pb::DeletePartitionRequest) -> ();
        reconfigure(()) -> ();
        ping(()) -> pb::PingResponse;
        get_job_metrics(()) -> pb::JobMetrics;
        get_node_metrics(()) -> pb::NodeMetrics;
        get_rpc_stats(()) -> pb::RpcStats;
        reset_diag_stats(()) -> ();
        get_sched_stats(()) -> pb::SchedStats;
        get_assoc_mgr_info(pb::GetAssocMgrInfoRequest) -> pb::GetAssocMgrInfoResponse;
        register_agent(pb::RegisterAgentRequest) -> pb::RegisterAgentResponse;
        heartbeat(pb::HeartbeatRequest) -> pb::HeartbeatResponse;
        create_token(pb::CreateTokenRequest) -> pb::CreateTokenResponse;
        list_tokens(pb::ListTokensRequest) -> pb::ListTokensResponse;
        revoke_token(pb::RevokeTokenRequest) -> pb::RevokeTokenResponse;
        report_job_status(pb::ReportJobStatusRequest) -> ();
        report_stepd_recovery(pb::StepdRecoveryRequest) -> pb::StepdRecoveryResponse;
        create_reservation(pb::CreateReservationRequest) -> ();
        update_reservation(pb::UpdateReservationRequest) -> ();
        delete_reservation(pb::DeleteReservationRequest) -> ();
        list_reservations(pb::ListReservationsRequest) -> pb::ListReservationsResponse;
        exec_in_job(pb::ExecInJobRequest) -> pb::ExecInJobResponse;
        run_step(pb::RunStepRequest) -> pb::RunStepResponse;
        cluster_up(pb::ClusterUpRequest) -> pb::ClusterUpResponse;
        cluster_down(pb::ClusterDownRequest) -> pb::ClusterDownResponse;
        cluster_status(pb::ClusterStatusRequest) -> pb::ClusterStatusResponse;
        cluster_kubeconfig(pb::ClusterKubeconfigRequest) -> pb::ClusterKubeconfigResponse;
        cluster_add_nodes(pb::ClusterAddNodesRequest) -> pb::ClusterAddNodesResponse;
        cluster_remove_nodes(pb::ClusterRemoveNodesRequest) -> pb::ClusterRemoveNodesResponse;
    }

    #[tokio::test]
    async fn stub_controller_rejects_rpcs_sinfo_does_not_use() {
        let err = StubController::default()
            .ping(Request::new(()))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unimplemented);
    }

    #[tokio::test]
    async fn state_filter_reaches_the_controller() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        let stub = StubController::default();
        let seen = stub.nodes_req.clone();
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(spur_proto::controller_server(stub))
                .serve_with_incoming(incoming),
        );

        main_with_args(vec![
            "sinfo".into(),
            "-t".into(),
            "idle".into(),
            "--controller".into(),
            format!("http://{addr}"),
        ])
        .await
        .expect("sinfo against the stub controller");

        let req = seen.lock().unwrap().clone().expect("get_nodes was called");
        assert_eq!(req.states, vec![NodeState::NodeIdle as i32]);
    }

    #[test]
    fn test_group_nodes_by_display_state_mixed() {
        let nodes = [
            make_node("n1", NodeState::NodeIdle, "p"),
            make_node("n2", NodeState::NodeIdle, "p"),
            make_node("n3", NodeState::NodeDown, "p"),
            make_node("n4", NodeState::NodeDrain, "p"),
        ];
        let refs: Vec<&NodeInfo> = nodes.iter().collect();
        let groups = group_nodes_by_display_state(&refs);

        assert_eq!(groups.len(), 3);
        // BTreeMap ordering: alphabetical — "down", "drain", "idle"
        assert_eq!(groups[0].0, "down");
        assert_eq!(groups[0].1.len(), 1);
        assert_eq!(groups[1].0, "drain");
        assert_eq!(groups[1].1.len(), 1);
        assert_eq!(groups[2].0, "idle");
        assert_eq!(groups[2].1.len(), 2);
    }

    #[test]
    fn test_group_nodes_by_display_state_all_same() {
        let nodes = [
            make_node("n1", NodeState::NodeIdle, "p"),
            make_node("n2", NodeState::NodeIdle, "p"),
            make_node("n3", NodeState::NodeIdle, "p"),
        ];
        let refs: Vec<&NodeInfo> = nodes.iter().collect();
        let groups = group_nodes_by_display_state(&refs);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "idle");
        assert_eq!(groups[0].1.len(), 3);
    }

    #[test]
    fn test_group_nodes_by_display_state_empty() {
        let groups = group_nodes_by_display_state(&[]);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_render_partition_groups_by_state() {
        let fields = default_fields();
        let partitions = vec![make_partition("batch", true)];
        let nodes = vec![
            make_node("n1", NodeState::NodeIdle, "batch"),
            make_node("n2", NodeState::NodeIdle, "batch"),
            make_node("n3", NodeState::NodeDown, "batch"),
        ];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, false);

        assert_eq!(
            lines.len(),
            2,
            "expected 2 rows (idle + down), got: {lines:?}"
        );

        let idle_line = lines
            .iter()
            .find(|l| l.contains("idle"))
            .expect("no idle row");
        assert!(
            idle_line.contains("2"),
            "idle row should show 2 nodes: {idle_line}"
        );
        assert!(
            idle_line.contains("n[1-2]"),
            "idle row should list a compressed n[1-2]: {idle_line}"
        );
        assert!(
            !idle_line.contains("n3"),
            "idle row should not list n3: {idle_line}"
        );

        let down_line = lines
            .iter()
            .find(|l| l.contains("down"))
            .expect("no down row");
        assert!(
            down_line.contains("1"),
            "down row should show 1 node: {down_line}"
        );
        assert!(
            down_line.contains("n3"),
            "down row should list n3: {down_line}"
        );
    }

    #[test]
    fn test_render_all_idle_single_row() {
        let fields = default_fields();
        let partitions = vec![make_partition("batch", true)];
        let nodes = vec![
            make_node("n1", NodeState::NodeIdle, "batch"),
            make_node("n2", NodeState::NodeIdle, "batch"),
            make_node("n3", NodeState::NodeIdle, "batch"),
        ];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, false);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("idle"));
        assert!(lines[0].contains("3"));
    }

    #[test]
    fn test_render_nodelist_compressed() {
        let fields = default_fields();
        let partitions = vec![make_partition("gpu", true)];
        let nodes = vec![
            make_node("gpu001", NodeState::NodeIdle, "gpu"),
            make_node("gpu002", NodeState::NodeIdle, "gpu"),
            make_node("gpu003", NodeState::NodeIdle, "gpu"),
            make_node("gpu004", NodeState::NodeIdle, "gpu"),
        ];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, false);
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0].contains("gpu[001-004]"),
            "NODELIST should be a compressed hostlist: {}",
            lines[0]
        );
    }

    #[test]
    fn test_render_nodelist_expanded_with_percent_n() {
        let fields = format_engine::parse_format("%P|%n", &format_engine::sinfo_header);
        let partitions = vec![make_partition("gpu", true)];
        let nodes = vec![
            make_node("gpu001", NodeState::NodeIdle, "gpu"),
            make_node("gpu002", NodeState::NodeIdle, "gpu"),
        ];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, false);
        assert_eq!(lines, ["gpu*|gpu001,gpu002"]);
    }

    #[test]
    fn test_render_empty_partition() {
        let fields = format_engine::parse_format("%P|%D|%t|%N", &format_engine::sinfo_header);
        let partitions = vec![make_partition("empty", false)];
        let nodes: Vec<NodeInfo> = vec![];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, false);
        assert_eq!(lines, ["empty|0|n/a|"]);
    }

    #[test]
    fn test_unregistered_configured_nodes_are_not_reported() {
        let fields = format_engine::parse_format("%P|%D|%t|%N", &format_engine::sinfo_header);
        let mut partition = make_partition("gpu", false);
        partition.nodes = "gpu-node1,gpu-node2".into();

        let lines = render_sinfo_output(&fields, std::slice::from_ref(&partition), &[], false);
        assert_eq!(lines, ["gpu|0|n/a|"]);

        let nodes = [make_node("gpu-node1", NodeState::NodeIdle, "gpu")];
        let lines = render_sinfo_output(&fields, &[partition], &nodes, false);
        assert_eq!(lines, ["gpu|1|idle|gpu-node1"]);
    }

    #[test]
    fn test_render_node_oriented_unchanged() {
        let fields =
            format_engine::parse_format("%#N %.6D %#P %.11T", &format_engine::sinfo_header);
        let partitions = vec![make_partition("batch", true)];
        let nodes = vec![
            make_node("n1", NodeState::NodeIdle, "batch"),
            make_node("n2", NodeState::NodeDown, "batch"),
        ];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, true);
        assert_eq!(
            lines.len(),
            2,
            "node-oriented should emit one line per node"
        );
        assert!(lines[0].contains("n1"));
        assert!(lines[0].contains("idle"));
        assert!(lines[1].contains("n2"));
        assert!(lines[1].contains("down"));
    }

    #[test]
    fn node_oriented_output_displays_available_features() {
        let fields = format_engine::parse_format("%n|%f", &format_engine::sinfo_header);
        let partitions = vec![make_partition("gpu", true)];
        let mut node = make_node("gpu-node1", NodeState::NodeIdle, "gpu");
        node.features = vec!["mi350x".into(), "atl".into()];

        let lines = render_sinfo_output(&fields, &partitions, &[node], true);

        assert_eq!(lines, ["gpu-node1|mi350x,atl"]);
    }

    #[test]
    fn node_oriented_output_displays_null_when_features_are_empty() {
        let fields = format_engine::parse_format("%n|%f", &format_engine::sinfo_header);
        let partitions = vec![make_partition("cpu", true)];
        let node = make_node("cpu-node1", NodeState::NodeIdle, "cpu");

        let lines = render_sinfo_output(&fields, &partitions, &[node], true);

        assert_eq!(lines, ["cpu-node1|(null)"]);
    }

    // --- effective_state_str tests ---

    fn make_reserved_node(
        name: &str,
        state: NodeState,
        partition: &str,
        reservation: &str,
    ) -> NodeInfo {
        let mut n = make_node(name, state, partition);
        n.active_reservation = reservation.into();
        n
    }

    #[test]
    fn test_effective_state_mixed_reserved() {
        let node = make_reserved_node("n1", NodeState::NodeMixed, "p", "maint");
        assert_eq!(effective_state_str(&node), "mix");
    }

    #[test]
    fn test_effective_state_idle_with_planned_reservation_shows_plnd() {
        let mut node = make_node("n1", NodeState::NodeIdle, "p");
        node.planned_job_id = 42;
        assert_eq!(effective_state_str(&node), "plnd");
    }

    #[test]
    fn test_effective_state_active_reservation_wins_over_planned() {
        let mut node = make_reserved_node("n1", NodeState::NodeIdle, "p", "resv1");
        node.planned_job_id = 42;
        assert_eq!(effective_state_str(&node), "resv");
    }

    #[test]
    fn test_effective_state_planned_only_applies_when_idle() {
        let mut node = make_node("n1", NodeState::NodeMixed, "p");
        node.planned_job_id = 42;
        assert_eq!(effective_state_str(&node), "mix");
    }

    // --- grouping tests ---

    #[test]
    fn test_group_separates_reserved_idle_from_idle() {
        let nodes = [
            make_node("n1", NodeState::NodeIdle, "p"),
            make_node("n2", NodeState::NodeIdle, "p"),
            make_reserved_node("n3", NodeState::NodeIdle, "p", "maint"),
        ];
        let refs: Vec<&NodeInfo> = nodes.iter().collect();
        let groups = group_nodes_by_display_state(&refs);

        assert_eq!(groups.len(), 2, "expected idle + resv groups: {groups:?}");
        assert_eq!(groups[0].0, "idle");
        assert_eq!(groups[0].1.len(), 2);
        assert_eq!(groups[1].0, "resv");
        assert_eq!(groups[1].1.len(), 1);
    }

    #[test]
    fn test_group_alloc_reserved_stays_with_alloc() {
        let nodes = [
            make_node("n1", NodeState::NodeAllocated, "p"),
            make_reserved_node("n2", NodeState::NodeAllocated, "p", "maint"),
        ];
        let refs: Vec<&NodeInfo> = nodes.iter().collect();
        let groups = group_nodes_by_display_state(&refs);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "alloc");
        assert_eq!(groups[0].1.len(), 2);
    }

    // --- render integration tests ---

    #[test]
    fn test_render_reserved_and_idle_rows() {
        let fields = default_fields();
        let partitions = vec![make_partition("default", true)];
        let nodes = vec![
            make_node("n1", NodeState::NodeIdle, "default"),
            make_node("n2", NodeState::NodeIdle, "default"),
            make_reserved_node("n3", NodeState::NodeIdle, "default", "maint"),
        ];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, false);
        assert_eq!(
            lines.len(),
            2,
            "expected 2 rows (idle + resv), got: {lines:?}"
        );

        let idle_line = lines
            .iter()
            .find(|l| l.contains("idle"))
            .expect("no idle row");
        assert!(
            idle_line.contains("n[1-2]"),
            "idle row should list a compressed n[1-2]: {idle_line}"
        );
        assert!(!idle_line.contains("n3"), "idle row should not list n3");

        let resv_line = lines
            .iter()
            .find(|l| l.contains("resv"))
            .expect("no resv row");
        assert!(resv_line.contains("n3"), "resv row should list n3");
        assert!(!resv_line.contains("n1"), "resv row should not list n1");
    }

    #[test]
    fn test_render_node_oriented_reserved() {
        let fields =
            format_engine::parse_format("%#N %.6D %#P %.11T", &format_engine::sinfo_header);
        let partitions = vec![make_partition("batch", true)];
        let nodes = vec![
            make_node("n1", NodeState::NodeIdle, "batch"),
            make_reserved_node("n2", NodeState::NodeIdle, "batch", "maint"),
        ];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, true);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("n1"));
        assert!(lines[0].contains("idle"));
        assert!(lines[1].contains("n2"));
        assert!(lines[1].contains("resv"));
    }

    #[test]
    fn test_node_oriented_multi_partition_fanout() {
        let fields =
            format_engine::parse_format("%#N %.6D %#P %.11T", &format_engine::sinfo_header);
        let partitions = vec![
            make_partition("gpu", false),
            make_partition("catchall", true),
        ];
        let nodes = vec![NodeInfo {
            name: "n1".into(),
            state: NodeState::NodeIdle as i32,
            partitions: vec!["gpu".into(), "catchall".into()],
            ..Default::default()
        }];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, true);
        assert_eq!(
            lines.len(),
            2,
            "one line per node-partition pair: {lines:?}"
        );
        assert!(lines[0].contains("gpu"));
        assert!(lines[1].contains("catchall"));
        assert!(lines[0].contains("n1"));
        assert!(lines[1].contains("n1"));
    }

    #[test]
    fn test_partition_oriented_multi_partition_node() {
        let fields = default_fields();
        let partitions = vec![make_partition("gpu", false), make_partition("batch", true)];
        let nodes = vec![NodeInfo {
            name: "n1".into(),
            state: NodeState::NodeIdle as i32,
            partitions: vec!["gpu".into(), "batch".into()],
            ..Default::default()
        }];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, false);
        assert_eq!(
            lines.len(),
            2,
            "node appears under each partition: {lines:?}"
        );
        assert!(lines[0].contains("gpu"));
        assert!(lines[1].contains("batch"));
    }

    #[test]
    fn test_node_oriented_empty_partitions_fallback() {
        let fields =
            format_engine::parse_format("%#N %.6D %#P %.11T", &format_engine::sinfo_header);
        let partitions = vec![make_partition("batch", true)];
        let nodes = vec![NodeInfo {
            name: "orphan".into(),
            state: NodeState::NodeIdle as i32,
            partitions: vec![],
            ..Default::default()
        }];

        let lines = render_sinfo_output(&fields, &partitions, &nodes, true);
        assert_eq!(lines.len(), 1, "orphan node still gets one row");
        assert!(lines[0].contains("orphan"));
    }
}
