use anyhow::{Context, Result};
use clap::Parser;
use spur_proto::proto::slurm_controller_client::SlurmControllerClient;
use spur_proto::proto::{GetNodesRequest, GetPartitionsRequest};

use crate::format_engine;

/// View information about nodes and partitions.
#[derive(Parser, Debug)]
#[command(name = "sinfo", about = "View cluster information")]
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

    /// Controller address
    #[arg(
        long,
        env = "SPUR_CONTROLLER_ADDR",
        default_value = "http://localhost:6817"
    )]
    pub controller: String,
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

    let mut client = SlurmControllerClient::connect(args.controller)
        .await
        .context("failed to connect to spurctld")?;

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
        .get_nodes(GetNodesRequest {
            states: Vec::new(),
            partition: args.partition.unwrap_or_default(),
            nodelist: args.nodes.unwrap_or_default(),
        })
        .await
        .context("failed to get nodes")?;

    let nodes = nodes_resp.into_inner().nodes;

    // Print header
    if !args.noheader {
        println!("{}", format_engine::format_header(&fields));
    }

    if args.node_oriented {
        // One line per node
        for node in &nodes {
            let row = format_engine::format_row(&fields, &|spec| {
                resolve_node_field(node, &partitions, spec)
            });
            println!("{}", row);
        }
    } else {
        // One line per partition (summarized)
        for part in &partitions {
            // Collect nodes belonging to this partition
            let part_nodes: Vec<_> = nodes.iter().filter(|n| n.partition == part.name).collect();

            let row = format_engine::format_row(&fields, &|spec| {
                resolve_partition_field(part, &part_nodes, spec)
            });
            println!("{}", row);
        }
    }

    Ok(())
}

fn resolve_node_field(
    node: &spur_proto::proto::NodeInfo,
    _partitions: &[spur_proto::proto::PartitionInfo],
    spec: char,
) -> String {
    match spec {
        'N' | 'n' => node.name.clone(),
        'P' | 'R' => node.partition.clone(),
        't' | 'T' => node_state_str(node.state),
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
        'D' => {
            // part.total_nodes may be 0 (not populated by server),
            // so fall back to the actual node count from the query.
            if nodes.is_empty() && part.total_nodes > 0 {
                part.total_nodes.to_string()
            } else {
                nodes.len().to_string()
            }
        }
        't' | 'T' => {
            // Summarize node states
            if nodes.is_empty() {
                // No node details available; default to idle
                "idle".into()
            } else {
                // Show most common state
                let mut counts = std::collections::HashMap::new();
                for n in nodes {
                    *counts.entry(n.state).or_insert(0u32) += 1;
                }
                let (most_common, _) = counts.iter().max_by_key(|(_, &c)| c).unwrap();
                node_state_str(*most_common)
            }
        }
        'N' => part.nodes.clone(),
        'c' => part.total_cpus.to_string(),
        _ => "?".into(),
    }
}

fn node_state_str(state: i32) -> String {
    match state {
        0 => "idle",
        1 => "alloc",
        2 => "mix",
        3 => "down",
        4 => "drain",
        5 => "drng",
        6 => "err",
        _ => "unk",
    }
    .into()
}
