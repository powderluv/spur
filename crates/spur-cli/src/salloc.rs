use anyhow::{Context, Result};
use clap::Parser;
use spur_proto::proto::slurm_controller_client::SlurmControllerClient;
use spur_proto::proto::{CancelJobRequest, GetJobRequest, JobSpec, SubmitJobRequest};
use std::collections::HashMap;

/// Allocate resources for an interactive job.
#[derive(Parser, Debug)]
#[command(name = "salloc", about = "Allocate resources for an interactive job")]
pub struct SallocArgs {
    /// Job name
    #[arg(short = 'J', long)]
    pub job_name: Option<String>,

    /// Partition
    #[arg(short = 'p', long)]
    pub partition: Option<String>,

    /// Account
    #[arg(short = 'A', long)]
    pub account: Option<String>,

    /// Number of nodes
    #[arg(short = 'N', long, default_value = "1")]
    pub nodes: u32,

    /// Number of tasks
    #[arg(short = 'n', long, default_value = "1")]
    pub ntasks: u32,

    /// CPUs per task
    #[arg(short = 'c', long, default_value = "1")]
    pub cpus_per_task: u32,

    /// Memory per node (e.g., "4G", "4096M")
    #[arg(long)]
    pub mem: Option<String>,

    /// Time limit
    #[arg(short = 't', long, default_value = "1:00:00")]
    pub time: String,

    /// GRES
    #[arg(long)]
    pub gres: Vec<String>,

    /// GPUs
    #[arg(short = 'G', long)]
    pub gpus: Option<String>,

    /// Required node features (e.g., "mi300x,nvlink")
    #[arg(short = 'C', long)]
    pub constraint: Option<String>,

    /// Exclusive node allocation
    #[arg(long)]
    pub exclusive: bool,

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
    let args = SallocArgs::try_parse_from(&args)?;

    let name = args.job_name.unwrap_or_else(|| "interactive".into());
    let mut gres = args.gres;
    if let Some(gpus) = &args.gpus {
        gres.push(format!("gpu:{}", gpus));
    }

    let time_limit =
        spur_core::config::parse_time_minutes(&args.time).map(|mins| prost_types::Duration {
            seconds: mins as i64 * 60,
            nanos: 0,
        });

    let memory_mb = args
        .mem
        .as_ref()
        .map(|m| parse_memory_mb(m))
        .transpose()?
        .unwrap_or(0);

    let nodes = args.nodes;
    let controller = args.controller.clone();
    let exclusive = args.exclusive;
    let constraint = args.constraint;
    let partition = args.partition;
    let account = args.account;
    let ntasks = args.ntasks;
    let cpus_per_task = args.cpus_per_task;

    let mut client = SlurmControllerClient::connect(controller)
        .await
        .context("failed to connect to spurctld")?;

    // Submit interactive allocation (sleep infinity holds the allocation)
    let job_spec = JobSpec {
        name,
        partition: partition.unwrap_or_default(),
        account: account.unwrap_or_default(),
        user: whoami::username().unwrap_or_else(|_| "unknown".into()),
        uid: nix::unistd::getuid().as_raw(),
        gid: nix::unistd::getgid().as_raw(),
        num_nodes: nodes,
        num_tasks: ntasks,
        cpus_per_task,
        memory_per_node_mb: memory_mb,
        gres,
        script: "#!/bin/bash\nsleep infinity\n".into(),
        time_limit,
        exclusive,
        constraint: constraint.unwrap_or_default(),
        interactive: true,
        environment: HashMap::new(),
        ..Default::default()
    };

    let response = client
        .submit_job(SubmitJobRequest {
            spec: Some(job_spec),
        })
        .await
        .context("job submission failed")?;

    let job_id = response.into_inner().job_id;
    let user = whoami::username().unwrap_or_else(|_| "unknown".into());
    eprintln!("salloc: Pending job allocation {}...", job_id);

    // Set up Ctrl+C handler to cancel the job on interrupt
    let cancel_client = client.clone();
    let cancel_user = user.clone();
    tokio::spawn(async move {
        let mut cancel_client = cancel_client;
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("\nsalloc: cancelling job {}...", job_id);
            let _ = cancel_client
                .cancel_job(CancelJobRequest {
                    job_id,
                    signal: 2, // SIGINT
                    user: cancel_user,
                })
                .await;
            std::process::exit(130); // Standard SIGINT exit code
        }
    });

    // Wait for the job to start running
    #[allow(unused_assignments)]
    let mut nodelist = String::new();
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        match client.get_job(GetJobRequest { job_id }).await {
            Ok(resp) => {
                let job = resp.into_inner();
                match job.state {
                    1 => {
                        // RUNNING
                        nodelist = job.nodelist.clone();
                        break;
                    }
                    3..=7 => {
                        // Terminal
                        eprintln!("salloc: job {} ended before allocation was granted", job_id);
                        std::process::exit(1);
                    }
                    _ => {} // Still pending
                }
            }
            Err(e) => {
                eprintln!("salloc: warning: {}", e.message());
            }
        }
    }

    eprintln!("salloc: Nodes {} are ready for job {}", nodelist, job_id);
    eprintln!("salloc: Granted job allocation {}", job_id);

    // Spawn interactive shell with allocation env vars
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".into());
    let status = tokio::process::Command::new(&shell)
        .env("SPUR_JOB_ID", job_id.to_string())
        .env("SPUR_NODELIST", &nodelist)
        .env("SPUR_NNODES", nodes.to_string())
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .await
        .context("failed to spawn shell")?;

    // Shell exited — cancel allocation
    eprintln!("salloc: Relinquishing job allocation {}", job_id);
    let _ = client
        .cancel_job(CancelJobRequest {
            job_id,
            signal: 0,
            user,
        })
        .await;

    std::process::exit(status.code().unwrap_or(0));
}

fn parse_memory_mb(s: &str) -> Result<u64> {
    let s = s.trim();
    if let Some(gb) = s.strip_suffix('G').or_else(|| s.strip_suffix('g')) {
        let val: f64 = gb.parse().context("invalid memory value")?;
        Ok((val * 1024.0) as u64)
    } else if let Some(mb) = s.strip_suffix('M').or_else(|| s.strip_suffix('m')) {
        Ok(mb.parse().context("invalid memory value")?)
    } else {
        Ok(s.parse().context("invalid memory value")?)
    }
}
