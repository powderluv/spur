use anyhow::{Context, Result};
use clap::Parser;
use spur_proto::proto::slurm_controller_client::SlurmControllerClient;
use spur_proto::proto::{GetJobRequest, JobSpec, SubmitJobRequest};
use std::collections::HashMap;

/// Run a parallel job (interactive or allocation-based).
#[derive(Parser, Debug)]
#[command(name = "srun", about = "Run a parallel job")]
pub struct SrunArgs {
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
    #[arg(short = 't', long)]
    pub time: Option<String>,

    /// GRES
    #[arg(long)]
    pub gres: Vec<String>,

    /// GPUs
    #[arg(short = 'G', long)]
    pub gpus: Option<String>,

    /// Working directory
    #[arg(short = 'D', long)]
    pub chdir: Option<String>,

    /// CPU binding (none, cores, threads, sockets, ldoms, rank, map_cpu, mask_cpu)
    #[arg(long)]
    pub cpu_bind: Option<String>,

    /// GPU binding (closest, map_gpu, mask_gpu, none)
    #[arg(long)]
    pub gpu_bind: Option<String>,

    /// Job step label output
    #[arg(short = 'l', long)]
    pub label: bool,

    // Container
    /// Container image (OCI ref or squashfs path)
    #[arg(long)]
    pub container_image: Option<String>,

    /// Container bind mounts ("/src:/dst:ro")
    #[arg(long)]
    pub container_mounts: Vec<String>,

    /// Working directory inside the container
    #[arg(long)]
    pub container_workdir: Option<String>,

    /// Mount user home directory in container
    #[arg(long)]
    pub container_mount_home: bool,

    /// Set environment variable inside container (KEY=VAL)
    #[arg(long)]
    pub container_env: Vec<String>,

    /// Remap user to root inside container
    #[arg(long)]
    pub container_remap_root: bool,

    /// Controller address
    #[arg(
        long,
        env = "SPUR_CONTROLLER_ADDR",
        default_value = "http://localhost:6817"
    )]
    pub controller: String,

    /// Command and arguments
    #[arg(trailing_var_arg = true)]
    pub command: Vec<String>,
}

pub async fn main() -> Result<()> {
    let args = SrunArgs::try_parse_from(std::env::args())?;

    if args.command.is_empty() {
        eprintln!("srun: no command specified");
        std::process::exit(1);
    }

    let name = args.job_name.unwrap_or_else(|| args.command[0].clone());

    let work_dir = args
        .chdir
        .unwrap_or_else(|| std::env::current_dir().unwrap().to_string_lossy().into());

    // Build a wrapper script from the command
    let cmd_line = args.command.join(" ");
    let script = format!("#!/bin/bash\n{}\n", cmd_line);

    // Build GRES list
    let mut gres = args.gres;
    if let Some(gpus) = &args.gpus {
        gres.push(format!("gpu:{}", gpus));
    }

    let time_limit = args
        .time
        .as_ref()
        .and_then(|t| spur_core::config::parse_time_minutes(t))
        .map(|mins| prost_types::Duration {
            seconds: mins as i64 * 60,
            nanos: 0,
        });

    // Build environment — pass CPU/GPU binding via env vars
    let mut environment: HashMap<String, String> = std::env::vars().collect();
    if let Some(ref cpu_bind) = args.cpu_bind {
        environment.insert("SPUR_CPU_BIND".into(), cpu_bind.clone());
    }
    if let Some(ref gpu_bind) = args.gpu_bind {
        environment.insert("SPUR_GPU_BIND".into(), gpu_bind.clone());
    }

    let memory_mb = args
        .mem
        .as_ref()
        .map(|m| parse_memory_mb(m))
        .transpose()?
        .unwrap_or(0);

    // Submit as a batch job
    let mut client = SlurmControllerClient::connect(args.controller.clone())
        .await
        .context("failed to connect to spurctld")?;

    let job_spec = JobSpec {
        name,
        partition: args.partition.unwrap_or_default(),
        account: args.account.unwrap_or_default(),
        user: whoami::username().unwrap_or_else(|_| "unknown".into()),
        uid: nix::unistd::getuid().as_raw(),
        gid: nix::unistd::getgid().as_raw(),
        num_nodes: args.nodes,
        num_tasks: args.ntasks,
        cpus_per_task: args.cpus_per_task,
        memory_per_node_mb: memory_mb,
        gres,
        script,
        work_dir: work_dir.clone(),
        environment,
        time_limit,
        container_image: args.container_image.unwrap_or_default(),
        container_mounts: args.container_mounts,
        container_workdir: args.container_workdir.unwrap_or_default(),
        container_mount_home: args.container_mount_home,
        container_env: args
            .container_env
            .iter()
            .filter_map(|s| {
                s.split_once('=')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
            })
            .collect(),
        container_remap_root: args.container_remap_root,
        ..Default::default()
    };

    let response = client
        .submit_job(SubmitJobRequest {
            spec: Some(job_spec),
        })
        .await
        .context("job submission failed")?;

    let job_id = response.into_inner().job_id;
    eprintln!("srun: job {} submitted, waiting for completion...", job_id);

    // Poll for completion
    let mut poll_interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
    let mut was_running = false;

    loop {
        poll_interval.tick().await;

        let resp = client.get_job(GetJobRequest { job_id }).await;

        match resp {
            Ok(resp) => {
                let job = resp.into_inner();
                let state = job.state;

                if !was_running && state == 1 {
                    // Just started running
                    was_running = true;
                    if !job.nodelist.is_empty() {
                        eprintln!("srun: job {} running on {}", job_id, job.nodelist);
                    }
                }

                // Check terminal states
                match state {
                    3 => {
                        // COMPLETED
                        // Try to read output
                        print_job_output(&work_dir, job_id).await;
                        std::process::exit(job.exit_code);
                    }
                    4 => {
                        // FAILED
                        print_job_output(&work_dir, job_id).await;
                        eprintln!(
                            "srun: job {} failed with exit code {}",
                            job_id, job.exit_code
                        );
                        std::process::exit(job.exit_code.max(1));
                    }
                    5 => {
                        // CANCELLED
                        eprintln!("srun: job {} cancelled", job_id);
                        std::process::exit(1);
                    }
                    6 => {
                        // TIMEOUT
                        eprintln!("srun: job {} timed out", job_id);
                        std::process::exit(1);
                    }
                    7 => {
                        // NODE_FAIL
                        eprintln!("srun: job {} failed (node failure)", job_id);
                        std::process::exit(1);
                    }
                    _ => {} // Still pending or running
                }
            }
            Err(e) => {
                eprintln!("srun: warning: failed to get job status: {}", e.message());
            }
        }
    }
}

/// Print job output file to stdout (best-effort).
async fn print_job_output(work_dir: &str, job_id: u32) {
    let path = format!("{}/spur-{}.out", work_dir, job_id);
    if let Ok(content) = tokio::fs::read_to_string(&path).await {
        print!("{}", content);
    }
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
