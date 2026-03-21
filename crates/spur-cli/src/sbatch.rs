use anyhow::{bail, Context, Result};
use clap::Parser;
use spur_proto::proto::slurm_controller_client::SlurmControllerClient;
use spur_proto::proto::{JobSpec, SubmitJobRequest};
use std::collections::HashMap;

/// Submit a batch job script.
#[derive(Parser, Debug)]
#[command(name = "sbatch", about = "Submit a batch job script")]
pub struct SbatchArgs {
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

    /// Tasks per node
    #[arg(long)]
    pub ntasks_per_node: Option<u32>,

    /// CPUs per task
    #[arg(short = 'c', long, default_value = "1")]
    pub cpus_per_task: u32,

    /// Memory per node (e.g., "4G", "4096M", "4096")
    #[arg(long)]
    pub mem: Option<String>,

    /// Memory per CPU
    #[arg(long)]
    pub mem_per_cpu: Option<String>,

    /// Generic resources (e.g., "gpu:4", "gpu:mi300x:8")
    #[arg(long)]
    pub gres: Vec<String>,

    /// GPUs (shorthand, e.g., "4" or "mi300x:4")
    #[arg(short = 'G', long)]
    pub gpus: Option<String>,

    /// GPUs per node
    #[arg(long)]
    pub gpus_per_node: Option<String>,

    /// Time limit (e.g., "4:00:00", "1-00:00:00")
    #[arg(short = 't', long)]
    pub time: Option<String>,

    /// Minimum time limit
    #[arg(long)]
    pub time_min: Option<String>,

    /// Working directory
    #[arg(short = 'D', long)]
    pub chdir: Option<String>,

    /// Stdout file
    #[arg(short = 'o', long)]
    pub output: Option<String>,

    /// Stderr file
    #[arg(short = 'e', long)]
    pub error: Option<String>,

    /// QoS
    #[arg(short = 'q', long)]
    pub qos: Option<String>,

    /// Job dependency (e.g., "afterok:123")
    #[arg(short = 'd', long)]
    pub dependency: Option<String>,

    /// Node list
    #[arg(short = 'w', long)]
    pub nodelist: Option<String>,

    /// Exclude nodes
    #[arg(short = 'x', long)]
    pub exclude: Option<String>,

    /// Required node features (e.g., "mi300x,nvlink")
    #[arg(short = 'C', long)]
    pub constraint: Option<String>,

    /// Job array (e.g., "0-99%10")
    #[arg(short = 'a', long)]
    pub array: Option<String>,

    /// Allow requeue
    #[arg(long)]
    pub requeue: bool,

    /// Exclusive node access
    #[arg(long)]
    pub exclusive: bool,

    /// Hold job
    #[arg(short = 'H', long)]
    pub hold: bool,

    /// Comment
    #[arg(long)]
    pub comment: Option<String>,

    /// Export environment variables
    #[arg(long, default_value = "ALL")]
    pub export: String,

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

    /// Named container (persists across jobs)
    #[arg(long)]
    pub container_name: Option<String>,

    /// Read-only container rootfs
    #[arg(long)]
    pub container_readonly: bool,

    /// Mount user home directory in container
    #[arg(long)]
    pub container_mount_home: bool,

    /// Set environment variable inside container (KEY=VAL)
    #[arg(long)]
    pub container_env: Vec<String>,

    /// Override container entrypoint
    #[arg(long)]
    pub container_entrypoint: Option<String>,

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

    /// The batch script file
    pub script: Option<String>,
}

/// Parse #SBATCH directives from a script, returning them as argv-style strings.
pub fn parse_sbatch_directives(script: &str) -> Vec<String> {
    let mut args = Vec::new();
    for line in script.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("#SBATCH") {
            let rest = rest.trim();
            if !rest.is_empty() {
                // Split on whitespace, preserving quoted strings
                args.extend(shell_split(rest));
            }
        }
        // Also parse #PBS for migration
        if let Some(rest) = trimmed.strip_prefix("#PBS") {
            let rest = rest.trim();
            if !rest.is_empty() {
                if let Some(converted) = convert_pbs_to_sbatch(rest) {
                    args.extend(shell_split(&converted));
                }
            }
        }
        // Stop at first non-comment, non-blank, non-shebang line
        if !trimmed.is_empty() && !trimmed.starts_with('#') && !trimmed.starts_with("#!/") {
            break;
        }
    }
    args
}

/// Basic shell word splitting (handles simple quoting).
fn shell_split(s: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;

    for c in s.chars() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            ' ' | '\t' if !in_single && !in_double => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// Convert a PBS directive to sbatch equivalent (best-effort).
fn convert_pbs_to_sbatch(pbs_arg: &str) -> Option<String> {
    let parts: Vec<&str> = pbs_arg.splitn(2, ' ').collect();
    let flag = parts[0].trim_start_matches('-');
    let value = parts.get(1).map(|s| s.trim());

    match flag {
        "N" => value.map(|v| format!("--job-name={}", v)),
        "q" => value.map(|v| format!("--partition={}", v)),
        "l" => {
            // PBS resource specs like "walltime=4:00:00" or "nodes=2:ppn=8"
            value.and_then(|v| convert_pbs_resource(v))
        }
        "o" => value.map(|v| format!("--output={}", v)),
        "e" => value.map(|v| format!("--error={}", v)),
        "A" => value.map(|v| format!("--account={}", v)),
        _ => None,
    }
}

fn convert_pbs_resource(spec: &str) -> Option<String> {
    for part in spec.split(',') {
        let kv: Vec<&str> = part.splitn(2, '=').collect();
        if kv.len() == 2 {
            match kv[0] {
                "walltime" => return Some(format!("--time={}", kv[1])),
                "nodes" => {
                    // "nodes=2:ppn=8" → "--nodes=2 --ntasks-per-node=8"
                    let node_parts: Vec<&str> = kv[1].split(':').collect();
                    let mut result = format!("--nodes={}", node_parts[0]);
                    for np in &node_parts[1..] {
                        if let Some(ppn) = np.strip_prefix("ppn=") {
                            result.push_str(&format!(" --ntasks-per-node={}", ppn));
                        }
                    }
                    return Some(result);
                }
                "mem" => return Some(format!("--mem={}", kv[1])),
                _ => {}
            }
        }
    }
    None
}

/// Parse memory string (e.g., "4G", "4096M", "4096") into MB.
fn parse_memory_mb(s: &str) -> Result<u64> {
    let s = s.trim();
    if let Some(gb) = s.strip_suffix('G').or_else(|| s.strip_suffix('g')) {
        let val: f64 = gb.parse().context("invalid memory value")?;
        Ok((val * 1024.0) as u64)
    } else if let Some(mb) = s.strip_suffix('M').or_else(|| s.strip_suffix('m')) {
        Ok(mb.parse().context("invalid memory value")?)
    } else if let Some(kb) = s.strip_suffix('K').or_else(|| s.strip_suffix('k')) {
        let val: u64 = kb.parse().context("invalid memory value")?;
        Ok(val / 1024)
    } else {
        // Default: MB
        Ok(s.parse().context("invalid memory value")?)
    }
}

pub async fn main() -> Result<()> {
    main_with_args(std::env::args().collect()).await
}

pub async fn main_with_args(cli_args: Vec<String>) -> Result<()> {
    // Build argv: merge #SBATCH directives (from script) with CLI args.
    // CLI args take precedence (they come after in the merged argv).

    // If script is provided, parse directives from it
    let script_content = if let Some(script_path) = cli_args.last() {
        if !script_path.starts_with('-') && script_path != "sbatch" {
            match std::fs::read_to_string(script_path) {
                Ok(content) => Some(content),
                Err(_) => None,
            }
        } else {
            None
        }
    } else {
        None
    };

    let mut merged_args = vec!["sbatch".to_string()];
    if let Some(ref content) = script_content {
        merged_args.extend(parse_sbatch_directives(content));
    }
    // CLI args override (skip argv[0])
    merged_args.extend(cli_args.into_iter().skip(1));

    let args = SbatchArgs::try_parse_from(&merged_args)?;

    // Build the job spec
    let script = match &args.script {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("failed to read script: {}", path))?,
        None => {
            if atty::is(atty::Stream::Stdin) {
                bail!("sbatch: no script file specified");
            }
            // Read from stdin
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            buf
        }
    };

    let work_dir = args
        .chdir
        .unwrap_or_else(|| std::env::current_dir().unwrap().to_string_lossy().into());

    let name = args
        .job_name
        .unwrap_or_else(|| args.script.as_deref().unwrap_or("sbatch").to_string());

    // Build GRES list
    let mut gres = args.gres;
    if let Some(gpus) = &args.gpus {
        gres.push(format!("gpu:{}", gpus));
    }
    if let Some(gpn) = &args.gpus_per_node {
        gres.push(format!("gpu:{}", gpn));
    }

    // Parse time limit — use parse_time_seconds so that short values like
    // "0:00:10" (10 seconds) are stored with full second precision instead of
    // being rounded up to the nearest minute.
    let time_limit = args
        .time
        .as_ref()
        .and_then(|t| spur_core::config::parse_time_seconds(t))
        .map(|secs| prost_types::Duration {
            seconds: secs as i64,
            nanos: 0,
        });

    // Parse memory
    let memory_per_node = args.mem.as_ref().map(|m| parse_memory_mb(m)).transpose()?;
    let memory_per_cpu = args
        .mem_per_cpu
        .as_ref()
        .map(|m| parse_memory_mb(m))
        .transpose()?;

    // Build environment
    let environment: HashMap<String, String> = if args.export == "ALL" {
        std::env::vars().collect()
    } else if args.export == "NONE" {
        HashMap::new()
    } else {
        std::env::vars()
            .filter(|(k, _)| args.export.split(',').any(|e| e == k))
            .collect()
    };

    // Parse dependencies
    let dependencies: Vec<String> = args
        .dependency
        .map(|d| d.split(',').map(String::from).collect())
        .unwrap_or_default();

    let job_spec = JobSpec {
        name,
        partition: args.partition.unwrap_or_default(),
        account: args.account.unwrap_or_default(),
        user: whoami::username().unwrap_or_else(|_| "unknown".into()),
        uid: nix::unistd::getuid().as_raw(),
        gid: nix::unistd::getgid().as_raw(),
        num_nodes: args.nodes,
        num_tasks: args.ntasks,
        tasks_per_node: args.ntasks_per_node.unwrap_or(0),
        cpus_per_task: args.cpus_per_task,
        memory_per_node_mb: memory_per_node.unwrap_or(0),
        memory_per_cpu_mb: memory_per_cpu.unwrap_or(0),
        gres,
        script,
        argv: Vec::new(),
        work_dir,
        stdout_path: args.output.unwrap_or_default(),
        stderr_path: args.error.unwrap_or_default(),
        environment,
        time_limit,
        time_min: None,
        qos: args.qos.unwrap_or_default(),
        priority: 0,
        reservation: String::new(),
        dependency: dependencies,
        nodelist: args.nodelist.unwrap_or_default(),
        exclude: args.exclude.unwrap_or_default(),
        constraint: args.constraint.unwrap_or_default(),
        array_spec: args.array.unwrap_or_default(),
        requeue: args.requeue,
        exclusive: args.exclusive,
        hold: args.hold,
        comment: args.comment.unwrap_or_default(),
        wckey: String::new(),
        container_image: args.container_image.unwrap_or_default(),
        container_mounts: args.container_mounts,
        container_workdir: args.container_workdir.unwrap_or_default(),
        container_name: args.container_name.unwrap_or_default(),
        container_readonly: args.container_readonly,
        container_mount_home: args.container_mount_home,
        container_env: args
            .container_env
            .iter()
            .filter_map(|s| {
                s.split_once('=')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
            })
            .collect(),
        container_entrypoint: args.container_entrypoint.unwrap_or_default(),
        container_remap_root: args.container_remap_root,
    };

    // Submit to controller
    let mut client = SlurmControllerClient::connect(args.controller)
        .await
        .context("failed to connect to spurctld")?;

    let response = client
        .submit_job(SubmitJobRequest {
            spec: Some(job_spec),
        })
        .await
        .context("job submission failed")?;

    let job_id = response.into_inner().job_id;
    println!("Submitted batch job {}", job_id);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sbatch_directives() {
        let script = r#"#!/bin/bash
#SBATCH --job-name=test
#SBATCH -N 4
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:mi300x:8

echo "hello world"
"#;
        let args = parse_sbatch_directives(script);
        assert_eq!(
            args,
            vec![
                "--job-name=test",
                "-N",
                "4",
                "--time=4:00:00",
                "--gres=gpu:mi300x:8"
            ]
        );
    }

    #[test]
    fn test_parse_memory() {
        assert_eq!(parse_memory_mb("4G").unwrap(), 4096);
        assert_eq!(parse_memory_mb("4096M").unwrap(), 4096);
        assert_eq!(parse_memory_mb("4096").unwrap(), 4096);
        assert_eq!(parse_memory_mb("1024K").unwrap(), 1);
    }

    #[test]
    fn test_pbs_conversion() {
        assert_eq!(
            convert_pbs_to_sbatch("-N myname"),
            Some("--job-name=myname".into())
        );
        assert_eq!(
            convert_pbs_to_sbatch("-l walltime=4:00:00"),
            Some("--time=4:00:00".into())
        );
    }
}
