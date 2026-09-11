// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::env_defaults::{
    apply_csv, apply_flag, apply_num, apply_num_opt, apply_str, was_cli_set,
};
use anyhow::{Context, Result};
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser};
use spur_core::config::HooksConfig;
use spur_proto::proto::slurm_controller_client::SlurmControllerClient;
use spur_proto::proto::{
    CancelJobRequest, CompleteJobRequest, CompleteJobStepRequest, ContainerSpec,
    CreateJobStepRequest, GetJobRequest, GetNodeRequest, JobSpec, JobState, RunStepRequest,
    StreamJobOutputChunk, StreamJobOutputRequest, SubmitJobRequest,
};
use std::collections::HashMap;
use std::io::Write as _;

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

    /// QoS
    #[arg(short = 'q', long)]
    pub qos: Option<String>,

    /// Number of nodes
    #[arg(short = 'N', long, default_value = "1")]
    pub nodes: u32,

    /// Number of tasks (default: one per node)
    #[arg(short = 'n', long)]
    pub ntasks: Option<u32>,

    /// Tasks per node
    #[arg(long, overrides_with = "ntasks_per_node")]
    pub ntasks_per_node: Option<u32>,

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
    // Slurm semantics: comma-lists expand to multiple GRES; a repeated --gres
    // replaces rather than accumulates (last wins).
    #[arg(long, value_delimiter = ',', overrides_with = "gres")]
    pub gres: Vec<String>,

    /// Licenses (e.g., "fluent:5", "matlab:1")
    #[arg(short = 'L', long)]
    pub licenses: Vec<String>,

    /// GPUs total across the job (e.g., "4" or "mi300x:4")
    #[arg(short = 'G', long)]
    pub gpus: Option<String>,

    /// GPUs per node
    #[arg(long)]
    pub gpus_per_node: Option<String>,

    /// GPUs per task
    #[arg(long)]
    pub gpus_per_task: Option<String>,

    /// Working directory
    #[arg(short = 'D', long)]
    pub chdir: Option<String>,

    /// CPU binding (none, rank, map_cpu, mask_cpu; topology modes cores/threads/sockets/ldoms are not applied in step mode)
    #[arg(long)]
    pub cpu_bind: Option<String>,

    /// GPU binding (closest, map_gpu, mask_gpu, none)
    #[arg(long)]
    pub gpu_bind: Option<String>,

    /// Required node features (e.g., "mi300x,nvlink")
    #[arg(short = 'C', long)]
    pub constraint: Option<String>,

    /// Node list
    #[arg(
        short = 'w',
        long,
        overrides_with_all = ["nodelist", "nodefile"]
    )]
    pub nodelist: Option<String>,

    /// Read the node list from a file
    #[arg(
        short = 'F',
        long,
        overrides_with_all = ["nodelist", "nodefile"]
    )]
    pub nodefile: Option<String>,

    /// Exclude nodes
    #[arg(short = 'x', long)]
    pub exclude: Option<String>,

    /// Target a named reservation
    #[arg(long)]
    pub reservation: Option<String>,

    /// Exclusive node allocation
    #[arg(long)]
    pub exclusive: bool,

    /// MPI type (none, pmix). Inside an allocation, omit to inherit the job setting.
    #[arg(long)]
    pub mpi: Option<String>,

    /// Stdout file path (supports %j for job ID)
    #[arg(short = 'o', long)]
    pub output: Option<String>,

    /// Stderr file path (supports %j for job ID)
    #[arg(short = 'e', long)]
    pub error: Option<String>,

    /// Stdin file path
    #[arg(short = 'i', long)]
    pub input: Option<String>,

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

    /// Named container (persists across steps in an allocation)
    #[arg(long)]
    pub container_name: Option<String>,

    /// Read-only container rootfs (not yet implemented; rejected at submission)
    #[arg(long)]
    pub container_readonly: bool,

    /// Mount user home directory in container
    #[arg(long)]
    pub container_mount_home: bool,

    /// Set environment variable inside container (KEY=VAL)
    #[arg(long)]
    pub container_env: Vec<String>,

    /// Shell command run inside the container before the job script
    #[arg(long)]
    pub container_entrypoint: Option<String>,

    /// Remap user to root inside container
    #[arg(long)]
    pub container_remap_root: bool,

    /// Prolog script to run locally before step dispatch
    #[arg(long)]
    pub prolog: Option<String>,

    /// Epilog script to run locally after step completion
    #[arg(long)]
    pub epilog: Option<String>,

    /// Allocate a pseudo-terminal for the job
    #[arg(long)]
    pub pty: bool,

    /// Attach to a running job (for --overlap exec-into-job)
    #[arg(long)]
    pub jobid: Option<u32>,

    /// Share resources with the running job (requires --jobid)
    #[arg(long)]
    pub overlap: bool,

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
    main_with_args(std::env::args().collect()).await
}

pub async fn main_with_args(args: Vec<String>) -> Result<()> {
    let submit_line = crate::submitline::render(&args);
    let matches = SrunArgs::command().try_get_matches_from(&args)?;
    let mut args = SrunArgs::from_arg_matches(&matches)?;

    if args.jobid.is_some() && !args.overlap {
        anyhow::bail!("--jobid requires --overlap");
    }

    if args.container_readonly {
        anyhow::bail!(
            "--container-readonly is not yet implemented; the container root would be \
             writable despite the flag. Omit it rather than rely on a read-only rootfs."
        );
    }

    resolve_srun_env(&matches, &mut args)?;
    args.nodelist = crate::nodelist::resolve(args.nodelist.take(), args.nodefile.take())?;

    // --jobid --overlap: exec into a running job (interactive PTY session)
    if let Some(job_id) = args.jobid {
        let warnings = step_unsupported_warnings(&args)
            .into_iter()
            .chain(pty_step_unsupported_warnings(&args, &matches));
        for warning in warnings {
            eprintln!("{warning}");
        }
        let node = first_node(args.nodelist.as_deref().unwrap_or_default());
        let channel = crate::authclient::connect(&args.controller)
            .await
            .context("cannot connect to controller")?;
        let mut ctrl = spur_proto::controller_client(channel);
        let job = ctrl
            .get_job(GetJobRequest { job_id })
            .await
            .context("failed to get job info")?
            .into_inner();
        let known_owner = (!job.user.is_empty()).then_some(job.user.as_str());
        let user = crate::interactive::job_caller_user(&mut ctrl, job_id, known_owner).await?;
        let exit_code = run_interactive_pty(
            &mut ctrl,
            job_id,
            args.command.clone(),
            node,
            &user,
            container_spec_from_srun_args(&args),
        )
        .await?;
        std::process::exit(exit_code);
    }

    if args.mpi.as_deref() == Some("list") {
        let mpi_cfg = load_mpi_config();
        for line in spur_core::mpi::mpi_list_lines(&mpi_cfg.plugin_dir) {
            println!("{line}");
        }
        return Ok(());
    }

    let step_mode = std::env::var("SPUR_JOB_ID")
        .ok()
        .and_then(|id| id.parse::<u32>().ok())
        .is_some();
    let resolved_mpi = resolve_srun_mpi(&args, &matches, step_mode)?;

    if args.command.is_empty() {
        eprintln!("srun: no command specified");
        std::process::exit(1);
    }

    // Resolve hooks: CLI flags override config file
    let mut hooks = load_hooks_config();
    if let Some(ref prolog) = args.prolog {
        hooks.srun_prolog = Some(prolog.clone());
    }
    if let Some(ref epilog) = args.epilog {
        hooks.srun_epilog = Some(epilog.clone());
    }

    let work_dir = args
        .chdir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap().to_string_lossy().into());

    // SrunProlog: run locally before dispatching
    if let Some(ref srun_prolog) = hooks.srun_prolog {
        let ctx = srun_hook_context("prolog_srun", &work_dir);
        spur_core::hooks::run_hook(srun_prolog, &ctx)
            .await
            .context("SrunProlog failed — step not dispatched")?;
    }

    // Step mode: if running inside an allocation, create a step instead of a new job
    if let Ok(parent_job_id) = std::env::var("SPUR_JOB_ID") {
        if let Ok(job_id) = parent_job_id.parse::<u32>() {
            return run_as_step(&args, &matches, job_id, &hooks, &work_dir, &resolved_mpi).await;
        }
    }

    run_standalone_srun(&args, &hooks, &work_dir, &resolved_mpi, &submit_line).await
}

/// Resolve `--mpi` for standalone srun vs step mode inside an allocation.
fn resolve_srun_mpi(args: &SrunArgs, matches: &ArgMatches, step_mode: bool) -> Result<String> {
    let mpi_explicit = was_cli_set(matches, "mpi")
        || crate::env_defaults::env_first(&["SPUR_MPI_TYPE", "SLURM_MPI_TYPE"]).is_some();
    if step_mode && !mpi_explicit && args.mpi.is_none() {
        return Ok(String::new());
    }
    let raw = args.mpi.as_deref().unwrap_or(spur_core::mpi::MPI_NONE);
    Ok(spur_core::mpi::parse_mpi_option(raw)
        .map_err(|e| anyhow::anyhow!(e))?
        .expect("list handled above"))
}

/// Apply environment-variable defaults to any flag not set on the command
/// line. srun mirrors real Slurm: most flags read `SLURM_*` (inherited from an
/// allocation's output vars), a few read genuine `SRUN_*`, and each also
/// accepts its `SPUR_*` native twin. CLI always overrides env.
fn resolve_srun_env(matches: &ArgMatches, args: &mut SrunArgs) -> Result<()> {
    apply_str(
        matches,
        "job_name",
        &["SPUR_JOB_NAME", "SLURM_JOB_NAME"],
        &mut args.job_name,
    );
    apply_str(
        matches,
        "partition",
        &[
            "SPUR_PARTITION",
            "SPUR_JOB_PARTITION",
            "SLURM_PARTITION",
            "SLURM_JOB_PARTITION",
        ],
        &mut args.partition,
    );
    apply_str(
        matches,
        "account",
        &[
            "SPUR_ACCOUNT",
            "SPUR_JOB_ACCOUNT",
            "SLURM_ACCOUNT",
            "SLURM_JOB_ACCOUNT",
        ],
        &mut args.account,
    );
    apply_str(
        matches,
        "qos",
        &["SPUR_QOS", "SPUR_JOB_QOS", "SLURM_QOS", "SLURM_JOB_QOS"],
        &mut args.qos,
    );
    apply_num(
        matches,
        "nodes",
        &[
            "SPUR_NNODES",
            "SPUR_JOB_NUM_NODES",
            "SLURM_NNODES",
            "SLURM_JOB_NUM_NODES",
        ],
        &mut args.nodes,
    )?;
    apply_num_opt(
        matches,
        "ntasks",
        &["SPUR_NTASKS", "SPUR_NPROCS", "SLURM_NTASKS", "SLURM_NPROCS"],
        &mut args.ntasks,
    )?;
    apply_num(
        matches,
        "cpus_per_task",
        &["SPUR_CPUS_PER_TASK", "SLURM_CPUS_PER_TASK"],
        &mut args.cpus_per_task,
    )?;
    apply_str(
        matches,
        "mem",
        &["SPUR_MEM_PER_NODE", "SLURM_MEM_PER_NODE"],
        &mut args.mem,
    );
    apply_str(
        matches,
        "time",
        &["SPUR_TIMELIMIT", "SLURM_TIMELIMIT"],
        &mut args.time,
    );
    apply_csv(
        matches,
        "gres",
        &["SPUR_GRES", "SLURM_GRES"],
        &mut args.gres,
    );
    apply_str(
        matches,
        "gpus",
        &["SPUR_GPUS", "SLURM_GPUS"],
        &mut args.gpus,
    );
    apply_str(
        matches,
        "gpus_per_node",
        &["SPUR_GPUS_PER_NODE", "SLURM_GPUS_PER_NODE"],
        &mut args.gpus_per_node,
    );
    apply_str(
        matches,
        "gpus_per_task",
        &["SPUR_GPUS_PER_TASK", "SLURM_GPUS_PER_TASK"],
        &mut args.gpus_per_task,
    );
    apply_str(
        matches,
        "chdir",
        &[
            "SPUR_REMOTE_CWD",
            "SPUR_WORKING_DIR",
            "SLURM_REMOTE_CWD",
            "SLURM_WORKING_DIR",
        ],
        &mut args.chdir,
    );
    apply_str(
        matches,
        "cpu_bind",
        &["SPUR_CPU_BIND", "SLURM_CPU_BIND"],
        &mut args.cpu_bind,
    );
    apply_str(
        matches,
        "gpu_bind",
        &["SPUR_GPU_BIND", "SLURM_GPU_BIND"],
        &mut args.gpu_bind,
    );
    apply_str(
        matches,
        "constraint",
        &["SPUR_CONSTRAINT", "SLURM_CONSTRAINT"],
        &mut args.constraint,
    );
    apply_str(
        matches,
        "reservation",
        &["SPUR_RESERVATION", "SLURM_RESERVATION"],
        &mut args.reservation,
    );
    apply_str(
        matches,
        "mpi",
        &["SPUR_MPI_TYPE", "SLURM_MPI_TYPE"],
        &mut args.mpi,
    );
    apply_flag(
        matches,
        "label",
        &["SPUR_LABELIO", "SLURM_LABELIO"],
        &mut args.label,
    );
    apply_str(
        matches,
        "output",
        &["SPUR_OUTPUT", "SRUN_OUTPUT"],
        &mut args.output,
    );
    apply_str(
        matches,
        "error",
        &["SPUR_ERROR", "SRUN_ERROR"],
        &mut args.error,
    );
    apply_str(
        matches,
        "input",
        &["SPUR_INPUT", "SRUN_INPUT"],
        &mut args.input,
    );
    apply_str(
        matches,
        "container_image",
        &["SPUR_CONTAINER", "SRUN_CONTAINER"],
        &mut args.container_image,
    );
    apply_str(
        matches,
        "prolog",
        &["SPUR_PROLOG", "SLURM_PROLOG"],
        &mut args.prolog,
    );
    apply_str(
        matches,
        "epilog",
        &["SPUR_EPILOG", "SLURM_EPILOG"],
        &mut args.epilog,
    );

    Ok(())
}

struct ResolvedIoPaths {
    stdout: String,
    stderr: String,
    stdin: String,
}

/// Resolve I/O paths from CLI args. When `-o` is set but `-e` is not,
/// stderr follows stdout (Slurm default behavior).
fn resolve_io_paths(args: &SrunArgs) -> ResolvedIoPaths {
    let stdout = args.output.clone().unwrap_or_default();
    let stderr = args.error.clone().unwrap_or_else(|| {
        if stdout.is_empty() {
            String::new()
        } else {
            stdout.clone()
        }
    });
    let stdin = args.input.clone().unwrap_or_default();
    ResolvedIoPaths {
        stdout,
        stderr,
        stdin,
    }
}

#[derive(Debug)]
struct StepDispatchResult {
    exit_code: i32,
}

fn srun_dispatch_environment(args: &SrunArgs) -> HashMap<String, String> {
    let mut environment: HashMap<String, String> = std::env::vars().collect();
    if let Some(ref cpu_bind) = args.cpu_bind {
        environment.insert("SPUR_CPU_BIND".into(), cpu_bind.clone());
    }
    if let Some(ref gpu_bind) = args.gpu_bind {
        environment.insert("SPUR_GPU_BIND".into(), gpu_bind.clone());
    }
    if args.label {
        environment.insert("SPUR_LABEL".into(), "1".into());
    }
    environment
}

fn build_srun_job_spec(
    args: &SrunArgs,
    work_dir: &str,
    io: &ResolvedIoPaths,
    mpi: &str,
    submit_line: &str,
) -> Result<JobSpec> {
    // GPU requests use dedicated proto fields; --gres=gpu:* stays in gres.
    let gres = args.gres.clone();
    let gpus = crate::sbatch::parse_gpu_flag(args.gpus.as_deref())?;
    let gpus_per_node = crate::sbatch::parse_gpu_flag(args.gpus_per_node.as_deref())?;
    let gpus_per_task = crate::sbatch::parse_gpu_flag(args.gpus_per_task.as_deref())?;
    crate::sbatch::validate_gpu_flags(
        gpus.is_some(),
        gpus_per_node.is_some(),
        gpus_per_task.is_some(),
        &gres,
    )?;

    let time_limit = args
        .time
        .as_ref()
        .and_then(|t| spur_core::config::parse_time_minutes(t))
        .map(|mins| prost_types::Duration {
            seconds: mins as i64 * 60,
            nanos: 0,
        });

    let environment = srun_dispatch_environment(args);

    let memory_mb = args
        .mem
        .as_ref()
        .map(|m| parse_memory_mb(m))
        .transpose()?
        .unwrap_or(0);

    Ok(JobSpec {
        name: args
            .job_name
            .clone()
            .unwrap_or_else(|| args.command[0].clone()),
        partition: args.partition.clone().unwrap_or_default(),
        account: args.account.clone().unwrap_or_default(),
        qos: args.qos.clone().unwrap_or_default(),
        user: crate::interactive::current_user()?,
        uid: nix::unistd::getuid().as_raw(),
        gid: nix::unistd::getgid().as_raw(),
        num_nodes: args.nodes,
        num_tasks: crate::sbatch::effective_ntasks(args.ntasks, args.ntasks_per_node, args.nodes),
        tasks_per_node: args.ntasks_per_node.unwrap_or(0),
        cpus_per_task: args.cpus_per_task,
        memory_per_node_mb: memory_mb,
        gres,
        gpus,
        gpus_per_node,
        gpus_per_task,
        script: build_command_script(&args.command)?,
        work_dir: work_dir.to_string(),
        stdout_path: io.stdout.clone(),
        stderr_path: io.stderr.clone(),
        stdin_path: io.stdin.clone(),
        environment,
        time_limit,
        constraint: args.constraint.clone().unwrap_or_default(),
        submit_line: submit_line.to_string(),
        nodelist: args.nodelist.clone().unwrap_or_default(),
        exclude: args.exclude.clone().unwrap_or_default(),
        reservation: args.reservation.clone().unwrap_or_default(),
        exclusive: args.exclusive,
        mpi: mpi.to_string(),
        licenses: args.licenses.clone(),
        container_image: args.container_image.clone().unwrap_or_default(),
        container_mounts: args.container_mounts.clone(),
        container_workdir: args.container_workdir.clone().unwrap_or_default(),
        container_name: args.container_name.clone().unwrap_or_default(),
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
        container_entrypoint: args.container_entrypoint.clone().unwrap_or_default(),
        container_remap_root: args.container_remap_root,
        srun_job: true,
        pty: args.pty,
        ..Default::default()
    })
}

/// Build the step's `ContainerSpec` from srun's container flags. Returns `None`
/// when no image is requested, so the step inherits the parent job's container
/// config (or runs on the host). srun does not surface name/readonly/entrypoint.
fn container_spec_from_srun_args(args: &SrunArgs) -> Option<ContainerSpec> {
    let image = args.container_image.clone()?;
    if image.is_empty() {
        return None;
    }
    Some(ContainerSpec {
        image,
        mounts: args.container_mounts.clone(),
        workdir: args.container_workdir.clone().unwrap_or_default(),
        name: args.container_name.clone().unwrap_or_default(),
        // --container-readonly is rejected at submission (a no-op today), so it
        // never reaches here as true.
        readonly: false,
        mount_home: args.container_mount_home,
        env: args
            .container_env
            .iter()
            .filter_map(|s| {
                s.split_once('=')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
            })
            .collect(),
        entrypoint: args.container_entrypoint.clone().unwrap_or_default(),
        remap_root: args.container_remap_root,
    })
}

fn install_ctrl_c_cancel(
    client: SlurmControllerClient<crate::authclient::AuthChannel>,
    job_id: u32,
    submit_user: String,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut client = client;
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("\nsrun: cancelling job {}...", job_id);
            let cancel_user =
                crate::interactive::resolve_job_owner_for_cancel(&mut client, job_id, &submit_user)
                    .await;
            let _ = client
                .cancel_job(CancelJobRequest {
                    job_id,
                    signal: 2,
                    user: cancel_user,
                })
                .await;
            std::process::exit(130);
        }
    })
}

struct RunningAllocation {
    nodelist: String,
    user: String,
}

async fn wait_for_job_running(
    client: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
    job_id: u32,
) -> Result<RunningAllocation> {
    let mut poll_interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
    let mut warned_unknown_state = false;

    loop {
        poll_interval.tick().await;
        let job = client
            .get_job(GetJobRequest { job_id })
            .await
            .context("failed to get job status")?
            .into_inner();

        match JobState::try_from(job.state) {
            Ok(JobState::JobRunning) if !job.nodelist.is_empty() => {
                return Ok(RunningAllocation {
                    nodelist: job.nodelist,
                    user: job.user,
                });
            }
            Ok(JobState::JobRunning) => {}
            Ok(
                JobState::JobCompleted
                | JobState::JobFailed
                | JobState::JobCancelled
                | JobState::JobTimeout
                | JobState::JobNodeFail
                | JobState::JobDeadline,
            ) => {
                anyhow::bail!(
                    "job {} reached terminal state before allocation was ready",
                    job_id
                );
            }
            Ok(_) => {}
            Err(_) if !warned_unknown_state => {
                warned_unknown_state = true;
                eprintln!(
                    "srun: warning: job {} has unrecognized state {} \
                     (controller may be newer than client)",
                    job_id, job.state
                );
            }
            Err(_) => {}
        }
    }
}

async fn emit_step_output(
    io: &ResolvedIoPaths,
    job_id: u32,
    work_dir: &str,
    stdout: &str,
    stderr: &str,
) {
    let stderr_appends = !io.stderr.is_empty() && io.stderr == io.stdout;

    if !stdout.is_empty() {
        if io.stdout.is_empty() {
            print!("{stdout}");
        } else {
            write_step_file(stdout, &io.stdout, job_id, work_dir, false).await;
        }
    }
    if !stderr.is_empty() {
        if io.stderr.is_empty() {
            eprint!("{stderr}");
        } else {
            write_step_file(stderr, &io.stderr, job_id, work_dir, stderr_appends).await;
        }
    }
}

struct StepDispatchParams<'a> {
    args: &'a SrunArgs,
    explicit_num_nodes: Option<u32>,
    requested_nodelist: Option<&'a str>,
    work_dir: &'a str,
    io: &'a ResolvedIoPaths,
    mpi: &'a str,
    user: &'a str,
}

async fn dispatch_step(
    client: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
    job_id: u32,
    params: &StepDispatchParams<'_>,
) -> Result<StepDispatchResult> {
    let args = params.args;
    let work_dir = params.work_dir;
    let io = params.io;
    let step_mpi = params.mpi;
    let ntasks = crate::sbatch::effective_ntasks(args.ntasks, args.ntasks_per_node, args.nodes);
    let step_id = client
        .create_job_step(CreateJobStepRequest {
            job_id,
            command: args.command.clone(),
            num_tasks: ntasks,
            cpus_per_task: args.cpus_per_task,
            overlap: false,
            pty: false,
            winsize: None,
            node: String::new(),
            user: params.user.to_string(),
            uid: nix::unistd::geteuid().as_raw(),
            // The buffered path carries the container on RunStep; this call only
            // registers the step to obtain its id.
            container: None,
            num_nodes: params.explicit_num_nodes.unwrap_or(0),
            nodelist: params.requested_nodelist.unwrap_or_default().to_string(),
        })
        .await
        .context("failed to create job step")?
        .into_inner()
        .step_id;

    for warning in step_unsupported_warnings(args) {
        eprintln!("{warning}");
    }

    let environment = srun_dispatch_environment(args);
    validate_step_cpu_bind(&environment, ntasks)?;
    // Live-stream the step's output when it lands on a single node and the user
    // has not redirected to a file. The tail runs concurrently with the blocking
    // RunStep and ends when the step leaves the agent's active_steps (#781).
    let live_node = if io.stdout.is_empty() && io.stderr.is_empty() {
        single_step_node(client, job_id).await
    } else {
        None
    };
    let mut stream_handle = live_node.map(|node| {
        let mut ctrl = client.clone();
        let user = params.user.to_string();
        tokio::spawn(async move {
            stream_step_output_live(&mut ctrl, &node, job_id, step_id, &user).await
        })
    });

    let resp = client
        .run_step(RunStepRequest {
            job_id,
            command: args.command.clone(),
            uid: nix::unistd::geteuid().as_raw(),
            gid: nix::unistd::getegid().as_raw(),
            work_dir: work_dir.to_string(),
            environment,
            step_id,
            label: args.label,
            mpi: step_mpi.to_string(),
            user: params.user.to_string(),
            container: container_spec_from_srun_args(args),
        })
        .await
        .context("RunStep dispatch failed")?
        .into_inner();

    if !resp.node.is_empty() {
        eprintln!("srun: dispatched to node {}", resp.node);
    }

    // The step has left active_steps, so a live tail eofs promptly; bound the
    // join so a tail that never resolved (step failed to start) can't hang, and
    // fall back to the buffered output whenever streaming didn't cover it.
    let streamed = if let Some(handle) = stream_handle.as_mut() {
        match tokio::time::timeout(std::time::Duration::from_secs(3), &mut *handle).await {
            Ok(joined) => joined.unwrap_or(false),
            Err(_) => {
                handle.abort();
                false
            }
        }
    } else {
        false
    };

    if !streamed {
        emit_step_output(io, job_id, work_dir, &resp.stdout, &resp.stderr).await;
    }

    Ok(StepDispatchResult {
        exit_code: resp.exit_code,
    })
}

async fn release_srun_allocation(
    client: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
    job_id: u32,
    user: &str,
    exit_code: i32,
) {
    if let Err(e) = client
        .complete_job(CompleteJobRequest {
            job_id,
            exit_code,
            user: user.to_string(),
        })
        .await
    {
        eprintln!(
            "srun: warning: failed to release allocation for job {}: {}",
            job_id, e
        );
        if let Err(ce) = client
            .cancel_job(CancelJobRequest {
                job_id,
                signal: 2,
                user: user.to_string(),
            })
            .await
        {
            eprintln!(
                "srun: warning: failed to cancel job {} after CompleteJob failure: {}",
                job_id, ce
            );
        }
    }
}

async fn dispatch_step_cancellable(
    client: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
    job_id: u32,
    params: &StepDispatchParams<'_>,
    cancel_job_on_interrupt: bool,
) -> Result<StepDispatchResult> {
    if cancel_job_on_interrupt {
        return dispatch_step(client, job_id, params).await;
    }
    tokio::select! {
        result = dispatch_step(client, job_id, params) => result,
        _ = tokio::signal::ctrl_c() => {
            eprintln!("\nsrun: step interrupted");
            std::process::exit(130);
        }
    }
}

async fn run_standalone_srun(
    args: &SrunArgs,
    hooks: &HooksConfig,
    work_dir: &str,
    mpi: &str,
    submit_line: &str,
) -> Result<()> {
    let io = resolve_io_paths(args);
    let channel = crate::authclient::connect(&args.controller)
        .await
        .context("failed to connect to spurctld")?;
    let mut client = SlurmControllerClient::new(channel);
    let job_spec = build_srun_job_spec(args, work_dir, &io, mpi, submit_line)?;
    let submit_user = job_spec.user.clone();
    let submit_resp = client
        .submit_job(SubmitJobRequest {
            spec: Some(job_spec),
        })
        .await
        .context("job submission failed")?
        .into_inner();
    for warning in &submit_resp.warnings {
        eprintln!("srun: warning: {warning}");
    }
    let job_id = submit_resp.job_id;

    eprintln!("srun: Pending job allocation {}...", job_id);

    let ctrl_c_handle = install_ctrl_c_cancel(client.clone(), job_id, submit_user.clone());

    let running = wait_for_job_running(&mut client, job_id).await?;
    if !running.nodelist.is_empty() {
        eprintln!("srun: job {} running on {}", job_id, running.nodelist);
    }

    let known_owner = (!running.user.is_empty()).then_some(running.user.as_str());
    let owner = crate::interactive::job_caller_user(&mut client, job_id, known_owner).await?;
    if owner.is_empty() {
        anyhow::bail!("srun: job {} has no owner after allocation", job_id);
    }

    // Guard drops (and stops pinging) on every return path, including `?`.
    let _keepalive =
        crate::interactive::spawn_keepalive(client.clone(), job_id, owner.clone(), "srun");

    if args.pty {
        ctrl_c_handle.abort();
        eprintln!("srun: opening interactive session on {}", running.nodelist);
        let result = run_interactive_pty(
            &mut client,
            job_id,
            args.command.clone(),
            String::new(),
            &owner,
            container_spec_from_srun_args(args),
        )
        .await;
        let _ = client
            .cancel_job(CancelJobRequest {
                job_id,
                signal: 0,
                user: owner.clone(),
            })
            .await;
        // SrunProlog already ran for this invocation; pair it.
        run_srun_epilog(hooks, work_dir).await;
        std::process::exit(result?);
    }

    let job = client
        .get_job(GetJobRequest { job_id })
        .await
        .context("failed to get job after allocation")?
        .into_inner();

    if job.srun_step_dispatch {
        let step_params = StepDispatchParams {
            args,
            explicit_num_nodes: None,
            requested_nodelist: None,
            work_dir,
            io: &io,
            mpi,
            user: &owner,
        };
        let dispatch_result =
            dispatch_step_cancellable(&mut client, job_id, &step_params, true).await;
        let exit_code = match &dispatch_result {
            Ok(result) => result.exit_code,
            Err(_) => 1,
        };
        release_srun_allocation(&mut client, job_id, &owner, exit_code).await;
        let result = dispatch_result?;
        let state = if result.exit_code == 0 {
            JobState::JobCompleted
        } else {
            JobState::JobFailed
        };
        handle_terminal_state(
            state,
            job_id,
            result.exit_code,
            work_dir,
            &io.stdout,
            hooks,
            true,
        )
        .await;
    }

    let output_streamed = if io.stdout.is_empty() {
        try_stream_output(&mut client, &running.nodelist, job_id, &owner).await
    } else {
        false
    };
    poll_for_completion(
        &mut client,
        job_id,
        work_dir,
        &io.stdout,
        hooks,
        output_streamed,
    )
    .await;

    Ok(())
}

/// Client-side output path resolution, mirroring the agent's logic.
fn resolve_output_path_client(pattern: &str, job_id: u32, work_dir: &str) -> String {
    let resolved = if pattern.is_empty() {
        format!("spur-{}.out", job_id)
    } else {
        pattern
            .replace("%j", &job_id.to_string())
            .replace("%J", &job_id.to_string())
    };

    if std::path::Path::new(&resolved).is_absolute() {
        resolved
    } else {
        std::path::PathBuf::from(work_dir)
            .join(resolved)
            .to_string_lossy()
            .into()
    }
}

/// Wrap the command argv in a bash script for batch submission.
///
/// Each argv element is shell-escaped so metacharacters (spaces, quotes, `;`,
/// `>`, `$`, globs) stay literal arguments instead of being reinterpreted by
/// the wrapper shell. This matches Slurm's semantics, where `srun` execs the
/// argv directly with no shell: `srun touch "my file"` creates one file, and
/// `srun bash -c 'a; b'` passes the whole script as a single `-c` argument.
fn build_command_script(command: &[String]) -> Result<String> {
    let cmd_line = shlex::try_join(command.iter().map(String::as_str))
        .context("command contains a NUL byte and cannot be run")?;
    Ok(format!("#!/bin/bash\n{cmd_line}\n"))
}

/// First node of a `-w` value. Interactive `--pty` attaches to a single node,
/// so a comma list reduces to its head; an empty value stays empty (controller
/// then defaults to the first allocated node). Hostlist brackets are not
/// expanded here — the controller rejects an unmatched name with a clear error.
fn first_node(nodelist: &str) -> String {
    nodelist
        .split(',')
        .next()
        .unwrap_or_default()
        .trim()
        .to_string()
}

async fn try_stream_output(
    controller: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
    nodelist: &str,
    job_id: u32,
    user: &str,
) -> bool {
    // The controller reports a hostlist, so `node[1-3]` has to be expanded
    // before a name is usable as an address.
    let Ok(nodes) = spur_core::hostlist::expand(nodelist) else {
        return false;
    };
    let Some(first_node) = nodes.first() else {
        return false;
    };

    if controller
        .get_node(GetNodeRequest {
            name: first_node.clone(),
        })
        .await
        .is_err()
    {
        return false;
    }

    let agent_addr = format!("http://{first_node}:6818");

    let mut agent = match crate::interactive::connect_agent(&agent_addr).await {
        Ok(c) => c,
        Err(_) => return false,
    };

    let mut stream = match agent
        .stream_job_output(StreamJobOutputRequest {
            job_id,
            stream: "stdout".into(),
            user: user.to_string(),
            ..Default::default()
        })
        .await
    {
        Ok(resp) => resp.into_inner(),
        Err(_) => return false,
    };

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    loop {
        match stream.message().await {
            Ok(Some(chunk)) => {
                if chunk.eof {
                    return true;
                }
                let _ = handle.write_all(&chunk.data);
                let _ = handle.flush();
            }
            Ok(None) => return true,
            Err(_) => return false,
        }
    }
}

/// Write a step output stream's chunks to this process's stdout/stderr as they
/// arrive, returning true on a clean eof. The lock is held only across each
/// synchronous write, never an await.
async fn drain_step_stream(
    mut stream: tonic::Streaming<StreamJobOutputChunk>,
    to_stderr: bool,
) -> bool {
    loop {
        match stream.message().await {
            Ok(Some(chunk)) => {
                if chunk.eof {
                    return true;
                }
                if to_stderr {
                    let mut h = std::io::stderr().lock();
                    let _ = h.write_all(&chunk.data);
                    let _ = h.flush();
                } else {
                    let mut h = std::io::stdout().lock();
                    let _ = h.write_all(&chunk.data);
                    let _ = h.flush();
                }
            }
            Ok(None) => return true,
            Err(_) => return false,
        }
    }
}

/// Tail a step's stdout and stderr live from the agent on `node`, writing chunks
/// to this process's stdout/stderr as they arrive. Returns true only if it
/// connected and both streams reached a clean eof, so the caller can suppress
/// the buffered response; any failure returns false and the caller falls back to
/// printing the buffered output. Runs concurrently with the blocking RunStep and
/// ends when the step leaves the agent's active_steps.
async fn stream_step_output_live(
    controller: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
    node: &str,
    job_id: u32,
    step_id: u32,
    user: &str,
) -> bool {
    if controller
        .get_node(GetNodeRequest {
            name: node.to_string(),
        })
        .await
        .is_err()
    {
        return false;
    }
    let agent_addr = format!("http://{node}:6818");
    let req = |stream: &str| StreamJobOutputRequest {
        job_id,
        step_id,
        stream: stream.to_string(),
        user: user.to_string(),
    };
    let Ok(mut agent_out) = crate::interactive::connect_agent(&agent_addr).await else {
        return false;
    };
    let Ok(mut agent_err) = crate::interactive::connect_agent(&agent_addr).await else {
        return false;
    };
    let out = match agent_out.stream_job_output(req("stdout")).await {
        Ok(r) => r.into_inner(),
        Err(_) => return false,
    };
    let err = match agent_err.stream_job_output(req("stderr")).await {
        Ok(r) => r.into_inner(),
        Err(_) => return false,
    };
    let (out_ok, err_ok) =
        tokio::join!(drain_step_stream(out, false), drain_step_stream(err, true));
    out_ok && err_ok
}

/// The single node a step will run on, or None when live streaming should be
/// skipped: a multi-node step (its output is aggregated in the response) or an
/// allocation whose nodelist cannot be read.
async fn single_step_node(
    controller: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
    job_id: u32,
) -> Option<String> {
    let job = controller
        .get_job(GetJobRequest { job_id })
        .await
        .ok()?
        .into_inner();
    match spur_core::hostlist::expand(&job.nodelist).ok()?.as_slice() {
        [only] => Some(only.clone()),
        _ => None,
    }
}

async fn poll_for_completion(
    client: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
    job_id: u32,
    work_dir: &str,
    stdout_path: &str,
    hooks: &HooksConfig,
    output_streamed: bool,
) {
    let mut poll_interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
    let mut warned_unknown_state = false;
    loop {
        poll_interval.tick().await;
        match client.get_job(GetJobRequest { job_id }).await {
            Ok(resp) => {
                let job = resp.into_inner();
                match JobState::try_from(job.state) {
                    Ok(
                        state @ (JobState::JobCompleted
                        | JobState::JobFailed
                        | JobState::JobCancelled
                        | JobState::JobTimeout
                        | JobState::JobNodeFail
                        | JobState::JobDeadline),
                    ) => {
                        handle_terminal_state(
                            state,
                            job_id,
                            job.exit_code,
                            work_dir,
                            stdout_path,
                            hooks,
                            output_streamed,
                        )
                        .await;
                    }
                    Ok(_) => {}
                    Err(_) if !warned_unknown_state => {
                        warned_unknown_state = true;
                        eprintln!(
                            "srun: warning: job {} has unrecognized state {} \
                             (controller may be newer than client)",
                            job_id, job.state
                        );
                    }
                    Err(_) => {}
                }
            }
            Err(e) => {
                eprintln!("srun: warning: failed to get job status: {}", e.message());
            }
        }
    }
}

async fn run_srun_epilog(hooks: &HooksConfig, work_dir: &str) {
    let Some(ref srun_epilog) = hooks.srun_epilog else {
        return;
    };
    let ctx = srun_hook_context("epilog_srun", work_dir);
    if let Err(e) = spur_core::hooks::run_hook(srun_epilog, &ctx).await {
        eprintln!("srun: warning: SrunEpilog failed: {}", e);
    }
}

async fn handle_terminal_state(
    state: JobState,
    job_id: u32,
    exit_code: i32,
    work_dir: &str,
    stdout_path: &str,
    hooks: &HooksConfig,
    output_streamed: bool,
) -> ! {
    run_srun_epilog(hooks, work_dir).await;

    let should_print = !output_streamed && stdout_path.is_empty();

    match state {
        JobState::JobCompleted => {
            if should_print {
                print_job_output(work_dir, stdout_path, job_id).await;
            }
            std::process::exit(exit_code);
        }
        JobState::JobFailed => {
            if should_print {
                print_job_output(work_dir, stdout_path, job_id).await;
            }
            eprintln!("srun: job {} failed with exit code {}", job_id, exit_code);
            std::process::exit(exit_code.max(1));
        }
        JobState::JobCancelled => {
            eprintln!("srun: job {} cancelled", job_id);
            std::process::exit(1);
        }
        JobState::JobTimeout => {
            eprintln!("srun: job {} timed out", job_id);
            std::process::exit(1);
        }
        JobState::JobNodeFail => {
            eprintln!("srun: job {} failed (node failure)", job_id);
            std::process::exit(1);
        }
        JobState::JobDeadline => {
            eprintln!("srun: job {} hit its --deadline", job_id);
            std::process::exit(1);
        }
        _ => {
            eprintln!("srun: job {} ended with state {:?}", job_id, state);
            std::process::exit(1);
        }
    }
}

async fn print_job_output(work_dir: &str, stdout_path: &str, job_id: u32) {
    let path = resolve_output_path_client(stdout_path, job_id, work_dir);
    if let Ok(content) = tokio::fs::read_to_string(&path).await {
        print!("{content}");
    }
}

/// Write step output to a file, used by step dispatch for stdout and stderr.
async fn write_step_file(content: &str, pattern: &str, job_id: u32, work_dir: &str, append: bool) {
    let resolved = resolve_output_path_client(pattern, job_id, work_dir);
    if let Some(parent) = std::path::Path::new(&resolved).parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    let result = if append {
        use tokio::io::AsyncWriteExt;
        match tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&resolved)
            .await
        {
            Ok(mut f) => f.write_all(content.as_bytes()).await,
            Err(e) => Err(e),
        }
    } else {
        tokio::fs::write(&resolved, content).await
    };
    if let Err(e) = result {
        eprintln!("srun: warning: failed to write to {}: {}", resolved, e);
    }
}

fn warn_unsupported_cpu_bind(environment: &HashMap<String, String>) {
    if let Some(cpu_bind) = spur_core::task_launch::unsupported_cpu_bind(environment) {
        eprintln!(
            "srun: warning: --cpu-bind={cpu_bind} is not applied in step mode \
             (supported: rank, map_cpu, mask_cpu)"
        );
    }
}

/// A session that never ran still closes its step out, reporting 1 so the step
/// does not linger Running for the life of the allocation.
fn interactive_exit_code(outcome: &Result<i32>) -> i32 {
    outcome.as_ref().copied().unwrap_or(1)
}

/// Report a PTY step's exit code. `RunStep` records its own steps controller-side;
/// an interactive session streams client-to-agent, so nothing else closes this one.
async fn complete_interactive_step(
    ctrl: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
    job_id: u32,
    step_id: u32,
    exit_code: i32,
    user: &str,
) {
    // Bounded: this runs after the PTY has closed, so an unreachable leader would
    // otherwise hang the user's shell with nothing on screen.
    let report = ctrl.complete_job_step(CompleteJobStepRequest {
        job_id,
        step_id,
        exit_code,
        user: user.to_string(),
        uid: nix::unistd::geteuid().as_raw(),
    });
    match tokio::time::timeout(std::time::Duration::from_secs(5), report).await {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => eprintln!("srun: warning: failed to record step {step_id} completion: {e}"),
        Err(_) => eprintln!("srun: warning: timed out recording step {step_id} completion"),
    }
}

/// Create an interactive PTY step on a running job and attach to it, reporting
/// the step's exit code on every exit path once the step exists.
async fn run_interactive_pty(
    ctrl: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
    job_id: u32,
    command: Vec<String>,
    node: String,
    user: &str,
    container: Option<ContainerSpec>,
) -> Result<i32> {
    let winsize = crate::interactive::get_terminal_size();

    let mut created_step: Option<u32> = None;
    let mut cached_step: Option<(u32, String)> = None;
    // The controller resolves the step's own container against the parent job's
    // (inheriting `salloc/sbatch --container-image`) and echoes the effective spec;
    // persisted across retries since a cached step skips the CreateJobStep call.
    let mut effective_container: Option<ContainerSpec> = None;

    let outcome: Result<i32> = 'session: {
        let mut last_err: Option<anyhow::Error> = None;

        for attempt in 0..5 {
            if attempt > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }

            let (step_id, node_addr) = if let Some(ref cached) = cached_step {
                cached.clone()
            } else {
                let step_resp = match ctrl
                    .create_job_step(CreateJobStepRequest {
                        job_id,
                        command: command.clone(),
                        num_tasks: 1,
                        cpus_per_task: 1,
                        // A PTY step always shares the allocation it runs in.
                        overlap: true,
                        pty: true,
                        winsize: Some(winsize),
                        node: node.clone(),
                        user: user.to_string(),
                        uid: nix::unistd::geteuid().as_raw(),
                        container: container.clone(),
                        num_nodes: 0,
                        nodelist: String::new(),
                    })
                    .await
                {
                    Ok(resp) => resp.into_inner(),
                    Err(status) if is_retryable_status(&status) && attempt < 4 => {
                        last_err = Some(anyhow::anyhow!("CreateJobStep: {}", status.message()));
                        continue;
                    }
                    Err(status) => {
                        break 'session Err(anyhow::anyhow!(
                            "CreateJobStep failed: {}",
                            status.message()
                        ))
                    }
                };

                created_step = Some(step_resp.step_id);
                effective_container = step_resp.container.clone();
                if step_resp.node_addr.is_empty() {
                    break 'session Err(anyhow::anyhow!(
                        "controller did not return a node address for job {}",
                        job_id
                    ));
                }
                let pair = (step_resp.step_id, format!("http://{}", step_resp.node_addr));
                cached_step = Some(pair.clone());
                pair
            };

            let mut agent = match crate::interactive::connect_agent(&node_addr).await {
                Ok(agent) => agent,
                Err(e) => break 'session Err(e),
            };

            match crate::interactive::open_interactive_session(
                &mut agent,
                job_id,
                step_id,
                command.clone(),
                winsize,
                true,
                user,
                effective_container.clone(),
            )
            .await
            {
                Ok(handle) => {
                    break 'session crate::interactive::drive_interactive_session(handle).await
                }
                Err(status) if is_retryable_status(&status) && attempt < 4 => {
                    last_err = Some(anyhow::anyhow!("InteractiveSession: {}", status.message()));
                    continue;
                }
                Err(status) => {
                    break 'session Err(anyhow::anyhow!(
                        "InteractiveSession RPC failed: {}",
                        status.message()
                    ))
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("interactive session failed after retries")))
    };

    if let Some(step_id) = created_step {
        let exit_code = interactive_exit_code(&outcome);
        complete_interactive_step(ctrl, job_id, step_id, exit_code, user).await;
    }

    outcome
}

fn is_retryable_status(status: &tonic::Status) -> bool {
    matches!(
        status.code(),
        tonic::Code::NotFound | tonic::Code::FailedPrecondition | tonic::Code::Unavailable
    )
}

/// Flags accepted on the command line that no job step can honor.
fn step_unsupported_warnings(args: &SrunArgs) -> Vec<String> {
    let mut warnings = Vec::new();
    if args.input.is_some() {
        warnings.push("srun: warning: --input is not supported for a job step, ignoring".into());
    }
    warnings
}

/// Flags a `--pty` step drops. Sizing reads `matches`, not `args`: salloc exports
/// SPUR_NTASKS into every step, so `args` would warn about flags nobody typed.
fn pty_step_unsupported_warnings(args: &SrunArgs, matches: &ArgMatches) -> Vec<String> {
    let mut warnings = Vec::new();
    if args.output.is_some() || args.error.is_some() {
        warnings.push(
            "srun: warning: --output/--error are not applied to a --pty step; \
             output streams to the terminal"
                .to_string(),
        );
    }

    let sized = ["ntasks", "ntasks_per_node", "cpus_per_task", "nodes"]
        .iter()
        .any(|id| crate::env_defaults::was_cli_set(matches, id));
    if sized {
        warnings.push(
            "srun: warning: --pty runs a single task on one node; \
             --ntasks/--ntasks-per-node/--cpus-per-task/--nodes are ignored"
                .into(),
        );
    }

    let nodelist = args.nodelist.as_deref().unwrap_or_default().trim();
    if nodelist.contains(',') {
        warnings.push(
            "srun: warning: --pty runs on one node; using the first entry of --nodelist".into(),
        );
    }
    if args.chdir.is_some() {
        warnings.push(
            "srun: warning: --chdir does not move a --pty step; it starts in the job's \
             working directory"
                .into(),
        );
    }

    // The agent spawns the PTY from the job's own environment and work dir, so
    // everything srun would otherwise pass through SPUR_* env is dropped.
    let env_scoped = args.cpu_bind.is_some()
        || args.gpu_bind.is_some()
        || args.label
        || crate::env_defaults::was_cli_set(matches, "mpi");
    if env_scoped {
        warnings.push(
            "srun: warning: a --pty step runs in the job's environment; \
             --cpu-bind/--gpu-bind/--label/--mpi are ignored"
                .into(),
        );
    }
    warnings
}

/// How a step inside an allocation is dispatched. Split out from `run_as_step`
/// so the routing decision is testable without a controller.
#[derive(Debug, PartialEq, Eq)]
enum StepDispatchKind {
    Interactive { node: String },
    Buffered,
}

fn step_dispatch_kind(args: &SrunArgs) -> StepDispatchKind {
    if args.pty {
        return StepDispatchKind::Interactive {
            node: first_node(args.nodelist.as_deref().unwrap_or_default()),
        };
    }
    StepDispatchKind::Buffered
}

/// Warn on unsupported `--cpu-bind` modes and reject a spec that cannot cover
/// `ntasks`. Buffered steps only — a `--pty` step forwards no env, so it warns.
fn validate_step_cpu_bind(environment: &HashMap<String, String>, ntasks: u32) -> Result<()> {
    warn_unsupported_cpu_bind(environment);
    if let Some(err) = spur_core::task_launch::map_cpu_bind_error(environment, ntasks)
        .or_else(|| spur_core::task_launch::mask_cpu_bind_error(environment, ntasks))
    {
        anyhow::bail!("{err}");
    }
    Ok(())
}

async fn run_as_step(
    args: &SrunArgs,
    matches: &ArgMatches,
    job_id: u32,
    hooks: &HooksConfig,
    work_dir: &str,
    step_mpi: &str,
) -> Result<()> {
    // dispatch_step warns for the buffered path; the interactive path returns
    // before it, so warn here — and before the controller round-trip.
    let dispatch_kind = step_dispatch_kind(args);
    let warnings = match dispatch_kind {
        StepDispatchKind::Buffered => Vec::new(),
        StepDispatchKind::Interactive { .. } => step_unsupported_warnings(args)
            .into_iter()
            .chain(pty_step_unsupported_warnings(args, matches))
            .collect(),
    };
    for warning in warnings {
        eprintln!("{warning}");
    }

    let channel = crate::authclient::connect(&args.controller)
        .await
        .context("failed to connect to spurctld")?;
    let mut client = spur_proto::controller_client(channel);

    let io = resolve_io_paths(args);
    let user = crate::interactive::job_caller_user(&mut client, job_id, None).await?;

    // An interactive step drives a PTY over InteractiveSession rather than the
    // buffered RunStep path, so it runs its own step and owns the exit code.
    if let StepDispatchKind::Interactive { node } = dispatch_kind {
        let result = run_interactive_pty(
            &mut client,
            job_id,
            args.command.clone(),
            node,
            &user,
            container_spec_from_srun_args(args),
        )
        .await;
        // Runs before `?` so a failed session still pairs the prolog. Not
        // handle_terminal_state: only the step exited, so "job N failed" is wrong.
        run_srun_epilog(hooks, work_dir).await;
        std::process::exit(result?);
    }

    let step_params = StepDispatchParams {
        args,
        explicit_num_nodes: was_cli_set(matches, "nodes").then_some(args.nodes),
        requested_nodelist: args.nodelist.as_deref(),
        work_dir,
        io: &io,
        mpi: step_mpi,
        user: &user,
    };
    let dispatch_result =
        dispatch_step_cancellable(&mut client, job_id, &step_params, false).await?;

    let state = if dispatch_result.exit_code == 0 {
        JobState::JobCompleted
    } else {
        JobState::JobFailed
    };
    handle_terminal_state(
        state,
        job_id,
        dispatch_result.exit_code,
        work_dir,
        &io.stdout,
        hooks,
        true,
    )
    .await;
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

fn load_hooks_config() -> HooksConfig {
    crate::spur_config::load_spur_config().hooks
}

fn load_mpi_config() -> spur_core::config::MpiConfig {
    crate::spur_config::load_spur_config().mpi
}

fn srun_hook_context(script_context: &str, work_dir: &str) -> spur_core::hooks::HookContext {
    spur_core::hooks::HookContext {
        job_id: std::env::var("SPUR_JOB_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        work_dir: work_dir.to_string(),
        uid: nix::unistd::getuid().as_raw(),
        gid: nix::unistd::getgid().as_raw(),
        partition: String::new(),
        nodelist: String::new(),
        script_context: script_context.into(),
        gpu_devices: Vec::new(),
        cpus: 1,
        memory_mb: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A --pty step now runs inside the requested container, so it no longer
    /// warns that container options are dropped.
    #[test]
    fn pty_step_does_not_warn_on_container_image() {
        let (args, matches) =
            parse_srun(&["srun", "--pty", "--container-image", "img.sqsh", "bash"]);
        assert!(pty_step_unsupported_warnings(&args, &matches).is_empty());
        assert!(step_unsupported_warnings(&args).is_empty());
    }

    #[test]
    fn pty_step_does_not_warn_on_container_flags() {
        // Container modifiers ride the interactive path too, so they no longer warn.
        let (args, matches) = parse_srun(&["srun", "--pty", "--container-mounts", "/a:/b", "bash"]);
        assert!(pty_step_unsupported_warnings(&args, &matches).is_empty());
    }

    #[test]
    fn step_warns_on_input() {
        let args =
            SrunArgs::try_parse_from(["srun", "--input", "in.txt", "hostname"]).expect("parse");
        let warnings = step_unsupported_warnings(&args);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("--input"));
    }

    #[test]
    fn step_silent_without_unsupported_flags() {
        let args = SrunArgs::try_parse_from(["srun", "hostname"]).expect("parse failed");
        assert!(step_unsupported_warnings(&args).is_empty());
    }

    /// Parse the way `main_with_args` does, so tests see the same `ArgMatches`
    /// the warning helpers consult for "did the user actually type this?".
    fn parse_srun(argv: &[&str]) -> (SrunArgs, ArgMatches) {
        let matches = SrunArgs::command()
            .try_get_matches_from(argv)
            .expect("parse failed");
        let args = SrunArgs::from_arg_matches(&matches).expect("from_arg_matches failed");
        (args, matches)
    }

    #[test]
    fn pty_step_warns_on_output_redirect() {
        let (args, matches) = parse_srun(&["srun", "--pty", "-o", "out.txt", "bash"]);
        let warnings = pty_step_unsupported_warnings(&args, &matches);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("--output/--error"));
    }

    #[test]
    fn pty_step_warns_when_sizing_flags_are_typed() {
        for argv in [
            &["srun", "--pty", "-n", "4", "bash"][..],
            &["srun", "--pty", "-N", "2", "bash"][..],
            &["srun", "--pty", "-c", "8", "bash"][..],
            &["srun", "--pty", "--ntasks-per-node", "2", "bash"][..],
        ] {
            let (args, matches) = parse_srun(argv);
            let warnings = pty_step_unsupported_warnings(&args, &matches);
            assert_eq!(warnings.len(), 1, "{argv:?}");
            assert!(warnings[0].contains("single task on one node"), "{argv:?}");
        }
    }

    /// salloc/sbatch export SPUR_NTASKS into every step, so warning off the
    /// resolved value would fire on a bare `srun --pty` the user did type.
    #[test]
    #[serial(env_injection)]
    fn pty_step_silent_when_sizing_came_from_the_allocation_env() {
        let _env = EnvGuard::new();
        std::env::set_var("SPUR_NTASKS", "4");
        std::env::set_var("SPUR_CPUS_PER_TASK", "8");

        let matches = SrunArgs::command()
            .try_get_matches_from(["srun", "--pty", "bash"])
            .expect("parse failed");
        let mut args = SrunArgs::from_arg_matches(&matches).expect("from_arg_matches failed");
        resolve_srun_env(&matches, &mut args).expect("env resolution failed");

        assert_eq!(args.ntasks, Some(4), "env must reach args");
        assert!(pty_step_unsupported_warnings(&args, &matches).is_empty());
    }

    #[test]
    fn pty_step_warns_on_env_scoped_flags() {
        let (args, matches) = parse_srun(&["srun", "--pty", "--cpu-bind", "cores", "bash"]);
        let warnings = pty_step_unsupported_warnings(&args, &matches);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("--cpu-bind/--gpu-bind/--label/--mpi"));
    }

    /// `--chdir` still sets the prolog/epilog working directory, so it gets its
    /// own wording rather than a blanket "ignored".
    #[test]
    fn pty_step_warns_that_chdir_does_not_move_the_pty() {
        let (args, matches) = parse_srun(&["srun", "--pty", "--chdir", "/tmp", "bash"]);
        let warnings = pty_step_unsupported_warnings(&args, &matches);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("--chdir does not move a --pty step"));
        assert!(warnings[0].contains("starts in the job's working directory"));
    }

    #[test]
    fn pty_step_warns_when_nodelist_names_several_nodes() {
        let (args, matches) = parse_srun(&["srun", "--pty", "-w", "n1,n2", "bash"]);
        let warnings = pty_step_unsupported_warnings(&args, &matches);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("first entry of --nodelist"));
    }

    /// A lost `\` continuation in a wrapped string literal leaves a run of
    /// spaces in output that no behavioral assertion would catch.
    #[test]
    fn warning_text_is_not_mangled_by_line_wrapping() {
        let (args, matches) = parse_srun(&[
            "srun", "--pty", "-w", "n1,n2", "-o", "o.txt", "-n", "2", "--chdir", "/tmp", "--label",
            "--input", "in.txt", "bash",
        ]);
        let all: Vec<String> = step_unsupported_warnings(&args)
            .into_iter()
            .chain(pty_step_unsupported_warnings(&args, &matches))
            .collect();
        // Every dropped-flag warning, so a lost continuation in any of them is
        // caught here.
        assert_eq!(all.len(), 6, "expected every warning to fire: {all:?}");
        for w in &all {
            assert!(!w.contains("  "), "collapsed continuation in: {w:?}");
            assert!(w.starts_with("srun: warning: "), "{w:?}");
        }
    }

    #[test]
    fn interactive_exit_code_prefers_the_session_result() {
        assert_eq!(interactive_exit_code(&Ok(0)), 0);
        assert_eq!(interactive_exit_code(&Ok(9)), 9);
        assert_eq!(
            interactive_exit_code(&Err(anyhow::anyhow!("session never opened"))),
            1
        );
    }

    #[test]
    fn pty_step_silent_for_plain_interactive_shell() {
        let (args, matches) = parse_srun(&["srun", "--pty", "bash"]);
        assert!(pty_step_unsupported_warnings(&args, &matches).is_empty());
    }

    #[test]
    fn step_dispatch_kind_routes_pty_to_interactive() {
        let args = SrunArgs::try_parse_from(["srun", "--pty", "-w", "n1,n2", "bash"])
            .expect("parse failed");
        assert_eq!(
            step_dispatch_kind(&args),
            StepDispatchKind::Interactive {
                node: "n1".to_string()
            }
        );
    }

    #[test]
    fn step_dispatch_kind_routes_plain_step_to_buffered() {
        let args = SrunArgs::try_parse_from(["srun", "hostname"]).expect("parse failed");
        assert_eq!(step_dispatch_kind(&args), StepDispatchKind::Buffered);
    }

    #[test]
    fn step_dispatch_kind_interactive_without_nodelist_leaves_node_empty() {
        let args = SrunArgs::try_parse_from(["srun", "--pty", "bash"]).expect("parse failed");
        assert_eq!(
            step_dispatch_kind(&args),
            StepDispatchKind::Interactive {
                node: String::new()
            }
        );
    }

    #[test]
    fn cpu_bind_validation_rejects_map_shorter_than_ntasks() {
        let mut env = HashMap::new();
        env.insert("SPUR_CPU_BIND".to_string(), "map_cpu:0,1".to_string());
        assert!(validate_step_cpu_bind(&env, 4).is_err());
    }

    #[test]
    fn first_node_reduces_comma_list_to_head() {
        assert_eq!(first_node("node001,node002"), "node001");
    }

    #[test]
    fn first_node_bare_name_passes_through() {
        assert_eq!(first_node("node007"), "node007");
    }

    #[test]
    fn first_node_trims_whitespace() {
        assert_eq!(first_node(" node003 , node004"), "node003");
    }

    #[test]
    fn first_node_empty_stays_empty() {
        assert_eq!(first_node(""), "");
    }

    #[test]
    fn parses_nodelist_and_exclude_short() {
        let args = SrunArgs::try_parse_from([
            "srun",
            "-w",
            "node001,node002",
            "-x",
            "node003",
            "hostname",
        ])
        .expect("parse failed");
        assert_eq!(args.nodelist.as_deref(), Some("node001,node002"));
        assert_eq!(args.exclude.as_deref(), Some("node003"));
    }

    #[test]
    fn parses_nodelist_and_exclude_long() {
        let args = SrunArgs::try_parse_from([
            "srun",
            "--nodelist",
            "node001",
            "--exclude",
            "node002",
            "hostname",
        ])
        .expect("parse failed");
        assert_eq!(args.nodelist.as_deref(), Some("node001"));
        assert_eq!(args.exclude.as_deref(), Some("node002"));
    }

    #[test]
    fn parses_nodefile_short_and_long() {
        let short = SrunArgs::try_parse_from(["srun", "-F", "nodes.txt", "hostname"])
            .expect("parse short nodefile");
        assert_eq!(short.nodefile.as_deref(), Some("nodes.txt"));
        assert!(short.nodelist.is_none());

        let long = SrunArgs::try_parse_from(["srun", "--nodefile=other.txt", "hostname"])
            .expect("parse long nodefile");
        assert_eq!(long.nodefile.as_deref(), Some("other.txt"));
        assert!(long.nodelist.is_none());
    }

    #[test]
    fn parses_output_error_input_short() {
        let args = SrunArgs::try_parse_from([
            "srun",
            "-o",
            "/tmp/out.txt",
            "-e",
            "/tmp/err.txt",
            "-i",
            "/tmp/in.txt",
            "hostname",
        ])
        .expect("parse failed");
        assert_eq!(args.output.as_deref(), Some("/tmp/out.txt"));
        assert_eq!(args.error.as_deref(), Some("/tmp/err.txt"));
        assert_eq!(args.input.as_deref(), Some("/tmp/in.txt"));
    }

    #[test]
    fn parses_output_error_input_long() {
        let args = SrunArgs::try_parse_from([
            "srun",
            "--output",
            "job-%j.out",
            "--error",
            "job-%j.err",
            "--input",
            "/dev/null",
            "hostname",
        ])
        .expect("parse failed");
        assert_eq!(args.output.as_deref(), Some("job-%j.out"));
        assert_eq!(args.error.as_deref(), Some("job-%j.err"));
        assert_eq!(args.input.as_deref(), Some("/dev/null"));
    }

    #[test]
    fn output_error_input_default_to_none() {
        let args = SrunArgs::try_parse_from(["srun", "hostname"]).expect("parse failed");
        assert!(args.output.is_none());
        assert!(args.error.is_none());
        assert!(args.input.is_none());
    }

    #[test]
    fn resolve_output_path_client_defaults() {
        assert_eq!(
            resolve_output_path_client("", 42, "/work"),
            "/work/spur-42.out"
        );
    }

    #[test]
    fn resolve_output_path_client_pattern() {
        assert_eq!(
            resolve_output_path_client("results-%j.log", 99, "/home/user"),
            "/home/user/results-99.log"
        );
    }

    #[test]
    fn resolve_output_path_client_absolute() {
        assert_eq!(
            resolve_output_path_client("/shared/output-%j.txt", 7, "/work"),
            "/shared/output-7.txt"
        );
    }

    #[test]
    fn resolve_io_paths_stderr_follows_stdout() {
        let args =
            SrunArgs::try_parse_from(["srun", "-o", "/tmp/out.txt", "hostname"]).expect("parse");
        let io = resolve_io_paths(&args);
        assert_eq!(io.stdout, "/tmp/out.txt");
        assert_eq!(io.stderr, "/tmp/out.txt");
        assert!(io.stdin.is_empty());
    }

    #[test]
    fn resolve_io_paths_explicit_stderr_overrides() {
        let args = SrunArgs::try_parse_from([
            "srun",
            "-o",
            "/tmp/out.txt",
            "-e",
            "/tmp/err.txt",
            "hostname",
        ])
        .expect("parse");
        let io = resolve_io_paths(&args);
        assert_eq!(io.stdout, "/tmp/out.txt");
        assert_eq!(io.stderr, "/tmp/err.txt");
    }

    #[test]
    fn resolve_io_paths_no_flags() {
        let args = SrunArgs::try_parse_from(["srun", "hostname"]).expect("parse");
        let io = resolve_io_paths(&args);
        assert!(io.stdout.is_empty());
        assert!(io.stderr.is_empty());
        assert!(io.stdin.is_empty());
    }

    /// The generated script must reconstruct the exact argv after one pass of
    /// shell parsing — i.e. metacharacters stay data, matching Slurm's direct
    /// exec of the argv (no top-level shell interpretation).
    fn assert_script_round_trips(argv: &[&str]) {
        let command: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
        let script = build_command_script(&command).expect("build script");
        let body = script
            .strip_prefix("#!/bin/bash\n")
            .expect("shebang")
            .trim_end();
        let reparsed = shlex::split(body).expect("generated script must be shell-parseable");
        assert_eq!(reparsed, command, "argv did not round-trip: {body:?}");
    }

    #[test]
    fn build_srun_job_spec_records_the_submit_line() {
        let args = SrunArgs::parse_from(["srun", "-w", "node1", "hostname"]);
        let io = resolve_io_paths(&args);
        let spec = build_srun_job_spec(&args, "/tmp/work", &io, "none", "srun -w node1 hostname")
            .expect("spec");
        assert_eq!(spec.submit_line, "srun -w node1 hostname");
    }

    #[test]
    fn build_srun_job_spec_sets_srun_job_and_command_script() {
        let args =
            SrunArgs::try_parse_from(["srun", "-N", "2", "-n", "4", "hostname"]).expect("parse");
        let io = ResolvedIoPaths {
            stdout: String::new(),
            stderr: String::new(),
            stdin: String::new(),
        };
        let spec =
            build_srun_job_spec(&args, "/tmp/work", &io, "none", "srun --test").expect("spec");
        assert!(spec.srun_job);
        assert_eq!(spec.num_nodes, 2);
        assert_eq!(spec.num_tasks, 4);
        assert_eq!(spec.work_dir, "/tmp/work");
        assert!(spec.script.starts_with("#!/bin/bash\n"));
        assert!(spec.script.contains("hostname"));
    }

    /// Launchers like PRTE's `plm:slurm` always pass `--ntasks-per-node`, so
    /// srun must accept both spellings rather than rejecting the command line.
    #[test]
    fn parses_ntasks_per_node() {
        let eq = SrunArgs::try_parse_from(["srun", "--ntasks-per-node=1", "hostname"])
            .expect("parse --ntasks-per-node=N");
        assert_eq!(eq.ntasks_per_node, Some(1));

        let spaced = SrunArgs::try_parse_from(["srun", "--ntasks-per-node", "8", "hostname"])
            .expect("parse --ntasks-per-node N");
        assert_eq!(spaced.ntasks_per_node, Some(8));

        let absent = SrunArgs::try_parse_from(["srun", "hostname"]).expect("parse");
        assert!(absent.ntasks_per_node.is_none());
    }

    #[test]
    fn build_srun_job_spec_sets_tasks_per_node() {
        let args = SrunArgs::try_parse_from(["srun", "--ntasks-per-node=4", "hostname"])
            .expect("parse failed");
        let io = ResolvedIoPaths {
            stdout: String::new(),
            stderr: String::new(),
            stdin: String::new(),
        };
        let spec = build_srun_job_spec(
            &args,
            "/tmp/work",
            &io,
            spur_core::mpi::MPI_NONE,
            "srun --test",
        )
        .expect("spec");
        assert_eq!(spec.tasks_per_node, 4);
    }

    #[test]
    fn build_srun_job_spec_tasks_per_node_defaults_zero() {
        let args = SrunArgs::try_parse_from(["srun", "hostname"]).expect("parse failed");
        let io = ResolvedIoPaths {
            stdout: String::new(),
            stderr: String::new(),
            stdin: String::new(),
        };
        let spec = build_srun_job_spec(
            &args,
            "/tmp/work",
            &io,
            spur_core::mpi::MPI_NONE,
            "srun --test",
        )
        .expect("spec");
        assert_eq!(spec.tasks_per_node, 0);
    }

    #[test]
    fn build_srun_job_spec_ntasks_defaults_to_node_count() {
        let args = SrunArgs::try_parse_from(["srun", "-N", "4", "hostname"]).expect("parse");
        assert_eq!(args.ntasks, None);
        let io = ResolvedIoPaths {
            stdout: String::new(),
            stderr: String::new(),
            stdin: String::new(),
        };
        let spec = build_srun_job_spec(
            &args,
            "/tmp/work",
            &io,
            spur_core::mpi::MPI_NONE,
            "srun --test",
        )
        .expect("spec");
        assert_eq!(spec.num_tasks, 4);
    }

    #[test]
    fn build_srun_job_spec_explicit_ntasks_overrides_node_default() {
        let args =
            SrunArgs::try_parse_from(["srun", "-N", "4", "-n", "2", "hostname"]).expect("parse");
        let io = ResolvedIoPaths {
            stdout: String::new(),
            stderr: String::new(),
            stdin: String::new(),
        };
        let spec = build_srun_job_spec(
            &args,
            "/tmp/work",
            &io,
            spur_core::mpi::MPI_NONE,
            "srun --test",
        )
        .expect("spec");
        assert_eq!(spec.num_tasks, 2);
    }

    #[test]
    fn parses_qos_short_and_long() {
        let short =
            SrunArgs::try_parse_from(["srun", "-q", "high", "hostname"]).expect("parse short qos");
        assert_eq!(short.qos.as_deref(), Some("high"));

        let long =
            SrunArgs::try_parse_from(["srun", "--qos=normal", "hostname"]).expect("parse long qos");
        assert_eq!(long.qos.as_deref(), Some("normal"));
    }

    #[test]
    fn build_srun_job_spec_sets_qos() {
        let args = SrunArgs::try_parse_from(["srun", "--qos", "high", "hostname"]).expect("parse");
        let io = ResolvedIoPaths {
            stdout: String::new(),
            stderr: String::new(),
            stdin: String::new(),
        };
        let spec = build_srun_job_spec(
            &args,
            "/tmp/work",
            &io,
            spur_core::mpi::MPI_NONE,
            "srun --test",
        )
        .expect("spec");
        assert_eq!(spec.qos, "high");
    }

    #[test]
    fn build_srun_job_spec_sets_exclusive() {
        let args = SrunArgs::try_parse_from(["srun", "--exclusive", "hostname"]).expect("parse");
        let io = ResolvedIoPaths {
            stdout: String::new(),
            stderr: String::new(),
            stdin: String::new(),
        };
        let spec = build_srun_job_spec(
            &args,
            "/tmp/work",
            &io,
            spur_core::mpi::MPI_NONE,
            "srun --test",
        )
        .expect("spec");
        assert!(spec.exclusive);
    }

    #[test]
    fn build_srun_job_spec_qos_defaults_empty() {
        let args = SrunArgs::try_parse_from(["srun", "hostname"]).expect("parse");
        let io = ResolvedIoPaths {
            stdout: String::new(),
            stderr: String::new(),
            stdin: String::new(),
        };
        let spec = build_srun_job_spec(
            &args,
            "/tmp/work",
            &io,
            spur_core::mpi::MPI_NONE,
            "srun --test",
        )
        .expect("spec");
        assert!(spec.qos.is_empty());
    }

    #[test]
    fn build_command_script_simple_argv() {
        assert_eq!(
            build_command_script(&["hostname".to_string()]).unwrap(),
            "#!/bin/bash\nhostname\n"
        );
    }

    #[test]
    fn build_command_script_preserves_spaces_in_arg() {
        // `srun touch "my file.txt"` must stay a single argument, not two.
        assert_script_round_trips(&["touch", "my file.txt"]);
    }

    #[test]
    fn build_command_script_keeps_shell_metacharacters_literal() {
        // Redirections, pipes, globs, and var refs are literal args, not shell
        // syntax — Slurm would pass them verbatim to the exec'd program.
        assert_script_round_trips(&["mytool", "a>b", "x|y", "*", "$HOME"]);
    }

    #[test]
    fn build_command_script_nested_bash_c_stays_one_arg() {
        // `srun bash -c 'echo A; echo B >&2'`: the `;` and `>&2` must remain
        // inside the single -c argument, not leak to the wrapper shell.
        assert_script_round_trips(&["bash", "-c", "echo A; echo B >&2"]);
    }

    #[test]
    fn build_command_script_preserves_quotes_in_arg() {
        assert_script_round_trips(&["echo", "it's a \"test\""]);
    }

    // These tests mutate process-global env vars, so they run serially and use
    // the shared EnvGuard, which clears every SPUR_/SLURM_/SALLOC_/SRUN_-prefixed
    // var on construction and drop, keeping them independent of the runner's
    // environment (which may inject SLURM_* vars) and of each other.
    use crate::env_defaults::EnvGuard;
    use serial_test::serial;

    /// Run the real matches-parse + env-resolve pipeline used by `main_with_args`.
    fn resolve_from(cli: &[&str]) -> SrunArgs {
        let matches = SrunArgs::command()
            .try_get_matches_from(cli)
            .expect("parse failed");
        let mut args = SrunArgs::from_arg_matches(&matches).expect("from_arg_matches failed");
        resolve_srun_env(&matches, &mut args).expect("resolve failed");
        args
    }

    #[test]
    #[serial(env_injection)]
    fn env_provides_default() {
        let env = EnvGuard::new();
        env.set("SLURM_PARTITION", "gpu");
        let args = resolve_from(&["srun", "hostname"]);
        assert_eq!(args.partition.as_deref(), Some("gpu"));
    }

    #[test]
    #[serial(env_injection)]
    fn cli_overrides_env() {
        let env = EnvGuard::new();
        env.set("SLURM_PARTITION", "gpu");
        let args = resolve_from(&["srun", "--partition=cpu", "hostname"]);
        assert_eq!(args.partition.as_deref(), Some("cpu"));
    }

    #[test]
    #[serial(env_injection)]
    fn spur_native_alias_works() {
        let env = EnvGuard::new();
        env.set("SPUR_PARTITION", "spur-part");
        let args = resolve_from(&["srun", "hostname"]);
        assert_eq!(args.partition.as_deref(), Some("spur-part"));
    }

    #[test]
    #[serial(env_injection)]
    fn spur_alias_precedes_slurm_twin() {
        let env = EnvGuard::new();
        env.set("SPUR_PARTITION", "from-spur");
        env.set("SLURM_PARTITION", "from-slurm");
        let args = resolve_from(&["srun", "hostname"]);
        assert_eq!(args.partition.as_deref(), Some("from-spur"));
    }

    #[test]
    #[serial(env_injection)]
    fn qos_env_default_and_cli_override() {
        let env = EnvGuard::new();
        env.set("SLURM_QOS", "high");
        assert_eq!(
            resolve_from(&["srun", "hostname"]).qos.as_deref(),
            Some("high")
        );

        // CLI wins over env.
        assert_eq!(
            resolve_from(&["srun", "--qos=low", "hostname"])
                .qos
                .as_deref(),
            Some("low")
        );
    }

    #[test]
    #[serial(env_injection)]
    fn qos_spur_alias_precedes_slurm_twin() {
        let env = EnvGuard::new();
        env.set("SPUR_QOS", "from-spur");
        env.set("SLURM_QOS", "from-slurm");
        assert_eq!(
            resolve_from(&["srun", "hostname"]).qos.as_deref(),
            Some("from-spur")
        );
    }

    /// An allocation exports QoS under the `*_JOB_QOS` twin, so srun inside a
    /// Spur allocation inherits it; the direct input var still takes precedence.
    #[test]
    #[serial(env_injection)]
    fn qos_inherits_allocation_twin() {
        let env = EnvGuard::new();
        env.set("SLURM_JOB_QOS", "alloc-qos");
        assert_eq!(
            resolve_from(&["srun", "hostname"]).qos.as_deref(),
            Some("alloc-qos")
        );

        env.set("SLURM_QOS", "input-qos");
        assert_eq!(
            resolve_from(&["srun", "hostname"]).qos.as_deref(),
            Some("input-qos")
        );

        let overridden = resolve_from(&["srun", "-q", "cli-qos", "hostname"]);
        assert_eq!(overridden.qos.as_deref(), Some("cli-qos"));
    }

    #[test]
    #[serial(env_injection)]
    fn nodes_alias_fallback_and_precedence() {
        let env = EnvGuard::new();
        env.set("SLURM_JOB_NUM_NODES", "3");
        assert_eq!(resolve_from(&["srun", "hostname"]).nodes, 3);

        // SLURM_NNODES is listed before SLURM_JOB_NUM_NODES, so it wins.
        env.set("SLURM_NNODES", "5");
        assert_eq!(resolve_from(&["srun", "hostname"]).nodes, 5);
    }

    #[test]
    #[serial(env_injection)]
    fn ntasks_nprocs_fallback() {
        let env = EnvGuard::new();
        env.set("SLURM_NPROCS", "7");
        assert_eq!(resolve_from(&["srun", "hostname"]).ntasks, Some(7));
    }

    #[test]
    #[serial(env_injection)]
    fn numeric_env_parsed() {
        let env = EnvGuard::new();
        env.set("SLURM_NNODES", "4");
        assert_eq!(resolve_from(&["srun", "hostname"]).nodes, 4);
    }

    #[test]
    #[serial(env_injection)]
    fn numeric_env_trims_whitespace() {
        let env = EnvGuard::new();
        env.set("SLURM_NTASKS", " 4 ");
        assert_eq!(resolve_from(&["srun", "hostname"]).ntasks, Some(4));
    }

    #[test]
    #[serial(env_injection)]
    fn invalid_numeric_env_errors() {
        let env = EnvGuard::new();
        env.set("SLURM_NNODES", "abc");
        let matches = SrunArgs::command()
            .try_get_matches_from(["srun", "hostname"])
            .expect("parse failed");
        let mut args = SrunArgs::from_arg_matches(&matches).expect("from_arg_matches failed");
        assert!(resolve_srun_env(&matches, &mut args).is_err());
    }

    #[test]
    #[serial(env_injection)]
    fn genuine_srun_vars() {
        let env = EnvGuard::new();
        env.set("SRUN_OUTPUT", "out-%j.log");
        env.set("SRUN_ERROR", "err-%j.log");
        env.set("SRUN_INPUT", "/dev/null");
        env.set("SRUN_CONTAINER", "img.sqsh");
        let args = resolve_from(&["srun", "hostname"]);
        assert_eq!(args.output.as_deref(), Some("out-%j.log"));
        assert_eq!(args.error.as_deref(), Some("err-%j.log"));
        assert_eq!(args.input.as_deref(), Some("/dev/null"));
        assert_eq!(args.container_image.as_deref(), Some("img.sqsh"));
    }

    #[test]
    #[serial(env_injection)]
    fn label_flag_semantics() {
        let env = EnvGuard::new();

        env.set("SLURM_LABELIO", "yes");
        assert!(resolve_from(&["srun", "hostname"]).label);

        env.set("SLURM_LABELIO", "1");
        assert!(resolve_from(&["srun", "hostname"]).label);

        env.set("SLURM_LABELIO", "0");
        assert!(!resolve_from(&["srun", "hostname"]).label);

        env.set("SLURM_LABELIO", "maybe");
        assert!(!resolve_from(&["srun", "hostname"]).label);
    }

    #[test]
    #[serial(env_injection)]
    fn gres_comma_split() {
        let env = EnvGuard::new();
        env.set("SLURM_GRES", "gpu:1,fpga:2");
        let args = resolve_from(&["srun", "hostname"]);
        assert_eq!(args.gres, vec!["gpu:1", "fpga:2"]);
    }

    #[test]
    #[serial(env_injection)]
    fn gres_env_trims_whitespace_and_drops_empties() {
        let env = EnvGuard::new();
        env.set("SLURM_GRES", "gpu:1, fpga:2 ,");
        let args = resolve_from(&["srun", "hostname"]);
        assert_eq!(args.gres, vec!["gpu:1", "fpga:2"]);
    }

    #[test]
    fn cli_gres_comma_list_splits_into_multiple() {
        let args = SrunArgs::try_parse_from(["srun", "--gres=gpu:1,fpga:2", "hostname"])
            .expect("parse failed");
        assert_eq!(args.gres, vec!["gpu:1", "fpga:2"]);
    }

    #[test]
    fn cli_repeated_gres_last_wins() {
        let args = SrunArgs::try_parse_from(["srun", "--gres=gpu:1", "--gres=fpga:2", "hostname"])
            .expect("parse failed");
        assert_eq!(args.gres, vec!["fpga:2"]);
    }

    #[test]
    #[serial(env_injection)]
    fn defaults_unchanged_without_env() {
        let _env = EnvGuard::new();
        let args = resolve_from(&["srun", "hostname"]);
        assert_eq!(args.nodes, 1);
        assert_eq!(args.ntasks, None);
        assert_eq!(args.cpus_per_task, 1);
        assert!(args.partition.is_none());
        assert!(args.mpi.is_none());
        assert!(!args.exclusive);
        assert!(!args.label);
    }

    /// srun launched inside an allocation inherits the allocation's task and
    /// CPU counts: salloc/sbatch export SLURM_NTASKS and SLURM_CPUS_PER_TASK,
    /// and srun reads them as defaults (matching Slurm). This is what feeds
    /// `create_job_step` in step mode, so the step sizes to the allocation
    /// rather than the bare `-n 1 -c 1` defaults.
    #[test]
    #[serial(env_injection)]
    fn inherits_allocation_task_and_cpu_counts() {
        let env = EnvGuard::new();
        env.set("SLURM_NTASKS", "4");
        env.set("SLURM_CPUS_PER_TASK", "2");
        env.set("SLURM_JOB_NUM_NODES", "3");
        let args = resolve_from(&["srun", "hostname"]);
        assert_eq!(args.ntasks, Some(4));
        assert_eq!(args.cpus_per_task, 2);
        assert_eq!(args.nodes, 3);

        // An explicit CLI value still wins over the inherited allocation env.
        let overridden = resolve_from(&["srun", "-n", "1", "hostname"]);
        assert_eq!(overridden.ntasks, Some(1));
    }

    #[test]
    #[serial(env_injection)]
    fn explicit_nodes_keeps_inherited_allocation_ntasks() {
        let env = EnvGuard::new();
        env.set("SLURM_NTASKS", "4");
        env.set("SLURM_JOB_NUM_NODES", "4");

        let args = resolve_from(&["srun", "-N", "1", "hostname"]);
        assert_eq!(args.nodes, 1);
        assert_eq!(args.ntasks, Some(4));
        assert_eq!(
            crate::sbatch::effective_ntasks(args.ntasks, args.ntasks_per_node, args.nodes),
            4
        );
    }

    /// Outside an allocation there is no `SLURM_NTASKS` to inherit, so the task
    /// count follows the node count — including when `-N` itself came from env.
    #[test]
    #[serial(env_injection)]
    fn ntasks_defaults_to_node_count_from_env() {
        let env = EnvGuard::new();
        env.set("SLURM_NNODES", "3");
        let args = resolve_from(&["srun", "hostname"]);
        assert_eq!(args.ntasks, None);
        assert_eq!(
            crate::sbatch::effective_ntasks(args.ntasks, None, args.nodes),
            3
        );
    }

    #[test]
    #[serial(env_injection)]
    fn invalid_ntasks_env_errors() {
        let env = EnvGuard::new();
        env.set("SLURM_NTASKS", "abc");
        let matches = SrunArgs::command()
            .try_get_matches_from(["srun", "hostname"])
            .expect("parse failed");
        let mut args = SrunArgs::from_arg_matches(&matches).expect("from_arg_matches failed");
        assert!(resolve_srun_env(&matches, &mut args).is_err());
    }

    /// `--ntasks-per-node` sizes the job when `-n` is absent:
    /// `srun --ntasks-per-node=1 -N 4` launches one task per node.
    #[test]
    #[serial(env_injection)]
    fn ntasks_per_node_scales_task_count() {
        let _env = EnvGuard::new();
        let args = resolve_from(&["srun", "-N", "4", "--ntasks-per-node=1", "hostname"]);
        assert_eq!(args.ntasks, None);
        assert_eq!(args.ntasks_per_node, Some(1));
        assert_eq!(
            crate::sbatch::effective_ntasks(args.ntasks, args.ntasks_per_node, args.nodes),
            4
        );

        let multi = resolve_from(&["srun", "-N", "2", "--ntasks-per-node=8", "hostname"]);
        assert_eq!(
            crate::sbatch::effective_ntasks(multi.ntasks, multi.ntasks_per_node, multi.nodes),
            16
        );

        // Zero per-node is treated as unset, so the count falls back to nodes.
        let zero = resolve_from(&["srun", "-N", "4", "--ntasks-per-node=0", "hostname"]);
        assert_eq!(
            crate::sbatch::effective_ntasks(zero.ntasks, zero.ntasks_per_node, zero.nodes),
            4
        );
    }

    #[test]
    #[serial(env_injection)]
    fn explicit_ntasks_overrides_ntasks_per_node() {
        let env = EnvGuard::new();
        let cli = resolve_from(&[
            "srun",
            "-N",
            "4",
            "--ntasks-per-node=8",
            "-n",
            "3",
            "hostname",
        ]);
        assert_eq!(
            crate::sbatch::effective_ntasks(cli.ntasks, cli.ntasks_per_node, cli.nodes),
            3
        );

        // An inherited allocation task count is a direct request, so it wins
        // over the per-node derivation. Typed `-N` still selects node count.
        env.set("SLURM_NTASKS", "5");
        let inherited = resolve_from(&["srun", "-N", "4", "--ntasks-per-node=8", "hostname"]);
        assert_eq!(
            crate::sbatch::effective_ntasks(
                inherited.ntasks,
                inherited.ntasks_per_node,
                inherited.nodes
            ),
            5
        );
    }

    /// Allocation context is exported under the `*_JOB_*` names
    /// (SLURM_JOB_PARTITION / SLURM_JOB_ACCOUNT), so srun must read those to
    /// inherit partition/account when run inside a Spur allocation.
    #[test]
    #[serial(env_injection)]
    fn inherits_allocation_partition_and_account() {
        let env = EnvGuard::new();
        env.set("SLURM_JOB_PARTITION", "alloc-part");
        env.set("SLURM_JOB_ACCOUNT", "alloc-acct");
        let args = resolve_from(&["srun", "hostname"]);
        assert_eq!(args.partition.as_deref(), Some("alloc-part"));
        assert_eq!(args.account.as_deref(), Some("alloc-acct"));

        // The direct input var takes precedence over the allocation twin.
        env.set("SLURM_PARTITION", "input-part");
        let direct = resolve_from(&["srun", "hostname"]);
        assert_eq!(direct.partition.as_deref(), Some("input-part"));

        // An explicit CLI value still wins over both.
        let overridden = resolve_from(&["srun", "-p", "cli-part", "hostname"]);
        assert_eq!(overridden.partition.as_deref(), Some("cli-part"));
    }

    fn empty_io() -> ResolvedIoPaths {
        ResolvedIoPaths {
            stdout: String::new(),
            stderr: String::new(),
            stdin: String::new(),
        }
    }

    async fn dispatch_with(
        client: &mut SlurmControllerClient<crate::authclient::AuthChannel>,
        cli: &[&str],
    ) -> Result<StepDispatchResult> {
        let matches = SrunArgs::command()
            .try_get_matches_from(cli)
            .expect("parse failed");
        let mut args = SrunArgs::from_arg_matches(&matches).expect("parse failed");
        resolve_srun_env(&matches, &mut args).expect("resolve failed");
        let io = empty_io();
        let params = StepDispatchParams {
            args: &args,
            explicit_num_nodes: was_cli_set(&matches, "nodes").then_some(args.nodes),
            requested_nodelist: args.nodelist.as_deref(),
            work_dir: "/tmp",
            io: &io,
            mpi: spur_core::mpi::MPI_NONE,
            user: "tester",
        };
        dispatch_step(client, 1, &params).await
    }

    /// A session that dies after CreateJobStep must still close the step out —
    /// nothing else completes an interactive step before the job ends.
    #[tokio::test]
    #[serial(env_injection)]
    async fn interactive_step_reports_completion_when_session_fails() {
        let _env = EnvGuard::new();
        let (addr, capture) = crate::mock_controller::spawn().await;
        let mut client = crate::mock_controller::client(addr).await;

        let err = run_interactive_pty(
            &mut client,
            1,
            vec!["bash".to_string()],
            String::new(),
            "tester",
            None,
        )
        .await
        .expect_err("mock returns no node address, so the session cannot open");

        assert!(err.to_string().contains("node address"), "{err}");
        assert_eq!(
            capture.complete_step_calls(),
            vec![(crate::mock_controller::MOCK_STEP_ID, 1)],
            "a failed session must still report the step complete"
        );
    }

    /// A create that never succeeds must not report a completion for a step
    /// that does not exist.
    #[tokio::test]
    #[serial(env_injection)]
    async fn interactive_step_reports_nothing_when_no_step_was_created() {
        let _env = EnvGuard::new();
        let (addr, capture) = crate::mock_controller::spawn().await;
        capture.set_create_step_error(tonic::Code::PermissionDenied);
        let mut client = crate::mock_controller::client(addr).await;

        run_interactive_pty(
            &mut client,
            1,
            vec!["bash".into()],
            String::new(),
            "tester",
            None,
        )
        .await
        .expect_err("CreateJobStep was rejected");

        assert!(
            capture.complete_step_calls().is_empty(),
            "no step exists, so nothing may be reported complete"
        );
    }

    /// With no `-n`, the step must ask the controller for one task per node.
    #[tokio::test]
    #[serial(env_injection)]
    async fn dispatch_step_requests_one_task_per_node_by_default() {
        let _env = EnvGuard::new();
        let (addr, capture) = crate::mock_controller::spawn().await;
        let mut client = crate::mock_controller::client(addr).await;

        let result = dispatch_with(&mut client, &["srun", "-N", "3", "hostname"])
            .await
            .expect("dispatch should succeed");

        assert_eq!(capture.create_step_num_tasks(), 3);
        assert_eq!(capture.create_step_num_nodes(), 3);
        assert_eq!(capture.run_step_calls(), 1);
        assert_eq!(
            capture.run_step_step_id(),
            crate::mock_controller::MOCK_STEP_ID,
            "RunStep must carry the step id returned by CreateJobStep"
        );
        assert_eq!(result.exit_code, crate::mock_controller::MOCK_EXIT_CODE);
    }

    #[tokio::test]
    #[serial(env_injection)]
    async fn dispatch_step_explicit_ntasks_overrides_node_default() {
        let _env = EnvGuard::new();
        let (addr, capture) = crate::mock_controller::spawn().await;
        let mut client = crate::mock_controller::client(addr).await;

        dispatch_with(&mut client, &["srun", "-N", "3", "-n", "2", "hostname"])
            .await
            .expect("dispatch should succeed");

        assert_eq!(capture.create_step_num_tasks(), 2);
        assert_eq!(capture.create_step_num_nodes(), 3);
    }

    #[tokio::test]
    #[serial(env_injection)]
    async fn dispatch_step_forwards_nodelist() {
        let _env = EnvGuard::new();
        let (addr, capture) = crate::mock_controller::spawn().await;
        let mut client = crate::mock_controller::client(addr).await;

        dispatch_with(
            &mut client,
            &["srun", "-w", "node[002-003]", "-n", "2", "hostname"],
        )
        .await
        .expect("dispatch should succeed");

        assert_eq!(capture.create_step_num_nodes(), 0);
        assert_eq!(capture.create_step_nodelist(), "node[002-003]");
    }

    #[tokio::test]
    #[serial(env_injection)]
    async fn dispatch_step_typed_nodes_keep_inherited_ntasks() {
        let env = EnvGuard::new();
        env.set("SLURM_NTASKS", "4");
        let (addr, capture) = crate::mock_controller::spawn().await;
        let mut client = crate::mock_controller::client(addr).await;

        dispatch_with(&mut client, &["srun", "-N", "1", "hostname"])
            .await
            .expect("dispatch should succeed");

        assert_eq!(capture.create_step_num_tasks(), 4);
        assert_eq!(capture.create_step_num_nodes(), 1);
    }

    /// The per-node default feeds cpu-bind validation, so a `map_cpu` list
    /// shorter than the node count is rejected before the step is run.
    #[tokio::test]
    #[serial(env_injection)]
    async fn dispatch_step_rejects_map_cpu_bind_shorter_than_default_ntasks() {
        let _env = EnvGuard::new();
        let (addr, capture) = crate::mock_controller::spawn().await;
        let mut client = crate::mock_controller::client(addr).await;

        let err = dispatch_with(
            &mut client,
            &["srun", "-N", "4", "--cpu-bind", "map_cpu:0,1", "hostname"],
        )
        .await
        .expect_err("two CPUs cannot cover four tasks");

        let msg = format!("{err:#}");
        assert!(
            msg.contains("map_cpu lists 2 CPU(s) but the step requires 4 task(s)"),
            "expected a cpu-bind arity error, got: {msg}"
        );
        assert_eq!(capture.create_step_num_tasks(), 4);
        assert_eq!(
            capture.run_step_calls(),
            0,
            "dispatch must bail before running the step"
        );
    }

    #[tokio::test]
    #[serial(env_injection)]
    async fn dispatch_step_surfaces_create_job_step_failure() {
        let _env = EnvGuard::new();
        let addr = crate::mock_controller::unreachable_addr().await;
        let mut client = crate::mock_controller::lazy_client(addr);

        let err = dispatch_with(&mut client, &["srun", "-N", "2", "hostname"])
            .await
            .expect_err("no server is listening");

        let msg = format!("{err:#}");
        assert!(
            msg.contains("failed to create job step"),
            "expected a CreateJobStep failure, got: {msg}"
        );
    }

    #[tokio::test]
    async fn jobid_without_overlap_errors() {
        let result = main_with_args(vec![
            "srun".into(),
            "--jobid".into(),
            "42".into(),
            "hostname".into(),
        ])
        .await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("--overlap"),
            "expected --overlap error, got: {msg}"
        );
    }

    /// --container-readonly is a no-op in the runtime today, so it is refused at
    /// submission rather than silently ignored (the #777-class failure mode).
    #[tokio::test]
    async fn container_readonly_is_rejected() {
        let result = main_with_args(vec![
            "srun".into(),
            "--container-image=img.sqsh".into(),
            "--container-readonly".into(),
            "hostname".into(),
        ])
        .await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("--container-readonly") && msg.contains("not yet implemented"),
            "expected a not-yet-implemented rejection, got: {msg}"
        );
    }

    /// The controller reports `nodelist` as a compressed hostlist, so streaming
    /// has to expand it. Splitting `node[001-002]` on commas yields the whole
    /// bracket literal, which is not a host, and the stream then goes nowhere.
    #[tokio::test]
    async fn try_stream_output_asks_the_controller_for_an_expanded_hostname() {
        let (addr, capture) = crate::mock_controller::spawn().await;
        let mut client = crate::mock_controller::client(addr).await;

        let streamed = try_stream_output(&mut client, "node[001-002]", 42, "tester").await;

        assert!(
            !streamed,
            "the mock reports every node missing, so streaming is unavailable"
        );
        assert_eq!(capture.get_node_requests(), vec!["node001".to_string()]);
    }
}
