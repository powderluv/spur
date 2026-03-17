mod format_engine;
mod net;
mod sacct;
mod sacctmgr;
mod sbatch;
mod scancel;
mod scontrol;
mod sinfo;
mod squeue;
mod srun;

use std::path::Path;

fn main() -> anyhow::Result<()> {
    // Multi-call binary: dispatch based on argv[0] (symlink name).
    let argv0 = std::env::args()
        .next()
        .unwrap_or_else(|| "spur".into());
    let bin_name = Path::new(&argv0)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("spur");

    let runtime = tokio::runtime::Runtime::new()?;

    // Slurm-compatible symlink dispatch (backward compat)
    match bin_name {
        "sbatch" => return runtime.block_on(sbatch::main()),
        "srun" => return runtime.block_on(srun::main()),
        "squeue" => return runtime.block_on(squeue::main()),
        "scancel" => return runtime.block_on(scancel::main()),
        "sinfo" => return runtime.block_on(sinfo::main()),
        "sacct" => return runtime.block_on(sacct::main()),
        "sacctmgr" => return runtime.block_on(sacctmgr::main()),
        "scontrol" => return runtime.block_on(scontrol::main()),
        _ => {}
    }

    // Native spur CLI: `spur <command> [args...]`
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        // Native spur commands
        "net" => runtime.block_on(net::main()),
        "submit" => runtime.block_on(sbatch::main()),
        "run" => runtime.block_on(srun::main()),
        "queue" | "jobs" => runtime.block_on(squeue::main()),
        "cancel" | "kill" => runtime.block_on(scancel::main()),
        "nodes" | "info" => runtime.block_on(sinfo::main()),
        "history" | "acct" => runtime.block_on(sacct::main()),
        "accounts" | "acctmgr" => runtime.block_on(sacctmgr::main()),
        "show" | "control" | "ctl" => runtime.block_on(scontrol::main()),

        // Slurm-compatible subcommands (for migration)
        "sbatch" => runtime.block_on(sbatch::main()),
        "srun" => runtime.block_on(srun::main()),
        "squeue" => runtime.block_on(squeue::main()),
        "scancel" => runtime.block_on(scancel::main()),
        "sinfo" => runtime.block_on(sinfo::main()),
        "sacct" => runtime.block_on(sacct::main()),
        "sacctmgr" => runtime.block_on(sacctmgr::main()),
        "scontrol" => runtime.block_on(scontrol::main()),

        "version" | "--version" | "-V" => {
            println!("spur {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        other => {
            eprintln!("spur: unknown command '{}'", other);
            eprintln!();
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("spur — HPC job scheduler");
    eprintln!();
    eprintln!("Usage: spur <command> [args...]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  net         Manage WireGuard mesh network (init/join/status)");
    eprintln!("  submit      Submit a batch job script");
    eprintln!("  run         Run a parallel job (interactive)");
    eprintln!("  queue       View the job queue");
    eprintln!("  cancel      Cancel pending or running jobs");
    eprintln!("  nodes       View cluster node information");
    eprintln!("  history     View job accounting history");
    eprintln!("  accounts    Manage accounts, users, and QOS");
    eprintln!("  show        Show detailed job/node/partition info");
    eprintln!("  version     Show version");
    eprintln!();
    eprintln!("Slurm-compatible aliases (also work as symlinks):");
    eprintln!("  sbatch srun squeue scancel sinfo sacct sacctmgr scontrol");
}
