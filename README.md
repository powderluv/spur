# Spur

An AI-native job scheduler written in Rust. Drop-in compatible with Slurm's CLI, REST API, and C FFI while providing WireGuard mesh networking, GPU-first scheduling, and modern state management.

## Quick Start

**See [docs/quickstart.md](docs/quickstart.md) for the full walkthrough** — single-node setup in 5 minutes, multi-node with WireGuard mesh, GPU job examples, and troubleshooting.

### Build

```bash
# Prerequisites
sudo apt install protobuf-compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
cargo build --release

# Run tests
cargo test
```

### Binaries

| Binary | Description |
|--------|-------------|
| `spur` | CLI (multi-call binary) |
| `spurctld` | Controller daemon |
| `spurd` | Node agent daemon |
| `spurdbd` | Accounting daemon (PostgreSQL) |
| `spurrestd` | REST API daemon |
| `libspur_compat.so` | C FFI shim |

### Start a Cluster

```bash
# 1. Create config
mkdir -p /etc/spur
cat > /etc/spur/spur.conf <<'EOF'
cluster_name = "my-cluster"

[[partitions]]
name = "default"
default = true
nodes = "node[001-008]"
max_time = "24:00:00"

[[nodes]]
names = "node[001-008]"
cpus = 64
memory_mb = 256000
EOF

# 2. Start controller
spurctld -D --state-dir /var/spool/spur

# 3. Start node agent (on each compute node)
spurd -D --controller http://controller:6817

# 4. Optionally start REST API and accounting
spurrestd --controller http://controller:6817
spurdbd --database-url postgresql://spur:spur@localhost/spur --migrate
```

## CLI Usage

### Native Commands

```bash
spur submit job.sh          # Submit a batch job
spur queue                  # View job queue
spur cancel 12345           # Cancel a job
spur nodes                  # View cluster nodes
spur history                # View job history (accounting)
spur show job 12345         # Detailed job info
spur show node node001      # Detailed node info
spur show partition gpu     # Detailed partition info
spur version                # Show version
```

### Slurm-Compatible Commands

Works via symlinks or subcommands — existing scripts and muscle memory work unchanged:

```bash
# Via symlinks (create once)
ln -s $(which spur) /usr/local/bin/sbatch
ln -s $(which spur) /usr/local/bin/squeue
ln -s $(which spur) /usr/local/bin/scancel
ln -s $(which spur) /usr/local/bin/sinfo
ln -s $(which spur) /usr/local/bin/sacct
ln -s $(which spur) /usr/local/bin/scontrol

# Then use as normal
sbatch --job-name=train -N4 --gres=gpu:mi300x:8 train.sh
squeue -u $USER
scancel 12345
sinfo -N
sacct -u $USER --starttime=2024-01-01

# Or as subcommands
spur sbatch job.sh
spur squeue -u alice
```

### #SBATCH Directives

Batch scripts work exactly like Slurm:

```bash
#!/bin/bash
#SBATCH --job-name=training
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=4:00:00
#SBATCH --partition=gpu

torchrun --nproc_per_node=8 train.py
```

`#PBS` directives are also parsed for PBS/Torque migration.

## REST API

Spur serves two API endpoints:

- **`/api/v1/`** — Native Spur API
- **`/slurm/v0.0.42/`** — Slurm-compatible API (drop-in for slurmrestd clients)

```bash
# Submit a job
curl -X POST http://localhost:6820/api/v1/job/submit \
  -H "Content-Type: application/json" \
  -d '{"job": {"name": "test", "partition": "gpu", "nodes": 2, "script": "#!/bin/bash\necho hello"}}'

# List jobs
curl http://localhost:6820/api/v1/jobs

# Cluster health
curl http://localhost:6820/api/v1/ping
```

## C FFI (libspur_compat.so)

Drop-in replacement for `libslurm.so`. Exports Slurm-compatible C symbols:

```c
#include <slurm/slurm.h>  // Use Slurm headers

job_desc_msg_t desc;
slurm_init_job_desc_msg(&desc);
desc.name = "my_job";
desc.script = "#!/bin/bash\necho hello";

uint32_t job_id;
slurm_submit_batch_job(&desc, &job_id);
printf("Submitted job %u\n", job_id);
```

**Exported functions:** `slurm_init_job_desc_msg`, `slurm_submit_batch_job`, `slurm_load_jobs`, `slurm_free_job_info_msg`, `slurm_load_node`, `slurm_load_partitions`, `slurm_kill_job`

## Architecture

```
                    ┌─────────────────────────────┐
                    │        API Surface           │
                    │  CLI (spur / sbatch / srun)  │
                    │  REST (spurrestd)            │
                    │  C FFI (libspur_compat.so)   │
                    └──────────────┬───────────────┘
                                   │ gRPC
                    ┌──────────────▼───────────────┐
                    │      spurctld (Rust)          │
                    │  ┌─────────┐ ┌────────────┐  │
                    │  │Backfill │ │  WAL +      │  │
                    │  │Scheduler│ │  Snapshots  │  │
                    │  │         │ │  (redb)     │  │
                    │  └─────────┘ └────────────┘  │
                    └──────┬───────────────┬───────┘
                           │               │
              ┌────────────▼──┐    ┌───────▼────────┐
              │  spurd (Rust)  │    │ spurdbd (Rust)  │
              │  Node agent    │    │ Accounting      │
              │  - cgroups v2  │    │ - PostgreSQL    │
              │  - GPU enum    │    │ - Fair-share    │
              │  - fork/exec   │    │ - Job history   │
              └────────────────┘    └─────────────────┘
```

## Configuration

TOML format (`/etc/spur/spur.conf`):

```toml
cluster_name = "production"

[controller]
listen_addr = "[::]:6817"
state_dir = "/var/spool/spur"
hosts = ["ctrl1", "ctrl2"]

[scheduler]
plugin = "backfill"
interval_secs = 1
fairshare_halflife_days = 14

[accounting]
database_url = "postgresql://spur:spur@db1/spur"

[[partitions]]
name = "gpu"
default = true
nodes = "gpu[001-064]"
max_time = "72:00:00"

[[partitions]]
name = "cpu"
nodes = "cpu[001-256]"
max_time = "168:00:00"

[[nodes]]
names = "gpu[001-064]"
cpus = 128
memory_mb = 512000
gres = ["gpu:mi300x:8"]

[[nodes]]
names = "cpu[001-256]"
cpus = 256
memory_mb = 1024000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPUR_CONTROLLER_ADDR` | `http://localhost:6817` | Controller gRPC address |
| `SPUR_ACCOUNTING_ADDR` | `http://localhost:6819` | Accounting gRPC address |

## Project Structure

```
spur/
├── proto/slurm.proto           # gRPC service definitions
├── docs/quickstart.md          # Getting started guide
├── crates/
│   ├── spur-proto/             # Generated gRPC code
│   ├── spur-core/              # Job, Node, ResourceSet, config, hostlist, WAL
│   ├── spur-net/               # WireGuard mesh networking, address detection
│   ├── spur-sched/             # Backfill scheduler, priority, timeline
│   ├── spur-state/             # WAL persistence, redb snapshots
│   ├── spurctld/               # Controller daemon
│   ├── spurd/                  # Node agent daemon
│   ├── spurdbd/                # Accounting daemon
│   ├── spurrestd/              # REST API daemon
│   ├── spur-cli/               # CLI binary (multi-call: spur, sbatch, squeue, ...)
│   ├── spur-ffi/               # C FFI shim (libspur_compat.so)
│   ├── spur-spank/             # SPANK plugin host
│   ├── spur-k8s/               # K8s integration (post-MVP)
│   └── spur-tests/             # Test suite (mirrors Slurm numbering)
```

## Testing

```bash
cargo test                    # Run all 314 tests
cargo test -p spur-tests      # Run integration test suite only
cargo test -p spur-core       # Run core library tests only
```

Test groups mirror Slurm's testsuite numbering for 1-1 mapping:

| Group | Coverage |
|-------|----------|
| t05 | Job queue filtering and display |
| t07 | Scheduler (backfill, timelines, partition filtering) |
| t17 | Job submission (directives, defaults, arrays, deps) |
| t24 | Priority and fair-share |
| t28 | Job arrays |
| t50 | Core types (state machine, resources, GRES) |
| t51 | Hostlist expansion/compression |
| t52 | Configuration parsing |
| t53 | WAL and snapshot persistence |
| t55 | Output format conformance |

## License

Apache-2.0
