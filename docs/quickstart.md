# Spur Quickstart Guide

This guide walks you through building Spur from source, running a single-node cluster on your laptop, then expanding to a multi-node GPU cluster with WireGuard mesh networking.

## Prerequisites

**Required:**

- Linux (kernel 5.6+ for in-kernel WireGuard; any recent Ubuntu/Fedora/RHEL works)
- Rust toolchain (1.75+)
- Protocol Buffers compiler

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install protobuf compiler
sudo apt install protobuf-compiler    # Debian/Ubuntu
sudo dnf install protobuf-compiler    # Fedora/RHEL
```

**Optional (for multi-node networking):**

```bash
sudo apt install wireguard-tools      # WireGuard CLI (wg, wg-quick)
```

## Build

```bash
git clone https://github.com/ROCm/spur.git
cd spur
cargo build --release
```

Binaries are in `target/release/`:

| Binary | Role |
|--------|------|
| `spur` | CLI — submit jobs, check status, manage cluster |
| `spurctld` | Controller daemon — scheduling, state management |
| `spurd` | Node agent — runs on every compute node |
| `spurdbd` | Accounting daemon (optional, needs PostgreSQL) |
| `spurrestd` | REST API (optional) |

## 1. Single-Node Cluster (5 minutes)

The fastest way to try Spur. Everything runs on one machine, no config file needed.

### Start the controller

```bash
# Create state directory
mkdir -p /tmp/spur-state

# Start controller in foreground (uses built-in defaults)
./target/release/spurctld -D --state-dir /tmp/spur-state --log-level info
```

The controller listens on port 6817 and creates a single "default" partition.

### Start the agent (new terminal)

```bash
./target/release/spurd -D --controller http://localhost:6817
```

The agent auto-discovers local CPUs, memory, and GPUs, then registers with the controller. You should see:

```
INFO spurd: resources discovered cpus=16 memory_mb=32000 gpus=0
INFO spurd::reporter: registered with controller
```

### Submit a job (new terminal)

```bash
# Create a simple job script
cat > /tmp/hello.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --time=00:01:00

echo "Hello from Spur!"
echo "Running on $(hostname) with $SPUR_CPUS_ON_NODE CPUs"
sleep 2
echo "Done."
EOF

# Submit it
./target/release/spur submit /tmp/hello.sh
```

Output: `Submitted batch job 1`

### Check status

```bash
# View the job queue
./target/release/spur queue

# View nodes
./target/release/spur nodes

# View detailed job info
./target/release/spur show job 1

# Check job output (default location)
cat /tmp/spur-1.out
```

### Run an interactive command

```bash
# srun-style: run a command directly
./target/release/spur run hostname
./target/release/spur run -- bash -c "echo I have \$SPUR_CPUS_ON_NODE CPUs"
```

### Slurm compatibility

If you're migrating from Slurm, create symlinks and use the familiar commands:

```bash
cd target/release
for cmd in sbatch srun squeue scancel sinfo sacct scontrol; do
    ln -sf spur $cmd
done

# Now use Slurm commands directly
./sbatch /tmp/hello.sh
./squeue
./sinfo -N
./scancel 2
```

## 2. Multi-Node Cluster with WireGuard

For a real cluster across multiple machines. WireGuard creates an encrypted mesh so all nodes can reach each other regardless of firewalls or NAT.

### Cluster layout example

```
controller (192.168.1.100)  →  WireGuard 10.44.0.1
gpu-node-1 (192.168.1.101)  →  WireGuard 10.44.0.2
gpu-node-2 (192.168.1.102)  →  WireGuard 10.44.0.3
```

### Step 1: Set up WireGuard mesh

**On the controller:**

```bash
# Initialize the mesh — generates keys, assigns 10.44.0.1, brings up spur0
sudo spur net init --cidr 10.44.0.0/16 --port 51820
```

This prints the server public key and a join command template. Save the public key.

**On each compute node:**

```bash
# Join the mesh — generates local keys, connects to controller
sudo spur net join \
    --endpoint 192.168.1.100:51820 \
    --server-key <controller-pubkey> \
    --address 10.44.0.2

# Then add this node as a peer on the controller
# (run on controller, using the pubkey printed by the join command)
sudo spur net add-peer \
    --key <node-pubkey> \
    --allowed-ip 10.44.0.2/32 \
    --endpoint 192.168.1.101:51820
```

Repeat for each node, incrementing the address (10.44.0.3, 10.44.0.4, ...).

**Verify connectivity:**

```bash
# On any node
spur net status           # Shows WireGuard peers and handshake times
ping 10.44.0.1            # Ping the controller through the mesh
```

### Step 2: Write a config file

Create `/etc/spur/spur.conf` (same file on all nodes, or distribute via your config management):

```toml
cluster_name = "gpu-cluster"

[controller]
listen_addr = "[::]:6817"
hosts = ["10.44.0.1"]
state_dir = "/var/spool/spur"

[scheduler]
plugin = "backfill"
interval_secs = 1

[network]
wg_enabled = true
wg_interface = "spur0"
agent_port = 6818

[[partitions]]
name = "gpu"
default = true
nodes = "gpu-node-[1-2]"
max_time = "72:00:00"

[[nodes]]
names = "gpu-node-[1-2]"
cpus = 128
memory_mb = 512000
gres = ["gpu:mi300x:8"]
```

### Step 3: Start the daemons

**On the controller (10.44.0.1):**

```bash
sudo mkdir -p /var/spool/spur
spurctld -D -f /etc/spur/spur.conf
```

**On each compute node:**

```bash
spurd -D \
    --controller http://10.44.0.1:6817 \
    --hostname gpu-node-1 \
    --listen [::]:6818
```

The agent detects the `spur0` WireGuard interface and registers with its 10.44.x.y address. The controller will dispatch jobs to that address.

### Step 4: Submit multi-node jobs

```bash
# 2-node job
spur submit --nodes=2 --ntasks-per-node=8 train.sh

# Or with SBATCH directives
cat > train.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=distributed-training
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=4:00:00

echo "Node: $(hostname)"
echo "Peers: $SPUR_PEER_NODES"
echo "Task offset: $SPUR_TASK_OFFSET"
echo "Total nodes: $SPUR_NUM_NODES"

torchrun \
    --nnodes=$SPUR_NUM_NODES \
    --node_rank=$SPUR_TASK_OFFSET \
    --master_addr=$(echo $SPUR_PEER_NODES | cut -d: -f1) \
    --master_port=29500 \
    --nproc_per_node=8 \
    train.py
EOF

spur submit train.sh
```

When a multi-node job is dispatched, each node receives:

| Environment Variable | Example | Description |
|---------------------|---------|-------------|
| `SPUR_JOB_ID` | `42` | Job ID |
| `SPUR_NUM_NODES` | `2` | Total nodes in allocation |
| `SPUR_TASK_OFFSET` | `0` or `8` | This node's starting task index |
| `SPUR_PEER_NODES` | `10.44.0.2:6818,10.44.0.3:6818` | All nodes in the allocation |
| `SPUR_CPUS_ON_NODE` | `128` | CPUs allocated on this node |

## 3. GPU Job Examples

### AMD MI300X

```bash
cat > gpu-test.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=rocm-test
#SBATCH -N 1
#SBATCH --gres=gpu:mi300x:4
#SBATCH --time=00:10:00

rocm-smi
hipcc -o /tmp/vectoradd vectoradd.cpp && /tmp/vectoradd
EOF

spur submit gpu-test.sh
```

Spur sets `ROCR_VISIBLE_DEVICES` to restrict GPU visibility.

### NVIDIA H100

```bash
cat > cuda-test.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=cuda-test
#SBATCH --gres=gpu:h100:2

nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
EOF

spur submit cuda-test.sh
```

Spur sets `CUDA_VISIBLE_DEVICES` for NVIDIA isolation.

## 4. Job Management

```bash
# Queue with formatting
spur queue                         # All jobs
spur queue -u $USER                # Your jobs
spur queue --states=PENDING        # Just pending

# Cancel
spur cancel 42                     # Cancel by ID
spur cancel --user=alice           # Cancel all of alice's jobs

# Hold and release
spur show control hold job 42      # Prevent scheduling
spur show control release job 42   # Allow scheduling

# Node management
spur nodes                         # All nodes
spur show node gpu-node-1          # Detailed info

# Drain a node for maintenance
spur show control update node gpu-node-1 state=drain reason="maintenance"

# Return to service
spur show control update node gpu-node-1 state=idle
```

## 5. Job Arrays

Submit many similar jobs at once:

```bash
cat > array-job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --array=0-99%10

echo "Task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_JOB_ID"
python train.py --lr=$(echo "0.001 * $SLURM_ARRAY_TASK_ID" | bc)
EOF

spur submit array-job.sh
```

`%10` limits concurrency to 10 tasks at a time.

## 6. Job Dependencies

Chain jobs together:

```bash
# Submit preprocessing
JOB1=$(spur submit preprocess.sh | grep -o '[0-9]*')

# Training starts after preprocessing succeeds
JOB2=$(spur submit --dependency=afterok:$JOB1 train.sh | grep -o '[0-9]*')

# Evaluation after training (even if it fails)
spur submit --dependency=afterany:$JOB2 eval.sh
```

## Architecture Overview

```
  User: spur submit / sbatch / REST API
           │
           ▼
  ┌─────────────────┐
  │    spurctld      │  Port 6817 (gRPC)
  │  - backfill      │  Port 6820 (REST, via spurrestd)
  │    scheduler     │
  │  - WAL + redb    │
  │    state store   │
  └────────┬────────┘
           │  dispatches jobs via gRPC
    ┌──────┼──────┐
    ▼      ▼      ▼
  spurd  spurd  spurd     Port 6818 each
  node1  node2  node3
  │       │       │
  └── WireGuard mesh (10.44.0.0/16) ──┘
```

- **spurctld** is the brain — one instance, manages all state, runs the scheduler
- **spurd** runs on every compute node — receives jobs, manages processes via cgroups v2
- Communication is all gRPC over WireGuard (encrypted, NAT-proof)
- State survives restarts via WAL + periodic snapshots (redb)

## Troubleshooting

**Controller won't start:**
```bash
# Check if port is in use
ss -tlnp | grep 6817
# Check state directory permissions
ls -la /var/spool/spur/
```

**Agent can't register:**
```bash
# Test connectivity to controller
grpcurl -plaintext localhost:6817 list
# Or just curl the port
curl http://localhost:6817  # Will fail but proves port is reachable
# Check agent logs
spurd -D --controller http://10.44.0.1:6817 --log-level debug
```

**Jobs stay PENDING:**
```bash
# Check node state — must be "idle" or "mixed"
spur nodes
# Check if nodes are in the right partition
spur show node <name>
# Check job requirements vs available resources
spur show job <id>
```

**WireGuard not working:**
```bash
spur net status                    # Shows interface and peer info
sudo wg show spur0                 # Raw WireGuard status
ping 10.44.0.1                     # Test mesh connectivity
ip addr show spur0                 # Check interface has IP
```

**Multi-node job only runs on one node:**
```bash
# Check that all nodes registered with their WireGuard IP
spur nodes
# Node addresses should be 10.44.x.y, not 127.0.0.1
spur show node gpu-node-1
```

## Next Steps

- **Accounting:** Start `spurdbd` with PostgreSQL for job history and fair-share scheduling
- **REST API:** Start `spurrestd` for HTTP access (Slurm-compatible `/slurm/v0.0.42/` endpoints)
- **Prolog/Epilog:** Set `SPUR_PROLOG` / `SPUR_EPILOG` environment variables to run scripts before/after jobs
- **Kubernetes:** K8s integration is planned for hybrid cloud+on-prem scheduling
