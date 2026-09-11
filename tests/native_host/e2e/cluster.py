# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Spur native-host cluster management for E2E tests.

Handles SSH connections, binary deployment, cluster startup/teardown,
and CLI wrappers for interacting with the running cluster.
"""

import os
import re
import shlex
import shutil
import subprocess
import time
import logging
from pathlib import Path

import paramiko
import pytest
import tomli_w

logger = logging.getLogger(__name__)

BINARIES = ["spurctld", "spurd", "spur", "spurstepd"]
CLI_SYMLINKS = ["sbatch", "srun", "squeue", "scancel", "sinfo", "scontrol"]
ACCOUNTING_SYMLINKS = ["sacct", "sacctmgr", "sshare", "sreport"]

CONTROLLER_PORT = int(os.environ.get("SPUR_TEST_CONTROLLER_PORT", "6817"))
AGENT_PORT = int(os.environ.get("SPUR_TEST_AGENT_PORT", "6818"))

# Injected into spurd's own environment (only) at launch so a test can prove the
# daemon's environment does not leak into jobs or into sessions that enter a
# running job (spur exec, srun --overlap, srun --overlap --pty). Job clients run
# over separate SSH exec channels and never see this var, so its presence in a
# session means it leaked from spurd.
DAEMON_ENV_CANARY = "spurd-private-do-not-leak"

# Extracts just the numeric host port for 5432/tcp. inspect returns structured
# data, so this sidesteps parsing the multi-line, IPv4/IPv6-variant `docker port`
# output. Dual-stack publishes both families on the same port, so index 0 suffices.
DOCKER_PORT_INSPECT_FMT = (
    '{{(index (index .NetworkSettings.Ports "5432/tcp") 0).HostPort}}'
)


def parse_published_port(inspect_output: str) -> int:
    """Host port from ``docker inspect --format DOCKER_PORT_INSPECT_FMT`` output."""
    port = inspect_output.strip()
    if not port.isdigit():
        raise RuntimeError(f"no published port in docker inspect output: {inspect_output!r}")
    return int(port)


def make_remote_dir() -> str:
    """Generate a unique remote working directory path."""
    return f"/tmp/spur-e2e-{os.getpid()}-{time.time_ns()}"


def deep_merge(base: dict, overrides: dict) -> dict:
    """Deep-merge *overrides* into *base* (mutates and returns *base*).

    - Dicts are merged recursively.
    - Everything else (scalars, lists) is replaced outright.
    """
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class SshNode:
    """SSH connection to a single test node."""

    def __init__(self, host: str, user: str, password: str | None = None, key_path: str | None = None):
        self.host = host
        self.user = user
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs = {"hostname": host, "username": user}
        if key_path:
            connect_kwargs["key_filename"] = key_path
        elif password:
            connect_kwargs["password"] = password
        else:
            connect_kwargs["allow_agent"] = True

        self.client.connect(**connect_kwargs)
        self._sftp = None

    @property
    def sftp(self) -> paramiko.SFTPClient:
        if self._sftp is None:
            self._sftp = self.client.open_sftp()
        return self._sftp

    def exec(self, cmd: str, check: bool = True) -> str:
        """Run a command via SSH. Returns stdout. Raises on non-zero exit if check=True."""
        _, stdout, stderr = self.client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode()
        err = stderr.read().decode()

        if check and exit_code != 0:
            raise RuntimeError(
                f"Command failed on {self.host} (exit {exit_code}): {cmd}\n"
                f"stdout: {out}\nstderr: {err}"
            )
        return out

    def exec_allow_fail(self, cmd: str) -> str:
        """Run a command, returning stdout+stderr regardless of exit code."""
        _, stdout, stderr = self.client.exec_command(cmd)
        stdout.channel.recv_exit_status()
        out = stdout.read().decode()
        err = stderr.read().decode()
        return out + err

    def upload(self, local_path: str, remote_path: str):
        """Upload a local file to the remote node."""
        self.sftp.put(local_path, remote_path)

    def write_file(self, remote_path: str, content: str, mode: int | None = None):
        """Write string content to a remote file, optionally setting permissions."""
        with self.sftp.open(remote_path, "w") as f:
            f.write(content)
        if mode is not None:
            self.sftp.chmod(remote_path, mode)

    def read_file(self, remote_path: str) -> str:
        """Read a remote file. Returns empty string if file doesn't exist."""
        return self.exec(f"cat '{remote_path}' 2>/dev/null || true", check=False)

    def close(self):
        if self._sftp:
            self._sftp.close()
        self.client.close()


def ensure_bins(nodes: list[SshNode], binaries_dir: str, bin_dir: str,
                with_accounting: bool = False, with_mpi_plugin: bool = False):
    """
    Upload binaries to all nodes if not already present (or size differs).
    bin_dir is the remote directory where binaries are installed.

    When *with_accounting* is True, also create accounting CLI symlinks.
    """
    for node in nodes:
        node.exec(f"mkdir -p '{bin_dir}'")

    binaries = BINARIES
    symlinks = CLI_SYMLINKS + (ACCOUNTING_SYMLINKS if with_accounting else [])

    for name in binaries:
        local_path = Path(binaries_dir) / name
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Missing binary: {local_path}\n"
                f"Set SPUR_TEST_BINARIES_DIR or run: cargo build --release"
            )
        local_size = local_path.stat().st_size

        for node in nodes:
            remote_path = f"{bin_dir}/{name}"
            remote_size = node.exec_allow_fail(
                f"stat -c%s '{remote_path}' 2>/dev/null || echo 0"
            ).strip()

            if remote_size == str(local_size):
                logger.debug("Binary %s already present on %s", name, node.host)
                continue

            logger.info("Uploading %s to %s", name, node.host)
            node.upload(str(local_path), remote_path)
            node.exec(f"chmod +x '{remote_path}'")

    # Create CLI symlinks
    symlink_cmd = (
        f"cd '{bin_dir}' && "
        + " && ".join(f"ln -sf spur {cmd} 2>/dev/null || true" for cmd in symlinks)
    )
    for node in nodes:
        node.exec_allow_fail(symlink_cmd)

    if with_mpi_plugin:
        plugin_local = os.environ.get("SPUR_TEST_MPI_PLUGIN", "").strip()
        if not plugin_local:
            plugin_local = str(Path(binaries_dir).parent / "lib" / "spur" / "spur_mpi_pmix.so")
            if not Path(plugin_local).is_file():
                plugin_local = str(Path(binaries_dir) / "libspur_mpi_pmix.so")
        plugin_path = Path(plugin_local)
        if not plugin_path.is_file():
            raise FileNotFoundError(
                f"Missing MPI plugin: {plugin_path}\n"
                "Build with: cargo build --release -p spur-mpi-pmix\n"
                "Or set SPUR_TEST_MPI_PLUGIN"
            )
        remote_plugin_dir = str(Path(bin_dir).parent / "lib" / "spur")
        for node in nodes:
            node.exec(f"mkdir -p '{remote_plugin_dir}'")
            remote_plugin = f"{remote_plugin_dir}/spur_mpi_pmix.so"
            node.upload(str(plugin_path), remote_plugin)

    logger.info("Binaries ready at %s on all nodes", bin_dir)


class SpurCluster:
    """
    Manages a Spur cluster lifecycle on remote nodes.

    Starts controller + agents in a unique working directory,
    waits for the cluster to become ready, and tears everything down.
    """

    # Payload carried by build_imported_container_image(): a directory that
    # exists only inside the image, so a bare command name resolves there
    # only when the image's own PATH reaches the container.
    IMAGE_ONLY_BIN_DIR = "/opt/img/bin"
    IMAGE_PAYLOAD = "img-payload"
    IMAGE_PAYLOAD_MARKER = "IMAGE_PAYLOAD_OK"

    # Real public image pulled by pull_container_image(). Its config.Env holds
    # PATH (first entry below), LANG and PYTHON_VERSION, none of which the job
    # environment supplies. Pinned so those expectations stay true.
    TEST_IMAGE = "mirror.gcr.io/library/python:3.11-slim"
    TEST_IMAGE_REGISTRY = "mirror.gcr.io"
    TEST_IMAGE_PATH_HEAD = "/usr/local/bin"

    def __init__(self, nodes: list[SshNode], remote_dir: str, bin_dir: str):
        self.nodes = nodes
        self.node_names: list[str] = []
        self.remote_dir = remote_dir
        self.bin_dir = bin_dir
        self.etc_dir = f"{remote_dir}/etc"
        self.state_dir = f"{remote_dir}/state"
        self.log_dir = f"{remote_dir}/log"
        self.controller_addr = f"http://{nodes[0].host}:{CONTROLLER_PORT}"
        self.config_overrides: dict = {}
        self.agent_as_root: bool = False
        self.agent_labels: dict[int, dict[str, str]] = {}
        self.agent_token: str | None = None
        self.accounting_enabled: bool = False
        self._pg_container = f"spur-e2e-pg-{os.getpid()}-{time.time_ns()}"
        self._pg_port: int | None = None

    @property
    def _db_url(self) -> str:
        if self._pg_port is None:
            raise RuntimeError(
                "Postgres host port unresolved: _start_postgres() must run before "
                "the controller config is written"
            )
        return f"postgresql://spur:spur@127.0.0.1:{self._pg_port}/spur"

    # --- Lifecycle ---

    def provision(self):
        """Create remote dirs and resolve hostnames.

        After this call the cluster infrastructure is ready (dirs exist,
        hostnames known) but no daemons are running.  Call :meth:`start`
        to bring up the cluster with a specific configuration.
        """
        self._create_dirs()
        self._resolve_hostnames()
        logger.info("Cluster provisioned: %s", self.node_names)

    def start(
        self,
        config_overrides: dict | None = None,
        kill_stale: bool = True,
        agent_as_root: bool = False,
        agent_labels: dict[int, dict[str, str]] | None = None,
    ):
        """Write config and start all daemons.

        *config_overrides* is deep-merged into the default config.
        Requires :meth:`provision` to have been called first.

        When *kill_stale* is True (default), any lingering spurctld/spurd
        processes on the nodes are killed before starting fresh.

        When *agent_as_root* is True, spurd is launched via sudo on each
        node (rootful agent). spurctld always runs as the SSH user.

        *agent_labels* maps node index to a dict of labels that will be
        passed as ``--label key=value`` args to spurd on that node.
        """
        if not self.node_names:
            raise RuntimeError("provision() must be called before start()")
        self.agent_as_root = agent_as_root
        self.agent_labels = agent_labels or {}
        if kill_stale:
            self._kill_controller()
            self._kill_agents(use_sudo=False, broad=True)
            self._kill_agents(use_sudo=True, broad=True)  # best-effort for rootful
        if self.accounting_enabled:
            self._start_postgres()
        self.start_controller(config_overrides, kill_stale=False)
        self.start_agents(kill_stale=False)
        self.wait_ready()
        logger.info(
            "Cluster ready: %s (agent_as_root=%s)",
            self.node_names,
            self.agent_as_root,
        )

    def stop(self):
        """Kill all daemons but keep the working directory intact."""
        self.stop_agents()
        self.stop_controller()
        if self.accounting_enabled:
            self._stop_postgres()

    def start_controller(self, config_overrides: dict | None = None, kill_stale: bool = True):
        """Start only spurctld. Writes config and waits for it to be ready.

        Use :meth:`start_agents` afterward to bring up agents separately.
        """
        if not self.node_names:
            raise RuntimeError("provision() must be called first")
        self.config_overrides = config_overrides or {}
        if kill_stale:
            self._kill_controller()
        self._write_config()
        self._start_controller()
        time.sleep(2)

    def start_agents(self, token: str | None = None, kill_stale: bool = True):
        """Start spurd on all nodes, optionally with a join token."""
        self.agent_token = token
        if kill_stale:
            self._kill_agents(use_sudo=self.agent_as_root)
        self._start_agents()

    def enable_accounting(self):
        """Opt in to accounting + Postgres. Call before provision()/start().

        Requires Docker on node 0.
        Raises RuntimeError if the prerequisites are missing — callers (the
        accounting_cluster fixture) translate that into a pytest skip.
        """
        node = self.nodes[0]
        # `docker info` (not `command -v docker`) so an unreachable daemon socket skips, not errors.
        if node.exec_allow_fail("docker info >/dev/null 2>&1 && echo ok").strip() != "ok":
            raise RuntimeError("docker not usable on node 0; cannot run accounting")
        self.accounting_enabled = True

    def stop_controller(self):
        """Kill spurctld only."""
        self._kill_controller()

    def stop_agents(self):
        """Kill spurd on all nodes only."""
        self._kill_agents(use_sudo=self.agent_as_root)

    def deploy(
        self,
        config_overrides: dict | None = None,
        agent_as_root: bool = False,
        agent_labels: dict[int, dict[str, str]] | None = None,
    ):
        """Provision + start in one call."""
        self.provision()
        if agent_as_root:
            self.root_agent_preflight()
        self.start(config_overrides, agent_as_root=agent_as_root, agent_labels=agent_labels)

    def teardown(self):
        """Kill all daemons and remove the working directory."""
        self.stop()
        rm_prefix = self._sudo_prefix() if self.agent_as_root else ""
        for node in self.nodes:
            node.exec_allow_fail(f"{rm_prefix}rm -rf '{self.remote_dir}'")
        logger.info("Cluster torn down")

    # --- CLI wrappers ---

    def cli(self, args: list[str], controller_addr: str | None = None) -> str:
        """Run a spur CLI command on the controller node.

        *controller_addr* overrides the endpoint(s) passed via
        ``SPUR_CONTROLLER_ADDR`` (e.g. a comma-separated failover list).
        """
        cmd_parts = [
            f"SPUR_CONTROLLER_ADDR='{controller_addr or self.controller_addr}'",
            f"PATH='{self.bin_dir}':$PATH",
            f"'{self.bin_dir}/{args[0]}'",
        ]
        cmd_parts.extend(f"'{a}'" for a in args[1:])
        return self.nodes[0].exec(" ".join(cmd_parts))

    def cli_allow_fail(self, args: list[str], controller_addr: str | None = None) -> str:
        """Run a spur CLI command, returning stdout+stderr regardless of exit
        code. Use to assert on expected submission rejections.

        *controller_addr* overrides ``SPUR_CONTROLLER_ADDR`` as in :meth:`cli`.
        """
        cmd_parts = [
            f"SPUR_CONTROLLER_ADDR='{controller_addr or self.controller_addr}'",
            f"PATH='{self.bin_dir}':$PATH",
            f"'{self.bin_dir}/{args[0]}'",
        ]
        cmd_parts.extend(f"'{a}'" for a in args[1:])
        return self.nodes[0].exec_allow_fail(" ".join(cmd_parts))

    def cli_as_user(
        self,
        run_as: str,
        args: list[str],
        controller_addr: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        """Run a spur CLI command as a specific UNIX user via sudo, returning
        stdout+stderr regardless of exit code.

        Commands that carry an identity (reservation create/update/delete,
        job cancel, ...) derive it from the invoking account (``whoami``), so
        this is how a test exercises owner-vs-non-owner behavior end to end.

        *extra_env* adds variables to the environment. Pass ``SPUR_AUTH_TOKEN`` to
        separate the identity the controller verifies from the invoking account.
        """
        inner = [
            f"SPUR_CONTROLLER_ADDR='{controller_addr or self.controller_addr}'",
            f"PATH='{self.bin_dir}':$PATH",
        ]
        for key, value in (extra_env or {}).items():
            inner.append(f"{key}='{value}'")
        inner.append(f"'{self.bin_dir}/{args[0]}'")
        inner.extend(f"'{a}'" for a in args[1:])
        cmd = f"{self._sudo_prefix()}-u '{run_as}' env {' '.join(inner)}"
        return self.nodes[0].exec_allow_fail(cmd)

    def cli_with_exit(
        self, args: list[str], controller_addr: str | None = None
    ) -> tuple[int, str]:
        """Run a spur CLI command and return (exit_code, combined stdout+stderr)."""
        cmd_parts = [
            f"SPUR_CONTROLLER_ADDR='{controller_addr or self.controller_addr}'",
            f"PATH='{self.bin_dir}':$PATH",
            f"'{self.bin_dir}/{args[0]}'",
        ]
        cmd_parts.extend(f"'{a}'" for a in args[1:])
        _, stdout, stderr = self.nodes[0].client.exec_command(" ".join(cmd_parts))
        code = stdout.channel.recv_exit_status()
        return code, stdout.read().decode() + stderr.read().decode()

    def sbatch(self, args: list[str]) -> str:
        return self.cli(["sbatch"] + args)

    def srun_with_exit(self, args: list[str]) -> tuple[int, str]:
        """Run srun and return (exit_code, combined stdout+stderr)."""
        cmd_parts = [
            f"SPUR_CONTROLLER_ADDR={shlex.quote(self.controller_addr)}",
            f"PATH={shlex.quote(self.bin_dir)}:$PATH",
            shlex.quote(f"{self.bin_dir}/srun"),
        ]
        cmd_parts.extend(shlex.quote(a) for a in args)
        _, stdout, stderr = self.nodes[0].client.exec_command(" ".join(cmd_parts))
        code = stdout.channel.recv_exit_status()
        return code, stdout.read().decode() + stderr.read().decode()

    def srun_in_allocation(
        self,
        job_id: int,
        args: list[str],
        extra_env: dict[str, str] | None = None,
        unset_env: list[str] | None = None,
    ) -> tuple[int, str]:
        """Run srun inside an existing allocation (step mode, as after salloc).

        *extra_env* can set ``SPUR_JOB_USER`` / ``SPUR_AUTH_TOKEN`` to exercise
        step-mode identity separately from the invoking account.  *unset_env*
        removes variables (via ``env -u``) so an empty value does not fall through
        to ``~/.spur/token``.
        """
        cmd_parts = [
            f"SPUR_JOB_ID={job_id}",
            f"SPUR_CONTROLLER_ADDR={shlex.quote(self.controller_addr)}",
            f"PATH={shlex.quote(self.bin_dir)}:$PATH",
        ]
        for key, value in (extra_env or {}).items():
            cmd_parts.append(f"{key}={shlex.quote(str(value))}")
        cmd_parts.append(shlex.quote(f"{self.bin_dir}/srun"))
        cmd_parts.extend(shlex.quote(a) for a in args)
        cmd = " ".join(cmd_parts)
        if unset_env:
            unset = " ".join(f"-u {shlex.quote(name)}" for name in unset_env)
            cmd = f"env {unset} {cmd}"
        _, stdout, stderr = self.nodes[0].client.exec_command(cmd)
        code = stdout.channel.recv_exit_status()
        return code, stdout.read().decode() + stderr.read().decode()

    def salloc_run(
        self,
        shell_body: str,
        *,
        salloc_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        """Run salloc with *shell_body* as the allocation shell (via ``SHELL``).

        salloc exports allocation env vars into the shell, then cancels the job
        when the shell exits.  Returns ``(exit_code, combined_output)``.
        """
        script_path = self.write_file(
            f"salloc-shell-{time.time_ns()}.sh",
            f"#!/bin/bash\nset -euo pipefail\n{shell_body}\n",
        )

        cmd_parts = [
            f"SPUR_CONTROLLER_ADDR={shlex.quote(self.controller_addr)}",
            f"PATH={shlex.quote(self.bin_dir)}:$PATH",
            f"SHELL={shlex.quote(script_path)}",
        ]
        for key, value in (extra_env or {}).items():
            cmd_parts.append(f"{key}={shlex.quote(str(value))}")
        cmd_parts.append(shlex.quote(f"{self.bin_dir}/spur"))
        cmd_parts.append("salloc")
        cmd_parts.extend(shlex.quote(a) for a in (salloc_args or ["-N", "1", "-t", "0:05"]))

        _, stdout, stderr = self.nodes[0].client.exec_command(" ".join(cmd_parts))
        code = stdout.channel.recv_exit_status()
        return code, stdout.read().decode() + stderr.read().decode()

    def squeue(self, args: list[str]) -> str:
        return self.cli(["squeue"] + args)

    def squeue_all(self) -> str:
        return self.cli(["squeue", "-t", "all"])

    def running_job_ids_by_name(self, name: str) -> list[int]:
        """Return running job IDs matching *name* (uses server-side name filter)."""
        out = self.squeue(["-n", name, "-t", "R", "-h", "-o", "%i"])
        ids: list[int] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.append(int(line.split()[0]))
            except ValueError:
                continue
        return ids

    def sinfo(self) -> str:
        return self.cli(["sinfo"])

    def sinfo_nodes(self, controller_addr: str | None = None) -> dict[str, str]:
        """Map node name to state via node-oriented sinfo.

        ``sinfo -N`` emits one line per node with the full, uncompressed name,
        so this is agnostic to NODELIST hostlist compression (``node[1-4]``).
        The state is the short form (idle/mix/alloc/down/drain/resv), with any
        trailing ``*`` (non-responding marker) stripped.

        *controller_addr* overrides the endpoint(s) as in :meth:`cli`.
        """
        out = self.cli(["sinfo", "-N", "-h", "-o", "%N %t"],
                       controller_addr=controller_addr)
        states: dict[str, str] = {}
        for line in out.splitlines():
            fields = line.split()
            if len(fields) >= 2:
                states[fields[0]] = fields[1].rstrip("*")
        return states

    def nodes_in_partition(self, partition: str) -> set[str]:
        """Return the set of node names belonging to *partition*.

        Uses node-oriented sinfo so membership is exact even when the default
        output would compress several nodes into one bracketed hostlist. A node
        in multiple partitions appears once per partition. The partition name
        carries a trailing ``*`` when it is the default, so that is stripped.
        """
        out = self.cli(["sinfo", "-N", "-h", "-o", "%N %P"])
        members: set[str] = set()
        for line in out.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[1].rstrip("*") == partition:
                members.add(fields[0])
        return members

    def scancel(self, job_id: str) -> str:
        return self.cli(["scancel", job_id])

    def scontrol(self, *args: str) -> str:
        return self.cli(["scontrol"] + list(args))

    def sdiag(self) -> str:
        return self.cli(["spur", "diag"])

    def sdiag_jobs_preempted(self) -> int:
        """Return the jobs_preempted counter from spur diag scheduler stats."""
        import re
        output = self.sdiag()
        match = re.search(r"Jobs preempted\s+:\s+(\d+)", output)
        if match is None:
            raise AssertionError(
                f"'Jobs preempted' not found in spur diag output:\n{output}"
            )
        return int(match.group(1))

    # --- Native k0s cluster (spur k8s) wrappers ---

    def k8s_up(self, args: list[str] | None = None) -> str:
        # k0s up/down are admin-gated; run as root so they pass without accounting.
        return self.cli_as_user("root", ["spur", "k8s", "up"] + (args or []))

    def k8s_down(self, reset: bool = True) -> str:
        extra = ["--reset"] if reset else []
        return self.cli_as_user("root", ["spur", "k8s", "down"] + extra)

    def k8s_add_nodes(self, args: list[str]) -> str:
        # add-nodes is admin-gated; run as root so it passes without accounting.
        return self.cli_as_user("root", ["spur", "k8s", "add-nodes"] + args)

    def k8s_remove_nodes(self, args: list[str]) -> str:
        # remove-nodes is admin-gated; run as root so it passes without accounting.
        return self.cli_as_user("root", ["spur", "k8s", "remove-nodes"] + args)

    def k8s_status(self) -> str:
        return self.cli(["spur", "k8s", "status"])

    def k8s_member_list(self) -> list[str]:
        """Parse `members:` into a name list; empty list means "all nodes" (whole inventory)."""
        members = self.k8s_members()
        if not members or members == "all nodes":
            return []
        return [n.strip() for n in members.split(",") if n.strip()]

    def k8s_control_planes(self) -> list[str]:
        """Parse the control-plane node list from `spur k8s status`."""
        for line in self.k8s_status().splitlines():
            if line.startswith("control-plane:"):
                names = line.split(":", 1)[1].strip()
                return [n.strip() for n in names.split(",") if n.strip()]
        return []

    def k8s_members(self) -> str:
        """Parse the `members:` line from `spur k8s status` ("all nodes" or a name list)."""
        for line in self.k8s_status().splitlines():
            if line.startswith("members:"):
                return line.split(":", 1)[1].strip()
        return ""

    def k8s_active_controllers(self) -> list[str]:
        """Node names whose `spur k8s status` row is role=controller/single and active."""
        out = []
        for line in self.k8s_status().splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[1] in ("controller", "single") and fields[2] == "active":
                out.append(fields[0])
        return out

    def wait_k8s_phase(self, phase: str, timeout: int = 600) -> str:
        """Poll `spur k8s status` until it reports the given phase."""
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            last = self.k8s_status()
            for line in last.splitlines():
                if line.startswith("phase:") and line.split(":", 1)[1].strip() == phase:
                    return last
            time.sleep(5)
        raise TimeoutError(f"k0s phase did not reach {phase} within {timeout}s:\n{last}")

    def etcd_member_count(self) -> int:
        """Ground-truth etcd quorum size via `k0s etcd member-list`. Runs on a control-plane node
        (etcd only exists there; node 0 may be a worker since CPs are chosen by lexical order)."""
        cps = set(self.k8s_control_planes())
        for i, name in enumerate(self.node_names):
            if name not in cps:
                continue
            out = self.nodes[i].exec_allow_fail(
                f"{self._sudo_prefix()}k0s etcd member-list 2>/dev/null"
            )
            members = re.findall(r"https?://[^\"]+:2380", out)
            if members:
                return len(members)
        return 0

    def wait_etcd_members(self, count: int, timeout: int = 120) -> int:
        """Poll until etcd reports `count` members. The `ready` phase only means every
        k0scontroller unit is systemd-active; etcd quorum converges shortly after."""
        deadline = time.time() + timeout
        last = 0
        while time.time() < deadline:
            last = self.etcd_member_count()
            if last >= count:
                return last
            time.sleep(3)
        return last

    def sacct(self, args: list[str]) -> str:
        return self.cli(["sacct"] + args)

    def sacctmgr(self, args: list[str]) -> str:
        return self.cli(["sacctmgr"] + args)

    def write_file(self, name: str, body: str, *,
                   all_nodes: bool = False, executable: bool = True) -> str:
        """Write a file under remote_dir. Returns the absolute remote path.

        By default the file is written to the controller node only and
        made executable.  Set *all_nodes=True* to write on every node,
        and *executable=False* for non-script files.
        """
        path = f"{self.remote_dir}/{name}"
        # Even non-executable files get an explicit mode, not the remote umask:
        # the job_submit hook refuses a group/world-writable file, so an umask
        # of 002 would otherwise stop the controller from starting.
        mode = 0o755 if executable else 0o644
        targets = self.nodes if all_nodes else self.nodes[:1]
        parent = path.rsplit("/", 1)[0]
        for node in targets:
            node.exec(f"mkdir -p '{parent}'")
            node.write_file(path, body, mode=mode)
        return path

    def read_output_on_any_node(self, path: str) -> str:
        """Try to read a file from any node (controller first)."""
        for node in self.nodes:
            content = node.read_file(path)
            if content.strip():
                return content
        return ""

    def read_output_all_nodes(self, path: str) -> str:
        """Read a file from all nodes and combine."""
        combined = []
        for node in self.nodes:
            content = node.read_file(path)
            if content.strip():
                combined.append(content)
        return "\n".join(combined)

    def debug_job(self, job_id: int) -> str:
        """Collect diagnostic info for a failed job."""
        lines = [f"=== DEBUG job {job_id} ==="]

        try:
            lines.append("scontrol show job:")
            lines.append(self.scontrol("show", "job", str(job_id)))
        except Exception:
            pass

        try:
            lines.append("sinfo:")
            lines.append(self.sinfo())
        except Exception:
            pass

        try:
            lines.append("squeue:")
            lines.append(self.squeue_all())
        except Exception:
            pass

        ctrl_log = f"{self.log_dir}/spurctld.log"
        log = self.nodes[0].exec_allow_fail(f"tail -30 '{ctrl_log}'")
        if log.strip():
            lines.append(f"spurctld.log (last 30 lines):\n{log}")

        for i, node in enumerate(self.nodes):
            agent_log = f"{self.log_dir}/spurd.log"
            log = node.exec_allow_fail(f"tail -40 '{agent_log}'")
            if log.strip():
                lines.append(f"spurd.log on {self.node_names[i]} (last 40 lines):\n{log}")

        # Most tests assert on -o only, so a failure written to the job's
        # stderr is otherwise invisible.
        try:
            match = re.search(r"StdErr=(\S+)", self.scontrol("show", "job", str(job_id)))
            if match:
                err = self.read_output_on_any_node(match.group(1))
                if err.strip():
                    lines.append(f"job stderr ({match.group(1)}):\n{err}")
        except Exception:
            pass

        return "\n".join(lines)

    def ship_file_to_all(self, local_path: Path, remote_name: str):
        """SCP a local file to all nodes under remote_dir."""
        for node in self.nodes:
            node.upload(str(local_path), f"{self.remote_dir}/{remote_name}")

    def gpu_preflight(self, min_nodes: int):
        """Verify that at least min_nodes have GPU hardware. Skips test if not."""
        gpu_nodes = []
        for i, node in enumerate(self.nodes):
            probe = node.exec_allow_fail(
                "{ ls /dev/kfd 2>/dev/null && echo HAS_GPU; } || "
                "{ nvidia-smi -L 2>/dev/null && echo HAS_GPU; } || "
                "echo NO_GPU"
            )
            if "HAS_GPU" in probe:
                gpu_nodes.append(self.node_names[i])

        if len(gpu_nodes) < min_nodes:
            pytest.skip(
                f"GPU preflight: need {min_nodes} GPU node(s), "
                f"found {len(gpu_nodes)} ({gpu_nodes})"
            )

    def root_agent_preflight(self):
        """Verify passwordless (or password-backed) sudo for rootful spurd. Skips if not."""
        for i, node in enumerate(self.nodes):
            try:
                node.exec(f"{self._sudo_prefix()}true")
            except RuntimeError:
                pytest.skip(
                    f"rootful spurd requires sudo on {self.node_names[i]} "
                    f"(set SPUR_TEST_SSH_PASSWORD or configure NOPASSWD sudo)"
                )

    def require_nodes(self, min_nodes: int):
        """Skip the test if fewer than min_nodes are configured."""
        if len(self.nodes) < min_nodes:
            pytest.skip(
                f"Need at least {min_nodes} nodes in SPUR_TEST_NODES "
                f"(got {len(self.nodes)})"
            )

    @staticmethod
    def devices_config(auto_detect: bool = True, **extra) -> dict:
        """Return config_overrides for spur-devices CDI auto-detect."""
        devices = {"auto_detect": auto_detect, **extra}
        return {"devices": devices}

    def spurd_log(self, node_index: int = 0) -> str:
        raw = self.nodes[node_index].read_file(f"{self.log_dir}/spurd.log")
        return re.sub(r"\x1b\[[0-9;]*m", "", raw)

    def spurd_registry_gpu_count(self, node_index: int = 0) -> int | None:
        """Parse cdi_devices count from spurd startup log."""
        log = self.spurd_log(node_index)
        match = re.search(r"cdi_devices=(\d+)", log)
        if match:
            return int(match.group(1))
        match = re.search(r"resources discovered.*?gpus=(\d+)", log)
        if match:
            return int(match.group(1))
        return None

    def assert_spurd_registry(self, node_index: int = 0, min_gpus: int = 1):
        log = self.spurd_log(node_index)
        assert "device registry initialized" in log, (
            f"spurd on {self.node_names[node_index]} must initialize device registry\n"
            f"log tail:\n{log[-2000:]}"
        )
        count = self.spurd_registry_gpu_count(node_index)
        assert count is not None and count >= min_gpus, (
            f"expected >= {min_gpus} CDI GPUs on {self.node_names[node_index]}, "
            f"got {count}\nlog tail:\n{log[-2000:]}"
        )

    def scontrol_show_node(self, node_name: str) -> str:
        return self.scontrol("show", "node", node_name)

    def node_gpu_count(self, node_name: str) -> int:
        """Return schedulable GPU count from scontrol show node."""
        out = self.scontrol_show_node(node_name)
        for line in out.splitlines():
            if "Gres=" not in line:
                continue
            gres = line.split("Gres=", 1)[1].strip()
            if not gres:
                return 0
            return len(re.findall(r"gpu:[^,]+", gres))
        return 0

    def assert_sinfo_gpus(self, min_per_node: int = 1):
        for name in self.node_names:
            count = self.node_gpu_count(name)
            assert count >= min_per_node, (
                f"node {name} must expose >= {min_per_node} GPU(s) in scontrol, "
                f"got {count}\n{self.scontrol_show_node(name)}"
            )

    def ship_fixture(self, fixture_name: str) -> str:
        """Ship a file from the fixtures/ directory to remote_dir on all nodes."""
        fixtures_dir = Path(__file__).resolve().parent / "fixtures"
        local_path = fixtures_dir / fixture_name
        if not local_path.is_file():
            raise FileNotFoundError(f"Missing fixture: {local_path}")
        self.ship_file_to_all(local_path, fixture_name)
        return f"{self.remote_dir}/{fixture_name}"

    def compile_hip_fixture(self, hip_filename: str) -> str:
        """Ship and compile a HIP fixture on all nodes. Returns remote binary path."""
        self.ship_fixture(hip_filename)
        bin_name = hip_filename.rsplit(".", 1)[0]
        remote_bin = f"{self.remote_dir}/{bin_name}"
        remote_hip = f"{self.remote_dir}/{hip_filename}"
        for node in self.nodes:
            node.exec_allow_fail(
                f"export PATH=/opt/rocm/bin:$PATH; "
                f"command -v hipcc >/dev/null && "
                f"hipcc -o '{remote_bin}' '{remote_hip}' 2>/dev/null || true"
            )
            if not node.exec_allow_fail(f"test -x '{remote_bin}' && echo OK").strip():
                pytest.skip(f"hipcc could not build {hip_filename} on {node.host}")
        return remote_bin

    def mpi_preflight(self, min_nodes: int = 1):
        """Skip unless MPI plugin, libpmix, and mpicc are available."""
        missing = []
        plugin_local = os.environ.get("SPUR_TEST_MPI_PLUGIN", "").strip()
        if not plugin_local:
            binaries_dir = os.environ.get(
                "SPUR_TEST_BINARIES_DIR",
                str(Path(__file__).resolve().parents[3] / "target" / "release"),
            )
            candidates = [
                Path(binaries_dir).parent / "lib" / "spur" / "spur_mpi_pmix.so",
                Path(binaries_dir) / "libspur_mpi_pmix.so",
            ]
            plugin_local = next((str(p) for p in candidates if p.is_file()), "")
        if not plugin_local or not Path(plugin_local).is_file():
            missing.append("spur_mpi_pmix.so (build -p spur-mpi-pmix or set SPUR_TEST_MPI_PLUGIN)")

        if len(self.nodes) < min_nodes:
            pytest.skip(
                f"MPI tests require at least {min_nodes} node(s) "
                f"(got {len(self.nodes)})"
            )

        for i, node in enumerate(self.nodes[:min_nodes]):
            libpmix = node.exec_allow_fail(
                "ldconfig -p 2>/dev/null | grep -q libpmix && echo OK || echo MISSING"
            )
            if "OK" not in libpmix:
                missing.append(f"libpmix on {self.node_names[i]}")

        mpicc = os.environ.get("SPUR_TEST_MPICC", "mpicc").strip() or "mpicc"
        build_node = self.nodes[0]
        mpicc_ok = build_node.exec_allow_fail(
            f"command -v {shlex.quote(mpicc)} >/dev/null && echo OK || echo MISSING"
        )
        if "OK" not in mpicc_ok:
            missing.append(f"{mpicc} on controller node {self.node_names[0]}")

        if missing:
            pytest.skip("MPI preflight failed: " + "; ".join(missing))

    def compile_mpi_fixture(self, source_name: str = "hello_mpi.c") -> str:
        """Ship and compile an MPI C fixture on all nodes. Returns remote binary path."""
        self.mpi_preflight(1)
        self.ship_fixture(source_name)
        bin_name = source_name.rsplit(".", 1)[0]
        remote_src = f"{self.remote_dir}/{source_name}"
        remote_bin = f"{self.remote_dir}/{bin_name}"
        mpicc = os.environ.get("SPUR_TEST_MPICC", "mpicc").strip() or "mpicc"
        for node in self.nodes:
            node.exec(
                f"{shlex.quote(mpicc)} -o {shlex.quote(remote_bin)} {shlex.quote(remote_src)}"
            )
            if not node.exec_allow_fail(f"test -x '{remote_bin}' && echo OK").strip():
                pytest.skip(f"{mpicc} could not build {source_name} on {node.host}")
        return remote_bin

    def mpi_plugin_dir(self) -> str:
        return str(Path(self.bin_dir).parent / "lib" / "spur")

    def restart_agent(self, node_index: int = 0):
        """Restart spurd on one node without touching the controller."""
        node = self.nodes[node_index]
        self._pkill(node, f"{self.bin_dir}/spurd", use_sudo=self.agent_as_root)
        time.sleep(1)
        node.exec(self._spurd_start_cmd(node_index))
        time.sleep(5)

    def restart_controller(self):
        """Restart spurctld without touching the agents. State is recovered
        from the Raft log on the existing state-dir. Waits for the controller
        to answer queries again (does not require nodes to be idle, since a
        suspended job keeps its allocation)."""
        self._pkill(self.nodes[0], f"{self.bin_dir}/spurctld", use_sudo=False)
        time.sleep(1)
        self._start_controller()
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                nodes = self.sinfo_nodes()
                if all(n in nodes for n in self.node_names):
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError("controller did not recover after restart")

    def spurd_agent_user(self, node_index: int = 0) -> str:
        """Return the user owning the spurd process on a node."""
        node = self.nodes[node_index]
        return node.exec_allow_fail(
            f"ps -o user= -p $(pgrep -f '{self.bin_dir}/spurd' | head -1) "
            f"2>/dev/null || echo unknown"
        ).strip()

    def wait_output(self, path: str, marker: str, timeout: int = 120) -> str:
        """Poll job output file until marker appears or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            content = self.read_output_on_any_node(path)
            if marker in content:
                return content
            if "MISSING" in content or "HIP error" in content:
                return content
            time.sleep(2)
        return self.read_output_on_any_node(path)

    def container_preflight(self):
        """Check container prerequisites. Skips test if anything is missing."""
        if not shutil.which("mksquashfs"):
            pytest.skip("mksquashfs not found on test runner (apt install squashfs-tools)")

        for i, node in enumerate(self.nodes):
            probe = node.exec_allow_fail(
                "command -v unsquashfs >/dev/null && echo OK || echo MISSING"
            )
            if "OK" not in probe:
                pytest.skip(
                    f"unsquashfs not found on {self.node_names[i]} "
                    f"(apt install squashfs-tools)"
                )

    def _build_test_rootfs(self, rootfs: Path, extra: str = "") -> None:
        """
        Populate *rootfs* locally with a minimal userland: host binaries plus
        the libraries ldd reports for them. *extra* is shell appended to the
        build script, run with $R pointing at the rootfs.
        """
        build_script = f"""set -e
R='{rootfs}'
mkdir -p "$R/bin" "$R/usr/bin" "$R/lib" "$R/lib64" \
  "$R/etc" "$R/dev" "$R/proc" "$R/sys" "$R/tmp" \
  "$R/run" "$R/home" "$R/mnt"
for b in bash cat echo sleep hostname id df env stat ls wc head tail tr touch mkdir getent; do
  src=$(which "$b" 2>/dev/null) || continue
  [ -f "$src" ] && cp "$src" "$R/usr/bin/"
done
ln -sf /usr/bin/bash "$R/bin/bash"
ln -sf /usr/bin/bash "$R/bin/sh"
for f in "$R/usr/bin/"*; do
  ldd "$f" 2>/dev/null | grep '=>' | awk '{{print $3}}' | while read -r lib; do
    if [ -f "$lib" ]; then
      dir=$(dirname "$lib")
      mkdir -p "$R$dir"
      cp -n "$lib" "$R$lib" 2>/dev/null || true
    fi
  done
done
for nsslib in libnss_dns.so.2 libnss_files.so.2 libresolv.so.2; do
  src="/lib/x86_64-linux-gnu/$nsslib"
  [ -f "$src" ] && mkdir -p "$R/lib/x86_64-linux-gnu" && cp -n "$src" "$R/lib/x86_64-linux-gnu/" 2>/dev/null || true
done
[ -f /lib64/ld-linux-x86-64.so.2 ] && cp -n /lib64/ld-linux-x86-64.so.2 "$R/lib64/" 2>/dev/null || true
for f in /etc/passwd /etc/group /etc/nsswitch.conf; do
  [ -f "$f" ] && cp "$f" "$R/etc/"
done
{extra}
"""
        result = subprocess.run(
            ["sh", "-c", build_script],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"rootfs build failed: {result.stderr}")

    def build_container_image(self, tmp_path: Path) -> str:
        """
        Build a minimal squashfs container image locally, ship to all nodes.
        Returns the remote path to the .sqsh file.
        """
        remote_path = f"{self.remote_dir}/test-container.sqsh"
        rootfs = tmp_path / "rootfs"
        local_img = tmp_path / "test-container.sqsh"
        self._build_test_rootfs(
            rootfs,
            extra=f"""mksquashfs "$R" '{local_img}' -noappend -quiet >/dev/null 2>&1""",
        )

        for node in self.nodes:
            node.upload(str(local_img), remote_path)

        return remote_path

    def build_imported_container_image(
        self,
        tmp_path: Path,
        image_env: list[str],
        tag: str = "spur-e2e-imgenv:test",
    ) -> str:
        """
        Build an image through the real `spur image import` path, so the image
        config recorded at import time is on the rootfs when a job launches.
        *image_env* becomes the image's config.Env.

        The rootfs carries a payload binary under IMAGE_ONLY_BIN_DIR, a
        directory that exists nowhere on the host, so it is reachable only if
        the image's own PATH survives into the container. It is a copy of a
        real ELF rather than a shebang script, so the job's execve resolves a
        binary the way `torchrun` from a venv image would.

        Skips the test when docker or mksquashfs is unusable on node 0, where
        both the docker build and the import run. Returns the remote .sqsh.
        """
        node = self.nodes[0]
        # `docker info` (not `command -v docker`) so an unreachable daemon socket skips.
        if node.exec_allow_fail("docker info >/dev/null 2>&1 && echo ok").strip() != "ok":
            pytest.skip("docker not usable on node 0; cannot run `spur image import`")

        rootfs = tmp_path / "import-rootfs"
        local_tar = tmp_path / "import-rootfs.tar.gz"
        payload = f"{self.IMAGE_ONLY_BIN_DIR}/{self.IMAGE_PAYLOAD}"
        self._build_test_rootfs(rootfs, extra=f"""
mkdir -p "$R{self.IMAGE_ONLY_BIN_DIR}"
cp "$R/usr/bin/echo" "$R{payload}"
tar -C "$R" -czf '{local_tar}' .
""")

        remote_tar = f"{self.remote_dir}/import-rootfs.tar.gz"
        node.upload(str(local_tar), remote_tar)

        # docker mints the config.Env; `spur image import` is what records it.
        changes = " ".join(f"--change {shlex.quote('ENV ' + v)}" for v in image_env)
        node.exec_allow_fail(f"docker rmi -f {shlex.quote(tag)}")
        node.exec(f"docker import {changes} '{remote_tar}' {shlex.quote(tag)}")
        return self.import_container_image(f"dockerd://{tag}", "dockerd")

    def pull_container_image(self) -> str:
        """
        Import TEST_IMAGE straight from its registry, so the pull path rather
        than `docker save` is what records the config. Skips when the registry
        is unreachable. Returns the remote .sqsh path on node 0.
        """
        probe = self.nodes[0].exec_allow_fail(
            f"getent hosts {self.TEST_IMAGE_REGISTRY} >/dev/null && echo ok"
        )
        if "ok" not in probe:
            pytest.skip(f"{self.TEST_IMAGE_REGISTRY} unreachable from node 0")
        return self.import_container_image(self.TEST_IMAGE, "registry")

    def import_container_image(self, uri: str, slug: str) -> str:
        """
        Run the real `spur image import` on node 0. Each import gets its own
        image directory, so the .sqsh it produced is the only one there and
        this does not have to reproduce spur's image naming rules.

        Unlike build_container_image, the result lands on node 0 alone, so a
        job using it has to be pinned there with `-w`.
        """
        node = self.nodes[0]
        if "ok" not in node.exec_allow_fail("command -v mksquashfs >/dev/null && echo ok"):
            pytest.skip("mksquashfs not found on node 0 (apt install squashfs-tools)")
        image_dir = f"{self.remote_dir}/images/{slug}"
        node.exec(f"mkdir -p '{image_dir}'")
        node.exec(
            f"SPUR_IMAGE_DIR='{image_dir}' '{self.bin_dir}/spur' image import {shlex.quote(uri)}"
        )
        images = node.exec(f"ls '{image_dir}'/*.sqsh").split()
        if len(images) != 1:
            raise RuntimeError(f"expected one imported image under {image_dir}, got {images}")
        return images[0]

    # --- Internal helpers ---

    def _create_dirs(self):
        for node in self.nodes:
            node.exec_allow_fail(f"rm -rf '{self.remote_dir}'")
            node.exec(f"mkdir -p '{self.remote_dir}' '{self.etc_dir}' '{self.state_dir}' '{self.log_dir}'")

    def _resolve_hostnames(self):
        style = os.environ.get("SPUR_TEST_HOSTNAME_STYLE", "short")
        self.node_names = []
        for node in self.nodes:
            if style == "short":
                name = node.exec("hostname -s").strip()
            else:
                name = node.exec("hostname -f").strip()
            if not name:
                name = node.host
            self.node_names.append(name)

    def _default_config(self) -> dict:
        nodes_list = ",".join(self.node_names)
        cfg = {
            "cluster_name": "e2e-test",
            "scheduler": {"interval_secs": 1, "plugin": "backfill"},
            "auth": {"plugin": "none"},
            "network": {"wg_enabled": False, "agent_port": AGENT_PORT},
            "partitions": [
                {
                    "name": "default",
                    "state": "UP",
                    "default": True,
                    "nodes": nodes_list,
                    "max_time": "24:00:00",
                    "default_time": "10:00",
                }
            ],
            "nodes": [
                {"names": name, "cpus": 64, "memory_mb": 262144}
                for name in self.node_names
            ],
        }
        if self.accounting_enabled:
            cfg["accounting"] = {
                "database_url": self._db_url,
                "fairshare_refresh_secs": 10,
            }
        return cfg

    def _write_config(self):
        cfg = self._default_config()
        deep_merge(cfg, self.config_overrides)
        config = tomli_w.dumps(cfg)

        for node in self.nodes:
            node.write_file(f"{self.etc_dir}/spur.conf", config)

    def _start_controller(self):
        listen = f"[::]:{CONTROLLER_PORT}"
        cmd = (
            f"nohup '{self.bin_dir}/spurctld' "
            f"-f '{self.etc_dir}/spur.conf' "
            f"--listen '{listen}' --state-dir '{self.state_dir}' --log-level info -D "
            f"> '{self.log_dir}/spurctld.log' 2>&1 & echo $!"
        )
        pid = self.nodes[0].exec(cmd).strip()
        logger.info("spurctld started on %s (pid %s)", self.node_names[0], pid)

    def _start_postgres(self):
        """Bring up Postgres (Docker) on node 0. Accounting runs inside spurctld."""
        node = self.nodes[0]
        node.exec_allow_fail(
            "c=$(docker ps -aq --filter name=spur-e2e-pg-); "
            "[ -n \"$c\" ] && docker rm -f $c || true"
        )
        node.exec(
            f"docker run -d --name '{self._pg_container}' "
            f"-e POSTGRES_USER=spur -e POSTGRES_PASSWORD=spur -e POSTGRES_DB=spur "
            f"-p 0:5432 mirror.gcr.io/library/postgres:16-alpine"
        )
        self._pg_port = parse_published_port(
            node.exec(
                f"docker inspect --format '{DOCKER_PORT_INSPECT_FMT}' "
                f"'{self._pg_container}'"
            )
        )
        logger.info("postgres published on host port %d", self._pg_port)
        self._wait_pg(node)
        logger.info("postgres ready on %s, spurctld handles accounting", self.node_names[0])

    def _wait_pg(self, node: SshNode, timeout: int = 60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready = node.exec_allow_fail(
                f"docker exec '{self._pg_container}' pg_isready -U spur 2>/dev/null "
                f"&& echo ready || true"
            )
            if "ready" in ready:
                return
            time.sleep(2)
        raise RuntimeError("Postgres did not become ready in time")

    def psql(self, sql: str) -> str:
        """Run *sql* against the accounting database and return stdout.

        Lets a test plant accounting rows that would otherwise take real
        workload hours to accumulate — fair-share usage in particular.
        """
        if not self.accounting_enabled:
            raise RuntimeError("psql() requires the accounting_cluster fixture")
        escaped = sql.replace("'", "'\"'\"'")
        return self.nodes[0].exec(
            f"docker exec '{self._pg_container}' "
            f"psql -U spur -d spur -tAc '{escaped}'"
        )

    def _stop_postgres(self):
        node = self.nodes[0]
        node.exec_allow_fail(f"docker rm -f '{self._pg_container}' 2>/dev/null || true")
        # Next container gets a fresh ephemeral port; drop the resolved one.
        self._pg_port = None

    def _sudo_prefix(self) -> str:
        pw = os.environ.get("SPUR_TEST_SSH_PASSWORD", "")
        if pw:
            escaped = pw.replace("'", "'\"'\"'")
            return f"echo '{escaped}' | sudo -S "
        return "sudo -n "

    def _pkill(self, node: SshNode, pattern: str, *, use_sudo: bool = False):
        prefix = self._sudo_prefix() if use_sudo else ""
        node.exec_allow_fail(f"{prefix}pkill -f '{pattern}' 2>/dev/null || true")

    def _kill_controller(self):
        self._pkill(self.nodes[0], f"{self.bin_dir}/spurctld")

    def _kill_agents(self, use_sudo: bool = False, broad: bool = False):
        for node in self.nodes:
            self._pkill(node, f"{self.bin_dir}/spurd", use_sudo=use_sudo)
            # Kill any process holding the agent port, even from a different
            # bin_dir (e.g. a stale spurd left by a previous pytest session).
            if broad:
                prefix = self._sudo_prefix() if use_sudo else ""
                node.exec_allow_fail(
                    f"{prefix}fuser -k {AGENT_PORT}/tcp 2>/dev/null || true"
                )

    def _spurd_start_cmd(self, node_index: int) -> str:
        node = self.nodes[node_index]
        hostname = self.node_names[node_index]
        address = node.host
        agent_listen = f"0.0.0.0:{AGENT_PORT}"
        # `env VAR=val` runs under any sudo prefix, so the canary lands in spurd's
        # own environment in both the rootless and rootful launch shapes.
        daemon_env = f"env SPUR_DAEMON_ENV_CANARY={DAEMON_ENV_CANARY}"
        spurd_bin = (
            f"{self._sudo_prefix()}{daemon_env} '{self.bin_dir}/spurd'"
            if self.agent_as_root
            else f"{daemon_env} '{self.bin_dir}/spurd'"
        )
        label_args = ""
        labels = self.agent_labels.get(node_index, {})
        if labels:
            label_args = " ".join(f"--label '{k}={v}'" for k, v in labels.items())
            label_args = f" {label_args}"
        token_arg = f" --token '{self.agent_token}'" if self.agent_token else ""
        return (
            f"nohup {spurd_bin} "
            f"-f '{self.etc_dir}/spur.conf' "
            f"--controller '{self.controller_addr}' "
            f"--listen '{agent_listen}' "
            f"--state-dir '{self.state_dir}' "
            f"--hostname '{hostname}' --address '{address}' --log-level info -D"
            f"{label_args}{token_arg} "
            f"> '{self.log_dir}/spurd.log' 2>&1 & echo $!"
        )

    def _start_agents(self):
        for i, node in enumerate(self.nodes):
            cmd = self._spurd_start_cmd(i)
            pid = node.exec(cmd).strip()
            logger.info(
                "spurd started on %s (pid %s, root=%s)",
                self.node_names[i],
                pid,
                self.agent_as_root,
            )

    def _wait_all_idle(self, timeout: int):
        self.wait_ready(timeout)

    def wait_ready(self, timeout: int = 120):
        """Poll sinfo until all provisioned nodes appear idle."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self._cluster_is_ready():
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(
            f"Cluster not ready after {timeout}s. "
            f"Expected {len(self.node_names)} idle nodes.\n"
            f"sinfo output:\n{self.sinfo()}"
        )

    def _cluster_is_ready(self) -> bool:
        states = self.sinfo_nodes()
        if not all(name in states for name in self.node_names):
            return False
        idle = sum(
            1 for name in self.node_names if states[name].startswith("idle")
        )
        return idle >= len(self.node_names)


# --- Job helpers ---


def parse_job_id(sbatch_output: str) -> int | None:
    """Extract job ID from sbatch output like 'Submitted batch job 42'."""
    parts = sbatch_output.split()
    if parts:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return None


# Every canonical ST code (mirrors JobState::code in spur-core).
_JOB_STATE_CODES = frozenset(
    {"PD", "R", "CG", "CD", "F", "CA", "TO", "NF", "PR", "S", "DL", "OOM", "RQ"}
)

# Codes that end a run for good (mirrors JobState::is_terminal). PR/RQ requeue
# back to pending, so they are deliberately not terminal.
_TERMINAL_STATE_CODES = frozenset({"CD", "F", "CA", "TO", "NF", "DL", "OOM"})

# The default `%.2t` squeue column truncates the only three-character code.
_TRUNCATED_STATE_CODES = {"OO": "OOM"}


def job_state(squeue_output: str, job_id: int) -> str | None:
    """Parse job state from squeue -t all output."""
    lines = squeue_output.splitlines()
    if not lines:
        return None
    try:
        state_index = lines[0].split().index("ST")
    except ValueError:
        return None

    id_str = str(job_id)
    for line in lines[1:]:
        fields = line.split()
        if not fields or fields[0] != id_str:
            continue
        if state_index >= len(fields):
            return None
        state = _TRUNCATED_STATE_CODES.get(fields[state_index], fields[state_index])
        return state if state in _JOB_STATE_CODES else None
    return None


def wait_job_state(
    cluster: SpurCluster, job_id: int, state: str, timeout: int = 120
) -> None:
    """Wait until the job reaches the given squeue state (e.g. PD, R)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        sq = cluster.squeue_all()
        if job_state(sq, job_id) == state:
            return
        time.sleep(2)
    raise TimeoutError(
        f"Job {job_id} did not reach state {state} within {timeout}s"
    )


def wait_job(cluster: SpurCluster, job_id: int, timeout: int = 120) -> str:
    """
    Wait for a job to reach a terminal state. Returns the final state string.
    Returns "GONE" if the job disappears from the queue after being seen.
    Raises TimeoutError if the job doesn't finish within the timeout.
    """
    deadline = time.time() + timeout
    last = ""
    seen = False
    while time.time() < deadline:
        sq = cluster.squeue_all()
        state = job_state(sq, job_id)
        if state in _TERMINAL_STATE_CODES:
            return state
        if state is None:
            if seen:
                return last if last else "GONE"
            # Job not visible yet — give scheduler a moment
            time.sleep(1)
            continue
        seen = True
        last = state
        time.sleep(2)
    raise TimeoutError(
        f"Job {job_id} did not finish within {timeout}s (last state: {last})"
    )


def wait_sacct_row(cluster: SpurCluster, job_id: int, fmt: str,
                   timeout: int = 60) -> str:
    """Poll `sacct -j <id> -n -o <fmt>` until the job's row appears.

    Job history is recorded asynchronously after the job ends, so callers
    must poll. Returns the matching row (whitespace-joined). Raises TimeoutError.
    """
    deadline = time.time() + timeout
    id_str = str(job_id)
    last = ""
    while time.time() < deadline:
        out = cluster.sacct(["-j", id_str, "-n", "-o", fmt])
        for line in out.splitlines():
            fields = line.split()
            if fields and fields[0] == id_str:
                return " ".join(fields)
        last = out
        time.sleep(2)
    raise TimeoutError(
        f"Job {job_id} did not appear in sacct within {timeout}s (last: {last!r})"
    )
