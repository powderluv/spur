# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration and fixtures for Spur native-host E2E tests.

See docs/developer/building.rst for full environment variable reference.
"""

import logging
import os
from pathlib import Path

import pytest

from cluster import SshNode, SpurCluster, deep_merge, ensure_bins, make_remote_dir
from wg_cluster import MESH_CIDR, WG_IFACE, WgMesh, wg_available

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


# Requesting any of these fixtures (directly or transitively) makes a test a GPU
# test; conftest auto-marks it `gpu` so CI can route it without a manual tag.
_GPU_FIXTURES = frozenset({"gpu_cluster", "gpu_container_cluster"})


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "mpi: MPI end-to-end tests requiring spur_mpi_pmix.so, libpmix, and mpicc",
    )
    config.addinivalue_line(
        "markers",
        "k0s: native spur-managed k0s cluster tests (rootful spurd + systemd + etcd; slow)",
    )
    config.addinivalue_line(
        "markers",
        "gpu: requires GPU hardware on the nodes (auto-applied to tests using GPU fixtures)",
    )


# tryfirst so the marker lands before pytest's built-in `-m` deselection runs.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        if _GPU_FIXTURES.intersection(getattr(item, "fixturenames", ())):
            item.add_marker("gpu")


def _get_nodes_config() -> list[str]:
    raw = os.environ.get("SPUR_TEST_NODES", "")
    nodes = [n.strip() for n in raw.split(",") if n.strip()]
    if not nodes:
        pytest.exit("SPUR_TEST_NODES not set — cannot run E2E tests", returncode=1)
    return nodes


def _get_ssh_user() -> str:
    user = os.environ.get("SPUR_TEST_SSH_USER", "")
    if not user:
        pytest.exit("SPUR_TEST_SSH_USER not set — cannot run E2E tests", returncode=1)
    return user


def _get_ssh_password() -> str | None:
    return os.environ.get("SPUR_TEST_SSH_PASSWORD") or None


def _get_ssh_key() -> str | None:
    key = os.environ.get("SPUR_TEST_SSH_KEY", "")
    return key if key else None


def _get_binaries_dir() -> str:
    return os.environ.get(
        "SPUR_TEST_BINARIES_DIR",
        str(_REPO_ROOT / "target" / "release"),
    )


@pytest.fixture(scope="session")
def ssh_nodes():
    """
    Session-scoped SSH connections to all nodes.
    Stays open for the entire test run.
    """
    nodes_config = _get_nodes_config()
    ssh_user = _get_ssh_user()
    ssh_password = _get_ssh_password()
    ssh_key = _get_ssh_key()

    nodes = []
    for host in nodes_config:
        node = SshNode(host, ssh_user, password=ssh_password, key_path=ssh_key)
        nodes.append(node)

    yield nodes

    for node in nodes:
        node.close()


@pytest.fixture(scope="session")
def remote_bin_dir(ssh_nodes, tmp_path_factory):
    """
    Session-scoped remote directory for binaries.

    If SPUR_TEST_REMOTE_BIN_DIR is set, uses that fixed path (not cleaned up).
    This is useful for CI where a predictable path is needed for AppArmor profiles.

    Otherwise, generates an ephemeral path from tmp_path_factory and cleans up
    at session end.
    """
    fixed = os.environ.get("SPUR_TEST_REMOTE_BIN_DIR", "")
    if fixed:
        yield fixed
        return

    session_tmp = tmp_path_factory.getbasetemp()
    remote_path = f"/tmp/spur-e2e-bin-{session_tmp.name}"

    yield remote_path

    for node in ssh_nodes:
        node.exec_allow_fail(f"rm -rf '{remote_path}'")


@pytest.fixture(scope="session", autouse=True)
def _ensure_bins(ssh_nodes, remote_bin_dir):
    """
    Session-scoped: uploads binaries to all nodes once.
    Skips upload if binary already exists with matching size.
    """
    ensure_bins(ssh_nodes, _get_binaries_dir(), remote_bin_dir)


def _deploy_cluster(ssh_nodes, remote_bin_dir, *, agent_as_root: bool = False,
                    config_overrides: dict | None = None,
                    agent_labels: dict[int, dict[str, str]] | None = None):
    """Helper: create, deploy, and return a SpurCluster. Tears down on deploy failure."""
    c = SpurCluster(ssh_nodes, make_remote_dir(), remote_bin_dir)
    try:
        c.deploy(config_overrides=config_overrides, agent_as_root=agent_as_root,
                 agent_labels=agent_labels)
    except Exception:
        c.teardown()
        raise
    return c


def _provision_cluster(ssh_nodes, remote_bin_dir):
    """Helper: create and provision (but do not start) a SpurCluster."""
    c = SpurCluster(ssh_nodes, make_remote_dir(), remote_bin_dir)
    try:
        c.provision()
    except Exception:
        c.teardown()
        raise
    return c


@pytest.fixture
def cluster_config_overrides():
    """Override this fixture in tests/classes to customise the cluster config.

    Return a dict that will be deep-merged into the default config before
    spurctld starts.  The default (no overrides) returns None.
    """
    return None


@pytest.fixture
def cluster(ssh_nodes, remote_bin_dir, cluster_config_overrides):
    """
    Per-test fixture: a fully running Spur cluster with default config.
    Torn down (processes killed, dirs removed) after the test.
    """
    spur_cluster = _deploy_cluster(ssh_nodes, remote_bin_dir,
                                   config_overrides=cluster_config_overrides)
    yield spur_cluster
    spur_cluster.teardown()


@pytest.fixture
def unstarted_cluster(ssh_nodes, remote_bin_dir):
    """
    Per-test fixture: a provisioned cluster (dirs created, hostnames
    resolved) but **not started**.

    The test should write any scripts or files it needs, then call
    ``cluster.start(config_overrides)`` to bring up the daemons with
    the desired configuration.
    """
    spur_cluster = _provision_cluster(ssh_nodes, remote_bin_dir)
    yield spur_cluster
    spur_cluster.teardown()


@pytest.fixture
def multi_node_cluster(ssh_nodes, remote_bin_dir, cluster_config_overrides):
    """
    Per-test fixture for multi-node tests.
    Skips if fewer than 2 nodes are configured.
    """
    if len(ssh_nodes) < 2:
        pytest.skip(
            f"Multi-node tests require at least 2 nodes in SPUR_TEST_NODES "
            f"(got {len(ssh_nodes)})"
        )

    spur_cluster = _deploy_cluster(ssh_nodes, remote_bin_dir,
                                   config_overrides=cluster_config_overrides)
    yield spur_cluster
    spur_cluster.teardown()


@pytest.fixture
def cgroup_cluster(ssh_nodes, remote_bin_dir, cluster_config_overrides):
    """
    Per-test fixture for cgroup enforcement tests: a rootful, non-GPU cluster.

    spurd must be root to create ``/sys/fs/cgroup/spur/job_<id>_<attempt>``; an
    unprivileged agent degrades to "no isolation" and every limit assertion
    would then pass vacuously, so the agent's user is checked up front.
    """
    if len(ssh_nodes) < 1:
        pytest.skip("cgroup tests require at least one node in SPUR_TEST_NODES")
    fstype = ssh_nodes[0].exec_allow_fail("stat -fc %T /sys/fs/cgroup").strip()
    if "cgroup2fs" not in fstype:
        pytest.skip(f"node 0 is not running cgroup v2 (/sys/fs/cgroup is {fstype!r})")

    c = _deploy_cluster(ssh_nodes, remote_bin_dir, agent_as_root=True,
                        config_overrides=cluster_config_overrides)
    try:
        agent_user = c.spurd_agent_user(0)
        assert agent_user == "root", (
            f"cgroup enforcement needs a rootful agent, got user {agent_user!r}"
        )
        yield c
    finally:
        c.teardown()


@pytest.fixture
def k8s_multicp_cluster(ssh_nodes, remote_bin_dir):
    """Native k0s enabled, spurd rootful. Skips unless >= 3 nodes (etcd quorum) + sudo."""
    if len(ssh_nodes) < 3:
        pytest.skip(
            f"multi-CP k0s tests require at least 3 nodes for an etcd quorum "
            f"(got {len(ssh_nodes)})"
        )
    c = SpurCluster(ssh_nodes, make_remote_dir(), remote_bin_dir)
    c.provision()
    c.root_agent_preflight()
    try:
        c.start(
            config_overrides={"cluster": {"enabled": True, "cni": "kuberouter"}},
            agent_as_root=True,
        )
    except Exception:
        c.teardown()
        raise
    yield c
    try:
        c.k8s_down(reset=True)
        c.wait_k8s_phase("down", timeout=180)
    except Exception:
        pass
    c.teardown()


@pytest.fixture
def k8s_calico_direct_cluster(ssh_nodes, remote_bin_dir):
    """Native k0s, calico CNI, WireGuard mesh OFF (`network.wg_enabled` unset ->
    False). A single control plane needs no etcd quorum, so this only needs 2
    nodes — unlike ``k8s_multicp_cluster``'s HA scenarios."""
    if len(ssh_nodes) < 2:
        pytest.skip(
            f"a single-CP k0s cluster needs at least 2 nodes (got {len(ssh_nodes)})"
        )
    c = SpurCluster(ssh_nodes, make_remote_dir(), remote_bin_dir)
    c.provision()
    c.root_agent_preflight()
    _reset_k0s_all_nodes(c)
    try:
        c.start(
            config_overrides={"cluster": {"enabled": True, "cni": "calico"}},
            agent_as_root=True,
        )
    except Exception:
        c.teardown()
        raise
    yield c
    try:
        c.k8s_down(reset=True)
        c.wait_k8s_phase("down", timeout=180)
    except Exception:
        pass
    c.teardown()
    _reset_k0s_all_nodes(c)


@pytest.fixture
def accounting_cluster(ssh_nodes, remote_bin_dir, cluster_config_overrides):
    """
    Per-test fixture: a running cluster with Postgres on node 0.

    Accounting runs inside spurctld. Skips if node 0 lacks Docker.
    """
    if len(ssh_nodes) < 1:
        pytest.skip("accounting tests require at least one node")
    try:
        ensure_bins(ssh_nodes[:1], _get_binaries_dir(), remote_bin_dir,
                    with_accounting=True)
    except FileNotFoundError as e:
        pytest.skip(f"accounting binaries unavailable: {e}")

    c = SpurCluster(ssh_nodes, make_remote_dir(), remote_bin_dir)
    try:
        c.enable_accounting()
    except RuntimeError as e:
        pytest.skip(str(e))
    try:
        c.deploy(config_overrides=cluster_config_overrides)
    except Exception:
        c.teardown()
        raise
    yield c
    c.teardown()


def _any_node_has_gpu(nodes: list[SshNode]) -> bool:
    for node in nodes:
        probe = node.exec_allow_fail(
            "ls /dev/kfd /dev/dri/card* /dev/dri/renderD* 2>/dev/null | head -1"
        )
        if probe.strip():
            return True
    return False


@pytest.fixture
def gpu_cluster(request, ssh_nodes, remote_bin_dir):
    """
    Per-test fixture for GPU tests.

    Skips the entire test if no node has GPU device nodes.
    Decorate a test with ``@pytest.mark.rootful`` to launch spurd via sudo.
    """
    if len(ssh_nodes) < 1:
        pytest.skip("GPU tests require at least one node in SPUR_TEST_NODES")
    if not _any_node_has_gpu(ssh_nodes):
        pytest.skip("no GPU device nodes (/dev/kfd, /dev/dri/card*, /dev/dri/renderD*) on any node")

    as_root = request.node.get_closest_marker("rootful") is not None
    c = _deploy_cluster(ssh_nodes, remote_bin_dir, agent_as_root=as_root)
    yield c
    c.teardown()


@pytest.fixture(scope="class")
def label_cluster(ssh_nodes, remote_bin_dir):
    """Class-scoped cluster for node label and partition selector tests."""
    if len(ssh_nodes) < 2:
        pytest.skip(
            f"Label cluster requires at least 2 nodes in SPUR_TEST_NODES "
            f"(got {len(ssh_nodes)})"
        )

    c = _deploy_cluster(
        ssh_nodes,
        remote_bin_dir,
        config_overrides={
            "partitions": [
                {
                    "name": "gpu",
                    "state": "UP",
                    "selector": {"gpu": "mi300x"},
                    "max_time": "1:00:00",
                },
                {
                    "name": "catchall",
                    "state": "UP",
                    "default": True,
                    "nodes": "ALL",
                    "max_time": "1:00:00",
                },
            ],
        },
        agent_labels={0: {"gpu": "mi300x"}},
    )
    yield c
    c.teardown()


@pytest.fixture
def mpi_multi_node_cluster(ssh_nodes, remote_bin_dir, cluster_config_overrides):
    """Multi-node MPI tests with spur_mpi_pmix.so deployed."""
    if len(ssh_nodes) < 2:
        pytest.skip(
            f"Multi-node MPI tests require at least 2 nodes in SPUR_TEST_NODES "
            f"(got {len(ssh_nodes)})"
        )
    try:
        ensure_bins(
            ssh_nodes,
            _get_binaries_dir(),
            remote_bin_dir,
            with_mpi_plugin=True,
        )
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    plugin_dir = str(Path(remote_bin_dir).parent / "lib" / "spur")
    mpi_cfg = {
        "mpi": {
            "plugin_dir": plugin_dir,
            "pmix_tmpdir": "/tmp/spur-pmix",
        }
    }
    overrides = cluster_config_overrides or {}
    merged = deep_merge(dict(overrides), mpi_cfg) if isinstance(overrides, dict) else mpi_cfg

    c = _deploy_cluster(ssh_nodes, remote_bin_dir, config_overrides=merged)
    c.mpi_preflight(2)
    yield c
    c.teardown()


@pytest.fixture
def mpi_cluster(ssh_nodes, remote_bin_dir, cluster_config_overrides):
    """Single-node MPI tests with spur_mpi_pmix.so deployed."""
    if len(ssh_nodes) < 1:
        pytest.skip("MPI tests require at least one node in SPUR_TEST_NODES")
    try:
        ensure_bins(
            ssh_nodes,
            _get_binaries_dir(),
            remote_bin_dir,
            with_mpi_plugin=True,
        )
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    plugin_dir = str(Path(remote_bin_dir).parent / "lib" / "spur")
    mpi_cfg = {
        "mpi": {
            "plugin_dir": plugin_dir,
            "pmix_tmpdir": "/tmp/spur-pmix",
        }
    }
    overrides = cluster_config_overrides or {}
    merged = deep_merge(dict(overrides), mpi_cfg) if isinstance(overrides, dict) else mpi_cfg

    c = _deploy_cluster(ssh_nodes, remote_bin_dir, config_overrides=merged)
    c.mpi_preflight(1)
    yield c
    c.teardown()


# --- WireGuard mesh fixtures -------------------------------------------------
#
# WireGuard tests run wherever the nodes can run a real mesh — no opt-in env var
# (mirroring gpu_cluster / k8s_multicp_cluster, which auto-run on capable beds).
# The fixtures flip network.wg_enabled on in config and stand up the spur0 mesh;
# they DETECT the wg data plane and skip cleanly when it is absent (a test does
# not install its own dependencies — nodes are expected to ship WireGuard).

# Test pod / service CIDRs, clear of the mesh CIDR and the CI runner underlay.
POD_CIDR = "10.47.0.0/16"
SERVICE_CIDR = "10.48.0.0/16"


def _wg_addresses() -> dict[int, str] | None:
    """Optional explicit per-node mesh addresses, positional like SPUR_TEST_NODES.

    Mirrors spur-toolkit's per-node ``spur_wg_address`` inventory var. Example:
    ``SPUR_TEST_WG_ADDRESSES=10.46.0.1,10.46.0.2,10.46.0.3``. When unset, WgMesh
    falls back to its default assignment. When set, the count must match
    SPUR_TEST_NODES.
    """
    raw = os.environ.get("SPUR_TEST_WG_ADDRESSES", "").strip()
    if not raw:
        return None
    addrs = [a.strip() for a in raw.split(",") if a.strip()]
    nodes = _get_nodes_config()
    if len(addrs) != len(nodes):
        pytest.exit(
            f"SPUR_TEST_WG_ADDRESSES has {len(addrs)} entries but SPUR_TEST_NODES "
            f"has {len(nodes)} — they must match", returncode=1
        )
    return {i: a for i, a in enumerate(addrs)}


def _require_wireguard(cluster: SpurCluster) -> None:
    """Skip unless every node can run a real WireGuard mesh.

    Checks ``wg``/``wg-quick`` plus a usable data plane (kernel module or the
    ``wireguard-go`` userspace fallback) via :func:`wg_available`. Nodes are
    expected to ship WireGuard (CI bakes it into the image); a test does not
    install its own dependencies, so a node without it skips rather than failing.
    """
    sudo = cluster._sudo_prefix()
    for node in cluster.nodes:
        ok, reason = wg_available(node, sudo)
        if not ok:
            pytest.skip(f"WireGuard unavailable on {node.host}: {reason}")


@pytest.fixture
def raw_wg_mesh(ssh_nodes, remote_bin_dir):
    """A raw WireGuard mesh across all nodes (no spur daemons).

    Provisions a :class:`SpurCluster` only to resolve node names, then stands up
    the ``spur0`` interface via ``net init`` on node 0 and ``net join`` on the
    rest, promoted to a full mesh. Unlike the other WG fixtures, no spurctld/spurd
    runs underneath. Fully torn down (interfaces + confs removed) after the test
    so a shared node is left clean.
    """
    if len(ssh_nodes) < 2:
        pytest.skip(f"WG mesh needs >= 2 nodes (got {len(ssh_nodes)})")
    c = SpurCluster(ssh_nodes, make_remote_dir(), remote_bin_dir)
    c.provision()  # resolves node_names; no daemons started
    c.root_agent_preflight()  # rootful: spur0 needs wg/ip via sudo; skips if none
    _require_wireguard(c)
    mesh = WgMesh(ssh_nodes, c.node_names, remote_bin_dir, c._sudo_prefix(),
                  iface=WG_IFACE, wg_addresses=_wg_addresses())
    indices = list(range(len(ssh_nodes)))
    try:
        mesh.bring_up(indices)
    except Exception:
        mesh.teardown(indices)
        c.teardown()
        raise
    yield mesh
    mesh.teardown(indices)
    c.teardown()


def _reset_k0s_all_nodes(c):
    """Forcibly reset k0s on every node, independent of the controller.

    `spur k8s down --reset` only flags phase=Down; the controller's reconcile
    loop stops the k0s components *asynchronously*. Because the fixture kills
    spurctld/spurd right after (c.teardown()), that async reset is cut off — so
    without this, k0s services and the etcd datadir survive and the NEXT wg_k0s
    test inherits dirty state (membership won't shrink, the API server refuses
    connections). Reset each node directly so every test starts from a clean slate.
    """
    sudo = c._sudo_prefix()
    for node in c.nodes:
        node.exec_allow_fail(
            f"{sudo}systemctl stop k0scontroller k0sworker 2>/dev/null || true"
        )
        node.exec_allow_fail(f"{sudo}k0s stop 2>/dev/null || true")
        node.exec_allow_fail(f"{sudo}k0s reset 2>/dev/null || true")
        node.exec_allow_fail(
            f"{sudo}rm -rf /etc/k0s /var/lib/k0s /run/k0s 2>/dev/null || true"
        )
        # Best-effort steps above (|| true) can't wedge teardown — but a reset
        # that silently failed would resurface as a baffling failure in the NEXT
        # test that inherits the leftover k0s state. Verify the datadir is gone
        # and warn (naming the node) if not, so blame lands on the offender.
        leftover = node.exec_allow_fail(
            "test -e /var/lib/k0s && echo DIRTY || echo CLEAN"
        )
        if "DIRTY" in leftover:
            logger.warning(
                "k0s reset left /var/lib/k0s behind on %s — the next wg_k0s test "
                "may inherit dirty state (stale etcd membership / API refusals)",
                node.host,
            )


def _deploy_wg_k0s(ssh_nodes, remote_bin_dir, *, control_plane_index: int = 0):
    """Deploy a rootful, WG-enabled, calico k0s cluster over the mesh.

    Returns a running SpurCluster (daemons up, k0s NOT yet `up`) with the mesh
    established. ``wg_enabled=true`` only tells spurd to *read* its mesh key/IP
    from ``spur0`` — it does not create the interface, so the mesh is stood up
    (``net init``/``join``) before the daemons start, else spurd reports no
    ``wg_pubkey`` and the reconcile runs meshless.

    The live :class:`WgMesh` is attached as ``c.wg_mesh`` so the fixture can tear
    it down; ``c.wg_mesh_indices`` records which nodes it covers.
    """
    c = SpurCluster(ssh_nodes, make_remote_dir(), remote_bin_dir)
    c.provision()
    c.root_agent_preflight()  # skips if no sudo
    _require_wireguard(c)

    # Start from a clean k0s slate: a prior test (or a crashed run) can leave k0s
    # services + an etcd datadir behind, which corrupts this cluster's membership.
    _reset_k0s_all_nodes(c)

    # Establish the mesh first (spur0 up on every node) so spurd reads its
    # wg_pubkey + mesh IP at startup and the reconcile has a real mesh to
    # converge. bring_up() idempotently pre-cleans any stale interface.
    indices = list(range(len(ssh_nodes)))
    mesh = WgMesh(ssh_nodes, c.node_names, remote_bin_dir, c._sudo_prefix(),
                  iface=WG_IFACE, wg_addresses=_wg_addresses())
    try:
        mesh.bring_up(indices)
    except Exception:
        mesh.teardown(indices)
        c.teardown()
        raise
    c.wg_mesh = mesh
    c.wg_mesh_indices = indices

    overrides = {
        "network": {"wg_enabled": True, "wg_cidr": MESH_CIDR},
        "cluster": {"enabled": True, "cni": "calico",
                    "control_plane_node": c.node_names[control_plane_index],
                    "pod_cidr": POD_CIDR, "service_cidr": SERVICE_CIDR},
    }
    try:
        c.start(config_overrides=overrides, agent_as_root=True)
    except Exception:
        mesh.teardown(indices)
        c.teardown()
        raise
    return c


def _teardown_wg_k0s(c):
    """Tear down a cluster from _deploy_wg_k0s: daemons first, then k0s, then mesh."""
    # Ask the controller to bring k0s down (best effort), but do NOT rely on the
    # async reset completing — force a direct per-node k0s reset below.
    try:
        c.k8s_down(reset=True)
        c.wait_k8s_phase("down", timeout=120)
    except Exception:
        pass
    # Kill spurctld/spurd BEFORE the manual k0s reset: k8s_down(reset=True) kicks
    # off an async reconcile that stops the k0s components itself, so if the
    # daemons are still alive while _reset_k0s_all_nodes runs its `k0s reset`/
    # `rm -rf`, that reconcile races the manual cleanup. Tearing the daemons down
    # first leaves nothing contending with it.
    c.teardown()
    _reset_k0s_all_nodes(c)
    mesh = getattr(c, "wg_mesh", None)
    if mesh is not None:
        mesh.teardown(getattr(c, "wg_mesh_indices", []))


@pytest.fixture
def wg_k0s_cluster(ssh_nodes, remote_bin_dir):
    """Rootful k0s-over-WireGuard cluster (calico, wg_enabled). Needs >= 3 nodes
    for an etcd quorum path and rootful sudo. Tears k0s + mesh down on exit."""
    if len(ssh_nodes) < 3:
        pytest.skip(f"k0s-over-mesh needs >= 3 nodes for quorum (got {len(ssh_nodes)})")
    c = _deploy_wg_k0s(ssh_nodes, remote_bin_dir)
    yield c
    _teardown_wg_k0s(c)


@pytest.fixture
def wg_login_cluster(ssh_nodes, remote_bin_dir):
    """k0s-over-mesh cluster wired for the login-node reachability scenario:

      node0 = spur controller (spurctld + spurd, mesh head, NOT in k0s scope)
      node1 = k8s control-plane (spurd, k0s control plane)
      node2 = login node (spurd, meshed, NOT in k0s scope)

    All three are meshed; only node1 is in the k0s scope. Same running
    :class:`SpurCluster` as ``wg_k0s_cluster``, just with the control plane pinned
    to node1.
    """
    if len(ssh_nodes) < 3:
        pytest.skip(f"login-node topology needs 3 nodes (got {len(ssh_nodes)})")
    c = _deploy_wg_k0s(ssh_nodes, remote_bin_dir, control_plane_index=1)
    yield c
    _teardown_wg_k0s(c)
