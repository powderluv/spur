# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for cgroup-v2 job resource enforcement.

The limits spurd writes derive from the controller's per-node allocation rather
than from what the user asked for, so these assert on the live control files
instead of on the submitted request. Each job reads its own cgroup through
``/proc/self/cgroup``, which needs no privilege — only the agent must be root,
which the ``cgroup_cluster`` fixture enforces.
"""

import time
from typing import NamedTuple

import pytest

from cluster import parse_job_id, wait_job, wait_job_state

MIB = 1024 * 1024

# Pure bash on purpose: the minimal container image has no awk or cut, and the
# same probe is reused there.
_PROBE = """#!/bin/bash
CG=""
while IFS= read -r line; do
  case "$line" in 0::*) CG="${line#0::}" ;; esac
done < /proc/self/cgroup
echo "CGROUP_PATH=$CG"
B="/sys/fs/cgroup$CG"
for f in cpu.max cpuset.cpus memory.max memory.high memory.swap.max \
         memory.oom.group pids.max; do
  if [ -r "$B/$f" ]; then echo "$f=$(cat "$B/$f")"; else echo "$f=UNREADABLE"; fi
done
echo CGROUP_PROBE_OK
"""


class _Probe(NamedTuple):
    values: dict[str, str]
    job_id: int
    output: str

    def context(self) -> str:
        return f"job {self.job_id} probe output:\n{self.output}"


def _parse(output: str) -> dict[str, str]:
    values = {}
    for line in output.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip()
    return values


def _core_count(cpuset: str) -> int:
    """Number of cores in a cgroup cpuset list.

    The kernel normalises what spurd writes (``0,1,2,3``) into ranges
    (``0-3``), so both spellings have to parse.
    """
    total = 0
    for part in cpuset.split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition("-")
        total += int(high) - int(low) + 1 if high else 1
    return total


def _require_cores(cluster, needed: int) -> None:
    cores = int(cluster.nodes[0].exec("nproc").strip())
    if cores < needed:
        pytest.skip(f"need a node with at least {needed} cores (node 0 has {cores})")


def _spurd_logs(cluster) -> str:
    """Every agent log joined: an unpinned job lands on whichever node the
    scheduler picked, so node 0 alone is not where the evidence has to be."""
    return "\n".join(cluster.spurd_log(i) for i in range(len(cluster.nodes)))


def _run_probe(
    cluster, sbatch_args: list[str], name: str, *, expect_enforced: bool = True
) -> _Probe:
    """Submit the probe pinned to node 0 and return its parsed cgroup values.

    With *expect_enforced* the job is required to have landed in its own
    ``/spur/job_<id>_<attempt>`` cgroup (attempt is always 1 for a fresh,
    non-requeued job). Checking that here turns "every control file reads
    UNREADABLE" into one legible failure naming the cgroup it did land in,
    which is otherwise an easy symptom to misread.
    """
    script = cluster.write_file(f"{name}.sh", _PROBE)
    out_path = f"{cluster.remote_dir}/{name}.out"
    sb = cluster.sbatch(
        ["-J", name, "-N", "1", "-w", cluster.node_names[0], "-o", out_path]
        + sbatch_args
        + [script]
    )
    job_id = parse_job_id(sb)
    assert job_id is not None, f"sbatch failed: {sb}"

    wait_job(cluster, job_id, timeout=120)
    content = cluster.wait_output(out_path, "CGROUP_PROBE_OK", timeout=120)
    assert "CGROUP_PROBE_OK" in content, (
        f"probe did not run to completion\n{cluster.debug_job(job_id)}\n"
        f"output:\n{content}"
    )

    probe = _Probe(_parse(content), job_id, content)
    if expect_enforced:
        assert probe.values.get("CGROUP_PATH") == f"/spur/job_{job_id}_1", (
            f"job ran outside its own cgroup, so no limit was applied to it "
            f"(agent user: {cluster.spurd_agent_user(0)!r})\n{probe.context()}\n"
            f"spurd log:\n{cluster.spurd_log(0)[-2000:]}"
        )
    return probe


class TestCgroupDefaults:
    def test_limits_match_the_node_allocation(self, cgroup_cluster):
        cluster = cgroup_cluster
        _require_cores(cluster, 2)
        probe = _run_probe(
            cluster, ["--cpus-per-task=2", "--mem=1024"], "cg-alloc"
        )
        vals = probe.values

        assert vals["memory.max"] == str(1024 * MIB), probe.context()
        # allowed_ram_percent defaults to 100, which collapses the soft ceiling
        # onto the hard one.
        assert vals["memory.high"] == vals["memory.max"], probe.context()
        # Swap is unconstrained by default, as in Slurm.
        assert vals["memory.swap.max"] == "max", probe.context()
        assert vals["memory.oom.group"] == "1", probe.context()
        assert _core_count(vals["cpuset.cpus"]) == 2, probe.context()
        # Slurm bounds CPU with the cpuset alone; the CFS quota is opt-in.
        assert vals["cpu.max"] == "max 100000", probe.context()

    def test_cpuset_covers_every_task_on_the_node(self, cgroup_cluster):
        # Sizing the budget from `cpus_per_task` alone would pin 2 cores here
        # instead of 4, which is the under-provisioning this closes.
        cluster = cgroup_cluster
        _require_cores(cluster, 4)
        probe = _run_probe(
            cluster,
            ["--ntasks-per-node=2", "--cpus-per-task=2", "--mem=512"],
            "cg-multitask",
        )
        assert _core_count(probe.values["cpuset.cpus"]) == 4, probe.context()

    def test_job_without_a_memory_request_is_left_unbounded(self, cgroup_cluster):
        # No memory budget means no memory ceiling, and so no swap ceiling either:
        # swap 0 with RAM unlimited is a pair Slurm never emits.
        probe = _run_probe(cgroup_cluster, ["--cpus-per-task=1"], "cg-nomem")
        vals = probe.values

        assert vals["memory.max"] == "max", probe.context()
        assert vals["memory.high"] == "max", probe.context()
        assert vals["memory.swap.max"] == "max", probe.context()

    def test_memory_ceiling_floors_at_min_ram_mb(self, cgroup_cluster):
        # Below the floor a job is OOM-killed during its own startup, before
        # anything can clean up after it.
        probe = _run_probe(
            cgroup_cluster, ["--cpus-per-task=1", "--mem=8"], "cg-tinymem"
        )
        assert probe.values["memory.max"] == str(30 * MIB), probe.context()
        assert probe.values["memory.high"] == str(30 * MIB), probe.context()

    def test_container_job_runs_inside_the_job_cgroup(self, cgroup_cluster, tmp_path):
        # Containers share the batch cgroup, so the risk is the pid-namespace tree
        # escaping it. /sys/fs/cgroup is unmounted inside, so read /proc/self/cgroup.
        cluster = cgroup_cluster
        cluster.container_preflight()
        image = cluster.build_container_image(tmp_path)
        probe = _run_probe(
            cluster,
            ["--cpus-per-task=1", "--mem=256", f"--container-image={image}"],
            "cg-container",
        )
        assert probe.values["CGROUP_PATH"] == f"/spur/job_{probe.job_id}_1", (
            probe.context()
        )


class TestCgroupCpuQuota:
    @pytest.fixture
    def cluster_config_overrides(self):
        return {"cgroup": {"cpu_quota": True}}

    def test_cpu_quota_opt_in_writes_cpu_max(self, cgroup_cluster):
        cluster = cgroup_cluster
        _require_cores(cluster, 2)
        probe = _run_probe(
            cluster, ["--cpus-per-task=2", "--mem=512"], "cg-quota"
        )
        assert probe.values["cpu.max"] == "200000 100000", probe.context()


class TestCgroupRamHeadroom:
    @pytest.fixture
    def cluster_config_overrides(self):
        return {"cgroup": {"allowed_ram_percent": 150}}

    def test_headroom_splits_the_soft_and_hard_ceilings(self, cgroup_cluster):
        # memory.high stays at the allocation so reclaim starts there, while
        # memory.max moves out to the configured headroom.
        probe = _run_probe(
            cgroup_cluster, ["--cpus-per-task=1", "--mem=1024"], "cg-headroom"
        )
        assert probe.values["memory.high"] == str(1024 * MIB), probe.context()
        assert probe.values["memory.max"] == str(1536 * MIB), probe.context()


class TestCgroupSwapOnly:
    @pytest.fixture
    def cluster_config_overrides(self):
        return {
            "cgroup": {
                "constrain_ram_space": False,
                "constrain_swap": True,
                "allowed_swap_percent": 25,
            }
        }

    def test_swap_only_still_bounds_the_ram_plus_swap_total(self, cgroup_cluster):
        # Constraining swap alone must not leave RAM unlimited: the RAM ceiling
        # becomes the combined total, and swap itself is then pinned to zero.
        probe = _run_probe(
            cgroup_cluster, ["--cpus-per-task=1", "--mem=1024"], "cg-swaponly"
        )
        assert probe.values["memory.max"] == str(1280 * MIB), probe.context()
        assert probe.values["memory.high"] == str(1024 * MIB), probe.context()
        assert probe.values["memory.swap.max"] == "0", probe.context()


# A child grows a heap string until the cgroup kills it; the parent records only
# whether it outlived the child, which `memory.oom.group` decides.
def _oom_probe(marker: str) -> str:
    return (
        "#!/bin/bash\n"
        f'marker="{marker}"\n'
        'rm -f "$marker" 2>/dev/null\n'
        "grow() { local a; a=$(head -c 1048576 /dev/zero | tr '\\0' x); "
        'while :; do a="$a$a"; done; }\n'
        "grow &\n"
        "child=$!\n"
        'wait "$child"\n'
        "# Reached only if this shell was not group-killed alongside the child.\n"
        'echo "PARENT_SURVIVED rc=$?" > "$marker"\n'
    )


def _run_oom_job(cluster, name: str) -> tuple[int, str, str]:
    """Submit the OOM probe pinned to node 0 under a 64 MiB ceiling and return
    (job_id, terminal state, survivor-marker contents)."""
    marker = f"{cluster.remote_dir}/{name}-survivor.txt"
    script = cluster.write_file(f"{name}.sh", _oom_probe(marker))
    sb = cluster.sbatch(
        ["-J", name, "-N", "1", "-w", cluster.node_names[0], "-t", "1",
         "--cpus-per-task=1", "--mem=64", script]
    )
    job_id = parse_job_id(sb)
    assert job_id is not None, f"sbatch failed: {sb}"
    state = wait_job(cluster, job_id, timeout=120)
    return job_id, state, cluster.nodes[0].read_file(marker)


# A step marks that it launched, then allocates far past the job ceiling. The
# second marker is written only if that allocation was never bounded.
def _step_mem_probe(marker: str) -> str:
    return (
        "#!/bin/bash\n"
        f'marker="{marker}"\n'
        'echo STEP_STARTED > "$marker"\n'
        "a=$(head -c $((160*1024*1024)) /dev/zero | tr '\\0' x)\n"
        'echo "STEP_OK len=${#a}" >> "$marker"\n'
    )


class TestCgroupOomGroupKill:
    # Default `oom_kill_job = true`: the existing tests assert memory.oom.group
    # reads 1; this asserts the effect.
    def test_oom_kill_group_takes_the_whole_job(self, cgroup_cluster):
        # oom.group kills every process in the cgroup, so the parent dies with the
        # child and never records that it survived; the job ends OUT_OF_MEMORY.
        job_id, state, marker = _run_oom_job(cgroup_cluster, "cg-oom-group")
        assert state == "OOM", (
            f"a job that exceeds memory.max must end OUT_OF_MEMORY, got {state}\n"
            f"{cgroup_cluster.debug_job(job_id)}"
        )
        assert "PARENT_SURVIVED" not in marker, (
            f"oom.group must kill the parent too, but it survived:\n{marker!r}"
        )


class TestCgroupOomSingleKill:
    @pytest.fixture
    def cluster_config_overrides(self):
        return {"cgroup": {"oom_kill_job": False}}

    def test_oom_kill_disabled_spares_the_siblings(self, cgroup_cluster):
        # Only the offending process is killed, so the parent outlives the child's
        # OOM. The job is still OUT_OF_MEMORY (a kill happened); the surviving
        # parent's marker is what proves the knob spared it.
        job_id, state, marker = _run_oom_job(cgroup_cluster, "cg-oom-single")
        assert "PARENT_SURVIVED" in marker, (
            f"with oom_kill_job = false only the child dies, so the parent must "
            f"outlive its OOM; marker was:\n{marker!r}\n"
            f"{cgroup_cluster.debug_job(job_id)}"
        )
        assert state == "OOM", (
            f"an OOM kill still marks the job OUT_OF_MEMORY, got {state}\n"
            f"{cgroup_cluster.debug_job(job_id)}"
        )


class TestCgroupStepMemoryBudget:
    """Phase 2a puts an ``srun`` step in the job's own cgroup, so a step's memory
    counts against the job's ``memory.max``. Before it, a step ran cgroup-free and
    could allocate past the job's ceiling unchecked.
    """

    def test_a_step_is_bound_by_the_jobs_memory_max(self, cgroup_cluster):
        cluster = cgroup_cluster
        marker = f"{cluster.remote_dir}/cg-step-mem-marker.txt"
        step = cluster.write_file("cg-step-mem-probe.sh", _step_mem_probe(marker))
        # The batch parent stays tiny; only the step allocates, so an OOM here can
        # only be the step's 160 MiB charged to the job's 64 MiB memory.max.
        script = cluster.write_file(
            "cg-step-mem.sh", "#!/bin/bash\n" f"srun bash {step}\n"
        )
        sb = cluster.sbatch(
            ["-J", "cg-step-mem", "-N", "1", "-w", cluster.node_names[0], "-t", "2",
             "--cpus-per-task=1", "--mem=64", script]
        )
        job_id = parse_job_id(sb)
        assert job_id is not None, f"sbatch failed: {sb}"
        state = wait_job(cluster, job_id, timeout=120)
        marker_out = cluster.nodes[0].read_file(marker)

        assert "STEP_STARTED" in marker_out, (
            f"the srun step never launched, so nothing was tested\n"
            f"{cluster.debug_job(job_id)}\nmarker:\n{marker_out!r}"
        )
        assert "STEP_OK" not in marker_out, (
            f"a step's memory must count against the job's memory.max: its 160 MiB "
            f"allocation completed under a 64 MiB ceiling (state {state})\n"
            f"marker:\n{marker_out!r}"
        )
        # The step shares the job cgroup, so oom.group takes the whole job: it ends
        # OUT_OF_MEMORY rather than the over-budget step dying alone.
        assert state == "OOM", (
            f"the job must end OUT_OF_MEMORY when its step exceeds memory.max, got "
            f"{state}\n{cluster.debug_job(job_id)}"
        )


class TestCgroupDisabled:
    @pytest.fixture
    def cluster_config_overrides(self):
        return {"cgroup": {"enabled": False}}

    def test_master_switch_creates_no_cgroup(self, cgroup_cluster):
        # The job still runs; it just inherits the agent's cgroup instead of
        # getting one of its own.
        probe = _run_probe(
            cgroup_cluster,
            ["--cpus-per-task=1", "--mem=512"],
            "cg-off",
            expect_enforced=False,
        )
        assert not probe.values["CGROUP_PATH"].startswith("/spur/job_"), (
            probe.context()
        )

    def test_disabled_does_not_bound_memory(self, cgroup_cluster):
        # Contrast to the OOM tests: the allocation that gets OOM-killed under a
        # 64 MiB ceiling completes when disabled, since no memory.max is written.
        name = "cg-off-mem"
        marker = f"{cgroup_cluster.remote_dir}/{name}-done.txt"
        script = cgroup_cluster.write_file(
            f"{name}.sh",
            "#!/bin/bash\n"
            # A bounded 128 MiB — over a 64 MiB ceiling, but far under node RAM,
            # so nothing here can OOM the node when the ceiling is absent.
            "a=$(head -c $((128*1024*1024)) /dev/zero | tr '\\0' x)\n"
            f'echo "ALLOC_DONE len=${{#a}}" > "{marker}"\n',
        )
        sb = cgroup_cluster.sbatch(
            ["-J", name, "-N", "1", "-w", cgroup_cluster.node_names[0], "-t", "2",
             "--cpus-per-task=1", "--mem=64", script]
        )
        job_id = parse_job_id(sb)
        assert job_id is not None, f"sbatch failed: {sb}"
        state = wait_job(cgroup_cluster, job_id, timeout=120)

        assert state == "CD", (
            f"with cgroups disabled a 128 MiB allocation under --mem=64 must not "
            f"be OOM-killed, got {state}\n{cgroup_cluster.debug_job(job_id)}"
        )
        assert "ALLOC_DONE" in cgroup_cluster.nodes[0].read_file(marker), (
            "the job must have allocated past the (unenforced) ceiling"
        )


class TestCgroupLeftover:
    """A rootful job (or a crash) can leave a root-owned
    ``/sys/fs/cgroup/spur/job_<id>_<attempt>`` that a non-root agent reusing that id cannot
    remove. The launch must degrade to no isolation and complete, not fail on
    every retry — the exact regression that once left jobs stuck PENDING.
    """

    def test_a_leftover_root_owned_cgroup_does_not_wedge_the_launch(self, cluster):
        node = cluster.nodes[0]
        if "OK" not in node.exec_allow_fail("sudo -n true 2>/dev/null && echo OK"):
            pytest.skip("planting a root-owned cgroup needs passwordless sudo on node 0")
        if "cgroup2fs" not in node.exec_allow_fail("stat -fc %T /sys/fs/cgroup").strip():
            pytest.skip("node 0 is not cgroup v2")
        if cluster.spurd_agent_user(0) == "root":
            pytest.skip("this regression is specific to a non-root agent")

        # A root-owned cgroup root is what lets the non-root agent reach the claim
        # path at all: it can traverse the directory but not create inside it.
        node.exec("sudo -n mkdir -p /sys/fs/cgroup/spur")

        # Job ids are handed out in order, so plant leftovers across the window the
        # next submission will land in, then submit into it.
        prime_script = cluster.write_file("cg-leftover-prime.sh", "#!/bin/bash\ntrue\n")
        prime = parse_job_id(
            cluster.sbatch(
                ["-J", "cg-leftover-prime", "-N", "1", "-w", cluster.node_names[0],
                 prime_script]
            )
        )
        assert prime is not None
        planted = list(range(prime + 1, prime + 6))
        for jid in planted:
            node.exec_allow_fail(f"sudo -n mkdir -p /sys/fs/cgroup/spur/job_{jid}_1")

        try:
            script = cluster.write_file(
                "cg-leftover.sh", "#!/bin/bash\necho LEFTOVER_RAN\n"
            )
            out_path = f"{cluster.remote_dir}/cg-leftover.out"
            job_id = parse_job_id(
                cluster.sbatch(
                    ["-J", "cg-leftover", "-N", "1", "-w", cluster.node_names[0],
                     "-o", out_path, script]
                )
            )
            assert job_id is not None
            assert job_id in planted, (
                f"job {job_id} landed outside the planted window {planted}; "
                f"widen it"
            )

            # Before the fix this stuck PENDING forever ("outlived its cleanup"),
            # so wait_job would raise TimeoutError.
            state = wait_job(cluster, job_id, timeout=90)
            assert state == "CD", (
                f"a job whose leftover cgroup a non-root agent cannot remove must "
                f"degrade and complete, got {state}\n{cluster.debug_job(job_id)}"
            )
            assert "LEFTOVER_RAN" in cluster.read_output_on_any_node(out_path), (
                "the degraded job must still run its payload"
            )
            # Prove the agent actually hit the unremovable-leftover (EEXIST) path
            # and degraded, so a run that never reached it cannot pass silently.
            log = cluster.spurd_log(0)
            assert "File exists" in log and "cgroup unavailable" in log, (
                f"expected a degrade-on-EEXIST for the planted leftover\n"
                f"spurd log tail:\n{log[-2000:]}"
            )
            assert "outlived its cleanup" not in log, (
                "the old fatal path must be gone"
            )
        finally:
            for jid in planted:
                node.exec_allow_fail(f"sudo -n rmdir /sys/fs/cgroup/spur/job_{jid}_1 2>/dev/null")
            node.exec_allow_fail("sudo -n rmdir /sys/fs/cgroup/spur 2>/dev/null")


class TestCgroupRequired:
    @pytest.fixture
    def cluster_config_overrides(self):
        return {"cgroup": {"required": True}}

    def test_required_refuses_to_launch_when_enforcement_is_unavailable(self, cluster):
        # Unprivileged `cluster`, not cgroup_cluster, on purpose: an agent that cannot
        # enforce must refuse the job rather than run it unconstrained.
        script = cluster.write_file("cg-required.sh", "#!/bin/bash\necho RAN\n")
        out_path = f"{cluster.remote_dir}/cg-required.out"
        sb = cluster.sbatch(
            ["-J", "cg-required", "-N", "1", "--cpus-per-task=1", "--mem=256",
             "-o", out_path, script]
        )
        job_id = parse_job_id(sb)
        assert job_id is not None, f"sbatch failed: {sb}"

        # A refused launch is retried, so the job stays pending: poll the log, not state.
        # Match the refusal itself — the startup banner also says `required`.
        refusal = "required but"
        deadline = time.time() + 45
        while time.time() < deadline:
            if refusal in _spurd_logs(cluster):
                break
            time.sleep(2)

        assert "RAN" not in cluster.read_output_on_any_node(out_path), (
            f"job body executed despite [cgroup] required\n{cluster.debug_job(job_id)}"
        )
        assert refusal in _spurd_logs(cluster), (
            "agent must log why it refused the launch\n"
            f"{_spurd_logs(cluster)[-2000:]}"
        )
        cluster.scancel(str(job_id))


class TestCgroupConfigValidation:
    def test_controller_refuses_a_zero_ram_percent(self, unstarted_cluster):
        # Zero silently floors every job's memory ceiling to min_ram_mb, so the
        # daemon refuses to start rather than run a whole cluster that way.
        cluster = unstarted_cluster
        conf = cluster.write_file(
            "bad-cgroup.conf",
            'cluster_name = "cgroup-validation"\n'
            "[cgroup]\n"
            "allowed_ram_percent = 0\n",
            executable=False,
        )
        out = cluster.nodes[0].exec_allow_fail(
            f"timeout 20 '{cluster.bin_dir}/spurctld' -f '{conf}' "
            f"--listen '[::]:16899' --state-dir '{cluster.remote_dir}/bad-state' "
            f"2>&1; echo EXIT=$?"
        )
        assert "EXIT=0" not in out, f"spurctld must not start with an invalid config:\n{out}"
        assert "allowed_ram_percent" in out, (
            f"startup failure must name the offending field:\n{out}"
        )


class TestCgroupContainerStep:
    """A standalone ``srun --container-image`` step takes the fork + pivot_root
    container path (``run_containerized_step``), distinct from a batch container,
    a plain step, and an nsenter step. It must still join the job cgroup.
    """

    def test_container_step_runs_inside_the_job_cgroup(self, cgroup_cluster, tmp_path):
        cluster = cgroup_cluster
        cluster.container_preflight()
        image = cluster.build_container_image(tmp_path)

        # A *standalone* srun --container-image is what takes the fork + pivot_root
        # container path; its own allocation is non-container. (The same flag on a
        # step inside an allocation stays a plain host step.)
        code, out = cluster.srun_with_exit(
            ["-w", cluster.node_names[0], "--mem=256", f"--container-image={image}",
             "sh", "-c",
             "cat /proc/self/cgroup; "
             "test -e /etc/os-release && echo IS_HOST || echo IS_CONTAINER"]
        )

        # /etc/os-release exists on the host but not in the minimal image, so this
        # proves the step ran inside the pivoted rootfs (arm 2), not on the host —
        # otherwise it would join the cgroup regardless and prove nothing.
        assert "IS_CONTAINER" in out, (
            f"the step must run inside the container rootfs (exit {code})\noutput:\n{out}"
        )
        # The container's /proc still reports the host cgroup path (no cgroup-ns
        # remap), so a joined step reads /spur/job_<id>_<attempt>; an uncontained
        # one would read spurd's own cgroup instead.
        assert "0::/spur/job_" in out, (
            f"a container step must join the job cgroup (exit {code})\noutput:\n{out}"
        )


def _hold_running_job(cluster, name: str) -> int:
    """Submit a long sleep pinned to node 0 and return its id once RUNNING, so
    exec/attach/step have a live job cgroup to enter."""
    script = cluster.write_file(f"{name}.sh", "#!/bin/bash\nsleep 300\n")
    job_id = parse_job_id(
        cluster.sbatch(
            ["-J", name, "-N", "1", "-w", cluster.node_names[0], "-t", "5",
             "--cpus-per-task=1", "--mem=256", script]
        )
    )
    assert job_id is not None, "sbatch failed to return a job id"
    wait_job_state(cluster, job_id, "R", timeout=90)
    return job_id


class TestCgroupEntryPathMembership:
    """A plain srun step, ``spur exec``, and a ``--pty`` attach each enter a
    running job's cgroup via the same pre-exec join. The device-filter tests that
    cover them use ``gpu_cluster`` and skip on a GPU-less node, so verify the
    membership directly through ``/proc/self/cgroup`` instead.
    """

    def test_a_step_runs_inside_the_job_cgroup(self, cgroup_cluster):
        cluster = cgroup_cluster
        job_id = _hold_running_job(cluster, "cg-step-member")
        try:
            code, out = cluster.srun_in_allocation(job_id, ["cat", "/proc/self/cgroup"])
        finally:
            cluster.scancel(str(job_id))
        assert f"0::/spur/job_{job_id}_1" in out, (
            f"an srun step must join the job cgroup (exit {code})\noutput:\n{out}"
        )

    def test_exec_runs_inside_the_job_cgroup(self, cgroup_cluster):
        cluster = cgroup_cluster
        job_id = _hold_running_job(cluster, "cg-exec-member")
        try:
            out = cluster.cli_allow_fail(
                ["spur", "exec", str(job_id), "cat", "/proc/self/cgroup"]
            )
        finally:
            cluster.scancel(str(job_id))
        assert f"0::/spur/job_{job_id}_1" in out, (
            f"spur exec must join the job cgroup\noutput:\n{out}"
        )

    def test_pty_attach_runs_inside_the_job_cgroup(self, cgroup_cluster):
        cluster = cgroup_cluster
        job_id = _hold_running_job(cluster, "cg-pty-member")
        try:
            # `--pty` streams over a PTY, so the cgroup line may carry a trailing
            # CR; match it as a substring rather than a whole line.
            code, out = cluster.srun_with_exit(
                ["--jobid", str(job_id), "--overlap", "--pty", "cat", "/proc/self/cgroup"]
            )
        finally:
            cluster.scancel(str(job_id))
        assert f"0::/spur/job_{job_id}_1" in out, (
            f"a --pty attach must join the job cgroup (exit {code})\noutput:\n{out}"
        )
