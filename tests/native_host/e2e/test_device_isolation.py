# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for kernel-enforced device isolation (cgroup-v2 BPF device filter).

spurd attaches a default-deny ``BPF_PROG_TYPE_CGROUP_DEVICE`` program to each
job's cgroup, so unlike the ``*_VISIBLE_DEVICES`` deny in ``test_gpu_deny.py``
no re-exported variable can undo it. ``/dev/kfd`` is the probe because it
reaches a job's allow-list only through a real allocation and lives outside the
``/dev/dri`` the namespace wrapper swaps for a tmpfs, so a denial there can
only be the filter. Loading it needs root, hence ``@pytest.mark.rootful``.
"""

from typing import NamedTuple

import pytest

from cluster import parse_job_id, wait_job, wait_job_state

KFD = "/dev/kfd"

# EPERM is the filter denying the open; EACCES is ordinary file permissions, which
# would mask it — hence the _require_unfiltered_access skip.
_PROBE_HELPER = """\
probe_open() {
  local p="$1" err
  if err=$( { : < "$p"; } 2>&1 ); then
    echo "$p=OPEN"
  else
    case "$err" in
      *"Operation not permitted"*) echo "$p=EPERM" ;;
      *"Permission denied"*)       echo "$p=EACCES" ;;
      *"No such file or directory"*) echo "$p=ENOENT" ;;
      *) echo "$p=OTHER[$err]" ;;
    esac
  fi
}
"""


def _probe_script(body: str) -> str:
    return f"#!/bin/bash\n{_PROBE_HELPER}{body}echo DEVICE_PROBE_OK\n"


class _Probe(NamedTuple):
    output: str
    job_id: int


def _run_probe(cluster, name: str, body: str, sbatch_args: list[str]) -> _Probe:
    """Submit a probe pinned to node 0: the allow-list is per node, so an unpinned
    job could land somewhere the assertion was not written for.
    """
    script = cluster.write_file(f"{name}.sh", _probe_script(body))
    out_path = f"{cluster.remote_dir}/{name}.out"
    sb = cluster.sbatch(
        ["-J", name, "-N", "1", "-w", cluster.node_names[0], "-o", out_path]
        + sbatch_args
        + [script]
    )
    job_id = parse_job_id(sb)
    assert job_id is not None, f"sbatch failed: {sb}"

    wait_job(cluster, job_id, timeout=120)
    content = cluster.wait_output(out_path, "DEVICE_PROBE_OK", timeout=120)
    assert "DEVICE_PROBE_OK" in content, (
        f"probe did not run to completion\n{cluster.debug_job(job_id)}\n"
        f"output:\n{content}"
    )
    return _Probe(content, job_id)


def _hold_job(cluster, name: str, sbatch_args: list[str]) -> int:
    """Submit a long-lived job pinned to node 0 and return its id once RUNNING.

    Steps and ``spur exec`` need a job that is still running to enter, which the
    run-to-completion shape of :func:`_run_probe` cannot give them. Pinned for
    the same reason: the allow-list is per node.
    """
    hold = cluster.write_file(f"{name}-hold.sh", "#!/bin/bash\nsleep 300\n")
    sb = cluster.sbatch(
        ["-J", name, "-N", "1", "-w", cluster.node_names[0]] + sbatch_args + [hold]
    )
    job_id = parse_job_id(sb)
    assert job_id is not None, f"sbatch failed: {sb}"

    wait_job_state(cluster, job_id, "R", timeout=120)
    return job_id


def _require_rootful(cluster) -> None:
    user = cluster.spurd_agent_user(0)
    assert user == "root", (
        f"the device filter needs a rootful agent to load and attach the BPF "
        f"program, got user {user!r}"
    )


def _require_unfiltered_access(cluster) -> None:
    """Skip unless the test user can already open /dev/kfd: if file permissions
    alone deny it, every deny assertion below passes for the wrong reason.
    """
    node = cluster.nodes[0]
    if not node.exec_allow_fail(f"ls {KFD} 2>/dev/null").strip():
        pytest.skip(f"node 0 has no {KFD} (not an AMD GPU node)")
    probe = node.exec_allow_fail(
        f'if : < {KFD} 2>/dev/null; then echo OPEN; else echo DENIED; fi'
    ).strip()
    if "OPEN" not in probe:
        pytest.skip(
            f"{KFD} is not openable by the test user outside a job, so file "
            f"permissions would mask the device filter"
        )


@pytest.mark.rootful
class TestDeviceIsolation:
    def test_zero_gpu_job_is_denied_the_gpu_control_node(self, gpu_cluster):
        # A job allocated no GPU shares the node with jobs that were, and must not
        # be able to reach the hardware.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)

        content = _run_probe(
            cluster,
            "dev-iso-zero",
            f"probe_open {KFD}\n"
            'head -c 8 /dev/urandom > /dev/null && echo urandom=OPEN\n'
            'echo x > /dev/null && echo null=OPEN\n',
            [],
        ).output

        assert f"{KFD}=EPERM" in content, (
            f"a zero-GPU job must be denied {KFD} by the kernel\noutput:\n{content}"
        )
        # Without these the test would also pass against a filter that denies
        # everything, which would break every job on the node.
        assert "urandom=OPEN" in content, (
            f"base pseudo-devices must stay reachable\noutput:\n{content}"
        )
        assert "null=OPEN" in content, (
            f"base pseudo-devices must stay reachable\noutput:\n{content}"
        )

    def test_allocated_gpu_job_can_open_the_control_node(self, gpu_cluster):
        # The other half of the property: isolation must not cost a job the
        # hardware it was actually given.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)

        content = _run_probe(
            cluster, "dev-iso-alloc", f"probe_open {KFD}\n", ["--gres=gpu:1"]
        ).output

        assert f"{KFD}=OPEN" in content, (
            f"a job allocated a GPU must be able to open {KFD}\noutput:\n{content}"
        )

    def test_partial_gpu_allocation_hides_the_other_gpus(self, gpu_cluster):
        # The rootful tmpfs wrapper stages only the allocated render node; the device
        # filter is the backstop, so the job sees and reaches just its own GPU.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)
        if cluster.node_gpu_count(cluster.node_names[0]) < 2:
            pytest.skip("need >= 2 GPUs on node 0 to prove the siblings are hidden")

        content = _run_probe(
            cluster,
            "dev-iso-partial",
            "n=0\n"
            'for d in /dev/dri/renderD*; do [ -e "$d" ] || continue; '
            'n=$((n+1)); probe_open "$d"; done\n'
            'echo "RENDER_COUNT=$n"\n'
            f"probe_open {KFD}\n",
            ["--gres=gpu:1"],
        ).output

        assert "RENDER_COUNT=1" in content, (
            f"a gpu:1 job on a multi-GPU node must see exactly its one render "
            f"node, not the siblings\noutput:\n{content}"
        )
        assert not any("renderD" in ln and "EPERM" in ln for ln in content.splitlines()), (
            f"the one render node the job was given must be usable, not denied\n"
            f"output:\n{content}"
        )
        assert f"{KFD}=OPEN" in content, (
            f"the allocated job must still reach the GPU control node\n"
            f"output:\n{content}"
        )

    def test_visible_devices_override_does_not_restore_access(self, gpu_cluster):
        # Re-exporting the selector is how a job defeats an advisory sentinel; it
        # must not move a kernel filter.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)

        content = _run_probe(
            cluster,
            "dev-iso-reexport",
            "export ROCR_VISIBLE_DEVICES=0\n"
            "export HIP_VISIBLE_DEVICES=0\n"
            "export CUDA_VISIBLE_DEVICES=0\n"
            f"probe_open {KFD}\n",
            [],
        ).output

        assert f"{KFD}=EPERM" in content, (
            f"re-exporting the GPU selectors must not restore access to {KFD}\n"
            f"output:\n{content}"
        )


@pytest.mark.rootful
class TestStepAndExecDeviceIsolation:
    """The filter is attached to the job's cgroup, so it only reaches a process
    that is in it. ``srun`` steps, ``spur exec``, and interactive ``--pty``
    attaches are not batch payloads and have to join it themselves; each of these
    opened ``/dev/kfd`` freely from inside a zero-GPU job before that join existed.
    """

    def test_step_in_a_zero_gpu_job_is_denied_the_gpu_control_node(self, gpu_cluster):
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)

        job_id = _hold_job(cluster, "dev-iso-step-zero", [])
        probe = cluster.write_file(
            "dev-iso-step-zero-probe.sh", _probe_script(f"probe_open {KFD}\n")
        )
        try:
            code, out = cluster.srun_in_allocation(job_id, [probe])
        finally:
            cluster.scancel(str(job_id))

        assert "DEVICE_PROBE_OK" in out, (
            f"the step did not run to completion (exit {code})\n"
            f"{cluster.debug_job(job_id)}\noutput:\n{out}"
        )
        assert f"{KFD}=EPERM" in out, (
            f"a step in a zero-GPU job must be denied {KFD} by the kernel\n"
            f"output:\n{out}"
        )

    def test_step_in_an_allocated_gpu_job_can_open_the_control_node(self, gpu_cluster):
        # Isolation must not cost a step the hardware its job was granted.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)

        job_id = _hold_job(cluster, "dev-iso-step-alloc", ["--gres=gpu:1"])
        probe = cluster.write_file(
            "dev-iso-step-alloc-probe.sh", _probe_script(f"probe_open {KFD}\n")
        )
        try:
            code, out = cluster.srun_in_allocation(job_id, [probe])
        finally:
            cluster.scancel(str(job_id))

        assert f"{KFD}=OPEN" in out, (
            f"a step in a job allocated a GPU must be able to open {KFD} "
            f"(exit {code})\noutput:\n{out}"
        )

    def test_step_in_a_zero_gpu_interactive_allocation_is_denied_the_control_node(
        self, gpu_cluster
    ):
        # An interactive allocation launches no payload, so its cgroup exists
        # only because registration created one — nothing later would.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)

        probe = cluster.write_file(
            "dev-iso-salloc-probe.sh", _probe_script(f"probe_open {KFD}\n")
        )
        out_path = f"{cluster.remote_dir}/dev-iso-salloc.out"
        code, out = cluster.salloc_run(
            f"'{cluster.bin_dir}/srun' '{probe}' > '{out_path}' 2>&1",
            salloc_args=["-N", "1", "-w", cluster.node_names[0], "-t", "0:05"],
        )
        assert code == 0, f"salloc failed (exit {code}):\n{out}"

        content = cluster.nodes[0].read_file(out_path)
        assert "DEVICE_PROBE_OK" in content, (
            f"the step did not run to completion\noutput:\n{content}\nsalloc:\n{out}"
        )
        assert f"{KFD}=EPERM" in content, (
            f"a step in a zero-GPU interactive allocation must be denied {KFD}\n"
            f"output:\n{content}"
        )

    def test_exec_into_a_zero_gpu_job_is_denied_the_gpu_control_node(self, gpu_cluster):
        # `spur exec` enters the job's namespaces, and namespaces are not
        # cgroups: entry alone would leave it holding spurd's own cgroup.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)

        job_id = _hold_job(cluster, "dev-iso-exec-zero", [])
        probe = cluster.write_file(
            "dev-iso-exec-zero-probe.sh", _probe_script(f"probe_open {KFD}\n")
        )
        try:
            out = cluster.cli_allow_fail(["spur", "exec", str(job_id), "bash", probe])
        finally:
            cluster.scancel(str(job_id))

        assert "DEVICE_PROBE_OK" in out, (
            f"spur exec did not run the probe\n{cluster.debug_job(job_id)}\n"
            f"output:\n{out}"
        )
        assert f"{KFD}=EPERM" in out, (
            f"spur exec into a zero-GPU job must be denied {KFD} by the kernel\n"
            f"output:\n{out}"
        )

    def test_pty_attach_into_a_zero_gpu_job_is_denied_the_gpu_control_node(
        self, gpu_cluster
    ):
        # `srun --pty` opens an interactive PTY step via spawn_pty_in_job, a path
        # of its own — not a batch payload, a run_command step, or spur exec.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)

        job_id = _hold_job(cluster, "dev-iso-pty-zero", [])
        probe = cluster.write_file(
            "dev-iso-pty-zero-probe.sh", _probe_script(f"probe_open {KFD}\n")
        )
        try:
            code, out = cluster.srun_with_exit(
                ["--jobid", str(job_id), "--overlap", "--pty", "bash", probe]
            )
        finally:
            cluster.scancel(str(job_id))

        # !r so the PTY's CRLFs and any control bytes are legible on failure.
        assert "DEVICE_PROBE_OK" in out, (
            f"the pty attach did not run the probe to completion (exit {code})\n"
            f"{cluster.debug_job(job_id)}\noutput:\n{out!r}"
        )
        assert f"{KFD}=EPERM" in out, (
            f"a pty attach into a zero-GPU job must be denied {KFD} by the kernel\n"
            f"output:\n{out!r}"
        )

    def test_a_step_lands_in_the_job_cgroup(self, gpu_cluster):
        # Membership is the mechanism every deny above rests on, so assert it
        # directly. Read from inside the step: a read from the test would race
        # the step's exit, which drops the pid from cgroup.procs.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)

        job_id = _hold_job(cluster, "dev-iso-step-cgroup", [])
        procs = f"/sys/fs/cgroup/spur/job_{job_id}_1/cgroup.procs"
        probe = cluster.write_file(
            "dev-iso-step-cgroup-probe.sh",
            f"""#!/bin/bash
if grep -qx "$$" {procs} 2>/dev/null; then
  echo STEP_CGROUP=JOINED
else
  echo STEP_CGROUP=OUTSIDE
fi
echo "STEP_PID=$$ PROCS=$(tr '\\n' ' ' < {procs} 2>/dev/null)"
""",
        )
        try:
            code, out = cluster.srun_in_allocation(job_id, [probe])
        finally:
            cluster.scancel(str(job_id))

        assert "STEP_CGROUP=JOINED" in out, (
            f"the step's pid must appear in {procs} (exit {code})\noutput:\n{out}"
        )


@pytest.mark.rootful
class TestDeviceIsolationConfig:
    def test_constrain_devices_false_leaves_the_node_reachable(self, gpu_cluster):
        # Proves the deny in the tests above is the filter and not some other
        # layer: the only thing that changes here is the switch.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)

        cluster.stop()
        cluster.start({"cgroup": {"constrain_devices": False}}, agent_as_root=True)

        content = _run_probe(
            cluster, "dev-iso-off", f"probe_open {KFD}\n", []
        ).output

        assert f"{KFD}=OPEN" in content, (
            f"with constrain_devices = false a zero-GPU job must still reach "
            f"{KFD}\noutput:\n{content}"
        )

    def test_extra_device_paths_allows_an_unallocated_node(self, gpu_cluster):
        # The operator escape hatch for a device the allocation does not cover
        # and the built-in host-infrastructure list does not name.
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)

        cluster.stop()
        cluster.start({"cgroup": {"extra_device_paths": [KFD]}}, agent_as_root=True)

        content = _run_probe(
            cluster, "dev-iso-extra", f"probe_open {KFD}\n", []
        ).output

        assert f"{KFD}=OPEN" in content, (
            f"{KFD} listed in extra_device_paths must be allowed even for a "
            f"zero-GPU job\noutput:\n{content}"
        )


@pytest.mark.rootful
class TestDeviceFilterLifecycle:
    def test_filter_is_released_when_the_job_cgroup_goes_away(self, gpu_cluster):
        """Nothing detaches the filter: removing the cgroup directory drops the
        kernel's last reference. A leak would accumulate one program per job.
        """
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)

        node = cluster.nodes[0]
        if not node.exec_allow_fail("command -v bpftool").strip():
            pytest.skip("bpftool is not installed on node 0")
        # Without a working non-interactive sudo the count below would always
        # read 0 and the assertion would hold for the wrong reason.
        if "PROBE_OK" not in node.exec_allow_fail(
            "sudo -n bpftool prog show >/dev/null 2>&1 && echo PROBE_OK"
        ):
            pytest.skip("bpftool needs passwordless sudo on node 0 to list programs")

        def loaded() -> int:
            out = node.exec_allow_fail(
                "sudo -n bpftool prog show 2>/dev/null | grep -c cgroup_device"
            ).strip()
            return int(out) if out.isdigit() else -1

        baseline = loaded()
        assert baseline >= 0, "could not read the loaded cgroup_device program count"

        probe = _run_probe(cluster, "dev-iso-life", "true\n", ["--gres=gpu:1"])
        cgroup = f"/sys/fs/cgroup/spur/job_{probe.job_id}_1"

        # Scoped to this job's cgroup rather than every job_* directory, so a
        # job belonging to another test cannot decide this one.
        assert not node.exec_allow_fail(f"ls -d {cgroup} 2>/dev/null").strip(), (
            f"{cgroup} outlived its job, so its filter is still attached"
        )
        after = loaded()
        assert after == baseline, (
            f"cgroup_device programs went from {baseline} to {after}; the filter "
            f"is not being released with the job cgroup"
        )


@pytest.mark.rootful
class TestContainerDeviceIsolation:
    """A native container's process tree lives in the job's cgroup, so the filter
    reaches it too. A zero-GPU container must not reach the GPU — whether the
    control node is simply absent from its ``/dev`` or the filter denies the open.
    """

    def test_zero_gpu_container_cannot_reach_the_gpu(self, gpu_cluster, tmp_path):
        cluster = gpu_cluster
        cluster.gpu_preflight(1)
        _require_rootful(cluster)
        _require_unfiltered_access(cluster)
        cluster.container_preflight()
        image = cluster.build_container_image(tmp_path)

        content = _run_probe(
            cluster,
            "dev-iso-ctr-zero",
            f"probe_open {KFD}\n"
            "probe_open /dev/null\n",
            [f"--container-image={image}"],
        ).output

        assert f"{KFD}=OPEN" not in content, (
            f"a zero-GPU container must not be able to open {KFD}\noutput:\n{content}"
        )
        assert f"{KFD}=EPERM" in content or f"{KFD}=ENOENT" in content, (
            f"{KFD} must be denied by the filter or absent from the container's "
            f"/dev\noutput:\n{content}"
        )
        # Guards against a filter that just denies everything, which would also
        # break a container that has no business losing its base devices.
        assert "/dev/null=OPEN" in content, (
            f"base devices must still work inside the container\noutput:\n{content}"
        )
