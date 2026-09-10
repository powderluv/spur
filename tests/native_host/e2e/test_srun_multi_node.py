# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for standalone and step-mode srun task fan-out."""

import re
import shlex
import time

from cluster import parse_job_id, wait_job, wait_job_state


class TestStandaloneSrun:
    def test_srun_two_nodes_two_tasks(self, multi_node_cluster):
        cluster = multi_node_cluster
        code, out = cluster.srun_with_exit(
            [
                "-N",
                "2",
                "-n",
                "2",
                "bash",
                "-c",
                'echo "host=$(hostname) rank=${SPUR_PROCID}"',
            ]
        )
        assert code == 0, f"srun failed (exit {code}):\n{out}"

        lines = [ln for ln in out.splitlines() if ln.startswith("host=")]
        assert len(lines) == 2, f"expected 2 task lines, got {len(lines)}:\n{out}"

        hosts = {ln.split("host=")[1].split()[0] for ln in lines}
        assert len(hosts) == 2, f"expected 2 distinct hosts, got {hosts}:\n{out}"

        ranks = {ln.split("rank=")[1].strip() for ln in lines}
        assert ranks == {"0", "1"}, f"expected ranks 0 and 1, got {ranks}:\n{out}"

    def test_srun_holds_allocation_until_step_finishes(self, multi_node_cluster):
        cluster = multi_node_cluster
        log = f"{cluster.remote_dir}/srun-hold.log"
        srun_cmd = " ".join(
            [
                f"SPUR_CONTROLLER_ADDR={shlex.quote(cluster.controller_addr)}",
                f"PATH={shlex.quote(cluster.bin_dir)}:$PATH",
                "nohup",
                shlex.quote(f"{cluster.bin_dir}/srun"),
                "-J",
                "srun-hold",
                "-N",
                "2",
                "-n",
                "2",
                "sleep",
                "8",
                ">",
                shlex.quote(log),
                "2>&1",
                "&",
                "echo",
                "$!",
            ]
        )
        pid_out = cluster.nodes[0].exec(srun_cmd).strip()
        assert pid_out.isdigit(), f"expected background srun pid, got: {pid_out!r}"
        time.sleep(2)

        job_ids = cluster.running_job_ids_by_name("srun-hold")
        assert job_ids, (
            "expected running srun-hold job in squeue:\n"
            f"{cluster.squeue(['-n', 'srun-hold', '-t', 'all'])}"
        )
        job_id = job_ids[0]

        wait_job_state(cluster, job_id, "R", timeout=30)
        show = cluster.scontrol("show", "job", str(job_id))
        assert "JobState=RUNNING" in show, f"expected RUNNING job:\n{show}"
        num_nodes_match = re.search(r"NumNodes=(\d+)", show)
        assert num_nodes_match, f"missing NumNodes in scontrol output:\n{show}"
        assert int(num_nodes_match.group(1)) == 2, (
            f"expected 2-node allocation while srun sleep runs:\n{show}"
        )
        nodelist_match = re.search(r"(?<![A-Za-z])NodeList=(\S+)", show)
        assert nodelist_match, f"missing NodeList in scontrol output:\n{show}"
        assert nodelist_match.group(1) != "(null)", (
            f"expected an allocated nodelist while srun sleep runs:\n{show}"
        )
        assert not cluster._cluster_is_ready(), (
            f"expected allocated nodes while srun sleep runs, sinfo:\n{cluster.sinfo()}"
        )

        wait_job(cluster, job_id, timeout=60)
        deadline = time.time() + 60
        while time.time() < deadline:
            if cluster._cluster_is_ready():
                return
            time.sleep(2)
        raise TimeoutError("nodes did not return to idle after srun completed")


class TestSrunInBatch:
    def test_sbatch_srun_hostname_on_allocated_nodes(self, multi_node_cluster):
        cluster = multi_node_cluster
        out_path = f"{cluster.remote_dir}/srun-step.out"
        script = cluster.write_file(
            "srun-in-batch.sh",
            "#!/bin/bash\n"
            "srun -n 2 bash -c 'echo host=$(hostname) rank=${SPUR_PROCID}'\n",
        )
        sb = cluster.sbatch(["-J", "srun-step", "-N", "2", "-n", "2", "-o", out_path, script])
        job_id = parse_job_id(sb)
        assert job_id is not None

        wait_job(cluster, job_id, timeout=90)
        content = cluster.read_output_on_any_node(out_path)
        lines = sorted({ln for ln in content.splitlines() if ln.startswith("host=")})
        assert len(lines) == 2, (
            f"expected 2 step task lines in batch output:\n{content}\n"
            f"{cluster.debug_job(job_id)}"
        )
