# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jobs are supervised by spurstepd, so restarting spurd must not kill them."""

import time

from cluster import parse_job_id, job_state, wait_job, wait_job_state


def _supervisors(cluster, node_index: int = 0) -> int:
    node = cluster.nodes[node_index]
    return int(node.exec("pgrep -x spurstepd | wc -l").strip())


def _sessions(cluster, node_index: int = 0) -> list[str]:
    node = cluster.nodes[node_index]
    listing = node.exec_allow_fail(
        f"ls '{cluster.state_dir}/runtime' 2>/dev/null || true"
    )
    return [name for name in listing.split() if name[:1].isdigit()]


class TestStepdRestartSurvival:
    def test_running_job_survives_an_agent_restart_and_completes(self, cluster):
        out_path = f"{cluster.remote_dir}/survive.out"
        script = cluster.write_file(
            "stepd-survive.sh",
            '#!/bin/bash\nfor i in $(seq 1 30); do echo "tick $i"; sleep 2; done\n'
            "echo SURVIVED\n",
            all_nodes=True,
        )
        node = cluster.node_names[0]
        job_id = parse_job_id(
            cluster.sbatch(
                ["-J", "stepd-survive", "-w", node, "-o", out_path, script]
            )
        )
        assert job_id is not None
        wait_job_state(cluster, job_id, "R")
        assert _supervisors(cluster) >= 1, "a running job must have a supervisor"

        cluster.restart_agent(0)

        assert _supervisors(cluster) >= 1, (
            "the supervisor must outlive the agent that spawned it"
        )
        assert job_state(cluster.squeue_all(), job_id) == "R", (
            "the job must still be running after its agent restarted"
        )

        assert wait_job(cluster, job_id, timeout=180) == "CD"
        assert "SURVIVED" in cluster.nodes[0].exec(f"cat '{out_path}'")

    # The session is the agent's only record of the job, so a completed one
    # must not be left behind to be rediscovered on the next restart.
    def test_a_completed_jobs_session_is_pruned(self, cluster):
        before = set(_sessions(cluster))
        job_id = parse_job_id(
            cluster.sbatch(
                ["-J", "stepd-prune", "-w", cluster.node_names[0], "--wrap", "true"]
            )
        )
        assert job_id is not None
        assert wait_job(cluster, job_id, timeout=120) == "CD"

        deadline = time.time() + 60
        while time.time() < deadline:
            if not set(_sessions(cluster)) - before:
                return
            time.sleep(2)
        assert False, f"session for job {job_id} was never pruned: {_sessions(cluster)}"

    def test_cancelling_a_job_reaps_its_supervisor(self, cluster):
        baseline = _supervisors(cluster)
        script = cluster.write_file(
            "stepd-cancel.sh",
            "#!/bin/bash\nfor i in $(seq 1 120); do sleep 2; done\n",
            all_nodes=True,
        )
        job_id = parse_job_id(
            cluster.sbatch(["-J", "stepd-cancel", "-w", cluster.node_names[0], script])
        )
        assert job_id is not None
        wait_job_state(cluster, job_id, "R")
        assert _supervisors(cluster) > baseline

        cluster.scancel(str(job_id))
        assert wait_job(cluster, job_id, timeout=120) in ("CA", "CD", "GONE")

        deadline = time.time() + 60
        while time.time() < deadline:
            if _supervisors(cluster) <= baseline:
                return
            time.sleep(2)
        assert False, "the cancelled job's supervisor was never reaped"
