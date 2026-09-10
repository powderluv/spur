# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jobs are supervised by spurstepd, so restarting spurd must not kill them."""

import time

from cluster import parse_job_id, job_state, wait_job, wait_job_state


def _supervisors(cluster, node_index: int = 0) -> int:
    return len(_supervisor_pids(cluster, node_index))


def _supervisor_pids(cluster, node_index: int = 0) -> set[str]:
    node = cluster.nodes[node_index]
    return set(node.exec_allow_fail("pgrep -x spurstepd || true").split())


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
        # Identity, not count: a supervisor killed with the agent and respawned
        # afterwards would satisfy any "still one running" check.
        before = _supervisor_pids(cluster)
        assert before, "a running job must have a supervisor"

        cluster.restart_agent(0)

        after = _supervisor_pids(cluster)
        assert before <= after, (
            "the supervisor must outlive the agent that spawned it: "
            f"{sorted(before)} before the restart, {sorted(after)} after"
        )
        assert job_state(cluster.squeue_all(), job_id) == "R", (
            "the job must still be running after its agent restarted"
        )

        assert wait_job(cluster, job_id, timeout=180) == "CD"
        content = cluster.nodes[0].exec(f"cat '{out_path}'")
        assert "SURVIVED" in content
        # A relaunched job would replay its output from the top.
        assert content.count("tick 1\n") == 1, (
            f"the job must not have been restarted:\n{content}"
        )

    # The session is the agent's only record of the job, so a completed one
    # must not be left behind to be rediscovered on the next restart.
    def test_a_completed_jobs_session_is_pruned(self, cluster):
        # Long enough to observe the session while it exists: a job that has
        # already exited cannot distinguish "pruned" from "never created".
        script = cluster.write_file(
            "stepd-prune.sh",
            "#!/bin/bash\nsleep 20\n",
            all_nodes=True,
        )
        job_id = parse_job_id(
            cluster.sbatch(["-J", "stepd-prune", "-w", cluster.node_names[0], script])
        )
        assert job_id is not None
        wait_job_state(cluster, job_id, "R")

        prefix = f"{job_id}."
        assert [name for name in _sessions(cluster) if name.startswith(prefix)], (
            f"expected a runtime session for job {job_id} while it runs, "
            f"saw {_sessions(cluster)} under {cluster.state_dir}/runtime"
        )

        assert wait_job(cluster, job_id, timeout=120) == "CD"

        deadline = time.time() + 60
        while time.time() < deadline:
            if not [name for name in _sessions(cluster) if name.startswith(prefix)]:
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
