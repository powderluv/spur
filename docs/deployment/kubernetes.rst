Kubernetes Deployment
=====================

Deploy Spur on an existing Kubernetes cluster. The controller runs as a StatefulSet with Raft consensus, and compute nodes are managed by the ``spur-k8s-operator``.

Prerequisites
-------------

- Kubernetes cluster with ``kubectl`` configured

Build and load container images:

.. code-block:: bash

   # Build
   docker build --target runtime -t spur:<tag> .

   # Load onto each node (if not using a registry)
   docker save spur:<tag> -o spur.tar
   # SCP to each node, then:
   sudo ctr -n k8s.io images import spur.tar

Components
----------

- **spurctld** — Controller. Runs as a StatefulSet with Raft consensus for high availability. Handles accounting (backed by PostgreSQL via ``accounting.database_url``) and serves the Slurm-compatible REST API on port 6820.
- **spurd** — Node agent. Runs on each compute node (DaemonSet or Deployment).
- **spur-k8s-operator** — Watches ``SpurJob`` custom resources, registers the Kubernetes nodes with the controller, and runs each job as a Pod.

Example manifests for production-style deployment live in ``examples/k8s/``.

.. _k8s-two-topologies:

Choose one execution backend
----------------------------

``spurd.yaml`` and ``operator.yaml`` are **alternatives, not a pair**. Both register the
Kubernetes nodes with the controller, so applying both makes two agents claim the same
nodes. Pick one:

.. list-table::
   :header-rows: 1

   * - Topology
     - Apply
     - What runs a job
   * - **Pod mode** (recommended)
     - ``operator.yaml``
     - The operator creates a Pod for each allocated node.
   * - **Agent mode**
     - ``spurd.yaml``
     - spurd runs the job as a process inside its own Pod, as on a bare node.

In Pod mode a job must be a ``SpurJob`` custom resource. The operator finds the job by
the label ``spur.amd.com/job-id``, which only a ``SpurJob`` carries, so the
Slurm-compatible CLI (``sbatch``, ``spur submit``) cannot launch work: the job is
accepted, no Pod is made, and it stays in the queue. Use ``kubectl apply`` of a
``SpurJob``.

Deploy
------

.. note::

   Before applying, review the manifests and update namespaces, image names/tags, resource limits, and storage classes to match your environment. In agent mode, ensure the ``--controller`` argument in ``spurd.yaml`` includes the ``http://`` scheme (e.g. ``http://spurctld.spur.svc.cluster.local:6817``).

.. warning::

   **Set the final replica count of** ``spurctld`` **before the first apply.** It cannot
   be raised later. Each new StatefulSet ordinal gets an empty volume from
   ``volumeClaimTemplates``, and a rolling update starts at the highest ordinal, so new
   empty replicas would try to form a second Raft cluster before the replica that holds
   the data is even restarted. spurctld detects this and refuses to start, which leaves
   the new replicas in ``CrashLoopBackOff``. To change the count, take the cluster down
   and build it again.

Apply manifests in order (Pod mode):

.. code-block:: bash

   kubectl apply -f examples/k8s/namespace.yaml
   kubectl apply -f examples/k8s/configmap.yaml
   kubectl apply -f examples/k8s/rbac.yaml
   kubectl apply -f examples/k8s/spurjob-crd.yaml
   kubectl apply -f examples/k8s/spurctld.yaml
   kubectl apply -f examples/k8s/operator.yaml
   kubectl apply -f examples/k8s/pdb.yaml

For agent mode, apply ``spurd.yaml`` in place of ``operator.yaml``.

Configuration
-------------

The ConfigMap (``examples/k8s/configmap.yaml``) embeds ``spur.conf``:

.. code-block:: toml

   cluster_name = "spur-k8s"

   [controller]
   peers = [
     "spurctld-0.spurctld.spur.svc.cluster.local:6821",
     "spurctld-1.spurctld.spur.svc.cluster.local:6821",
     "spurctld-2.spurctld.spur.svc.cluster.local:6821",
   ]

   [scheduler]
   interval_secs = 2
   plugin = "backfill"

   [[partitions]]
   name = "default"
   state = "UP"
   default = true

Raft peers use StatefulSet DNS names. The node ID is auto-derived from each pod's
position in ``peers`` by matching the pod hostname (e.g. ``spurctld-0``) against
each entry's host part, so ``controller.node_id`` never needs to be set. Each
pod's hostname must correspond to its own ``peers`` entry.

Resolution precedence is: explicit ``controller.node_id`` -> position in
``peers`` -> hostname ordinal. The resolved id must fall within
``1..=len(peers)``; if a pod's hostname matches no entry (or matches more than
one), the controller fails fast at startup rather than joining with a wrong ID.

Adjust partition definitions to match your cluster hardware. Once the controller
is running, ``scontrol reconfigure`` applies many sections live, while others
need a controller or agent restart — see
:ref:`the configuration reference <reload-scope>` for the per-field breakdown.
``reconfigure`` runs on the Raft leader only — followers keep their startup
config until restarted, at which point they re-read this same ConfigMap and
converge. To roll all controllers onto an updated ConfigMap, restart the
StatefulSet pods.

Submitting Jobs
---------------

Jobs are submitted as ``SpurJob`` custom resources:

.. code-block:: yaml

   apiVersion: spur.amd.com/v1alpha1
   kind: SpurJob
   metadata:
     name: training-run
   spec:
     script: |
       #!/bin/bash
       #SBATCH --job-name=train
       #SBATCH -N 2
       #SBATCH --gres=gpu:8
       torchrun --nnodes=2 train.py

Apply with ``kubectl``:

.. code-block:: bash

   kubectl apply -f job.yaml

The operator watches SpurJob resources, submits them to the controller, and updates status fields as the job progresses.

Authenticating the operator agent surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The operator serves a virtual-agent gRPC surface on ``--listen`` (port 6818) that carries a
cluster-wide pod-create privilege, so reaching it must not be enough to ask the operator to run
work. Authentication mirrors the cluster ``[auth] mode``:

- ``--auth-mode disabled`` does not authenticate callers at all; treat the port as an
  administrative boundary if you use this.
- ``--auth-mode permissive`` (default) verifies a credential when one is presented and otherwise
  logs and allows — the migration default.
- ``--auth-mode required`` rejects every call that carries no credential. It refuses to start
  without a key.
- ``--jwt-key`` / ``SPUR_JWT_KEY`` is the cluster ``[auth] jwt_key`` the operator verifies
  credentials against; source it from a Secret. In ``permissive`` mode with no key, a controller
  that *does* present a credential is rejected (there is no key to verify it), so set the key before
  controllers start sending one.

See the commented ``--auth-mode`` / ``SPUR_JWT_KEY`` lines in ``examples/k8s/operator.yaml``.

Enforcing account quotas
~~~~~~~~~~~~~~~~~~~~~~~~

``--enable-quota`` turns on a reconciler that projects each SPUR account's ``GrpTRES`` allocation
into a per-account ``ResourceQuota``/``LimitRange``. An account with no ``GrpTRES`` set gets a
**closed** quota (``pods: 0``) rather than an uncapped one, so pods submitted under it are rejected
until an allocation is granted:

.. code-block:: bash

   sacctmgr modify account name=myaccount set grptres=cpu=16,mem=32768,gres/gpu=8

Verify
------

.. code-block:: bash

   # All pods running
   kubectl get pods -n spur

   # Controller logs (check Raft leader election)
   kubectl logs statefulset/spurctld -n spur

   # Node registration
   kubectl exec -n spur spurctld-0 -- spur nodes
