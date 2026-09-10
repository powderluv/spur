Spur-Managed Kubernetes (k0s)
=============================

Spur can **provision and own** a Kubernetes cluster across its own nodes using
`k0s <https://k0sproject.io>`_. This is the inverse of :doc:`kubernetes` (running
Spur *inside* an existing cluster): here ``spur k8s up`` builds the cluster and a
spurd-owned systemd unit keeps k0s running on each node.

One command assigns roles, mesh IPs and pod CIDRs, installs a pinned k0s on every
node, brings up the control plane, mints join tokens, joins the workers, and
(optionally) programs a WireGuard-native CNI — all driven by the existing
spurctld/spurd control plane.

.. note::

   This feature is gated on ``[cluster].enabled``. With it off (the default),
   spurd never touches systemd or k0s, and nothing here applies.

Overview
--------

- **spurctld** (on the head node) owns the cluster lifecycle: role/IP/CIDR
  assignment and phase, replicated through Raft so it survives a restart.
- **spurd** (on every node) owns that node's k0s systemd unit: it installs k0s if
  missing, writes the config/join-token, and reconciles the unit. k0s is never a
  Spur job, so it survives a spurd restart (spurd re-adopts it on startup).
- **Roles** are assigned automatically: a single node becomes an all-in-one
  ``controller --single``; with two or more nodes the control-plane node becomes a
  ``controller`` and the rest become ``worker`` s.

For Administrators
------------------

Prerequisites
~~~~~~~~~~~~~~

- A working native-host Spur deployment — ``spurctld`` on the head node and
  ``spurd`` on every node, all registered. See :doc:`native-host`.
- ``spurd`` must run as root (it manages systemd units).
- Outbound HTTPS to ``github.com`` on each node for the k0s download (or
  pre-stage the binary — see `Installing k0s`_).
- For the mesh-native CNI only: a WireGuard mesh (``spur0``) already established
  across the nodes via ``spur net join`` / ``spur net mesh``.

Configure the cluster
~~~~~~~~~~~~~~~~~~~~~~~

Add a ``[cluster]`` section to ``spur.conf`` on every node (spurd reads
``k0s_version`` / ``cni`` from it; spurctld reads the CIDRs and control-plane
choice):

.. code-block:: toml

   [network]
   wg_cidr = "10.44.0.0/16"          # mesh CIDR; node mesh IPs are allocated from here

   [cluster]
   enabled = true
   control_plane_node = "head-node"  # hostname of the k0s control plane (else: first node)
   pod_cidr = "10.42.0.0/16"         # per-node /24s are carved from this
   service_cidr = "10.43.0.0/16"
   cni = "kuberouter"                # "kuberouter" (default) or "calico" (see Networking)
   cni_mtu = 1450                    # Calico MTU; headroom for WireGuard overhead on the mesh
   storage_provisioner = "local-path"  # default StorageClass for PVCs; or "none"
   # local_path_dir = "/mnt/scratch/local-path"  # point local-path at a big disk (default /var/lib)
   k0s_version = "v1.36.2+k0s.0"     # pinned; or "latest"

Installing k0s
~~~~~~~~~~~~~~~

``spur k8s up`` auto-installs the pinned k0s on any node that is missing it, so
usually you do nothing. To pre-stage it (e.g. for an air-gapped or
network-restricted node), run on that node **as root**:

.. code-block:: bash

   sudo spur k8s install-k0s                     # the pinned version -> /usr/local/bin/k0s
   sudo spur k8s install-k0s --version latest    # newest k0s release
   sudo spur k8s install-k0s --version v1.36.2+k0s.0 --path /opt/bin/k0s --force

The binary is downloaded from the official k0s GitHub release and SHA-256
verified before it is installed.

Bring the cluster up
~~~~~~~~~~~~~~~~~~~~~~

From the head node (or any host that can reach ``spurctld``):

.. code-block:: bash

   spur k8s up --controller http://localhost:6817

This is idempotent and asynchronous — spurctld reconciles toward ``Ready``:
control plane first, then workers join with freshly minted tokens. A fresh
cluster typically reaches ``Ready`` in one to two minutes (mostly the k0s
download + control-plane bootstrap). ``spur k8s up`` requires cluster admin
(``root``, or an accounting admin — see `Access control`_).

Scope the cluster to a subset of nodes
''''''''''''''''''''''''''''''''''''''

By default ``spur k8s up`` enrolls every registered node. To build a smaller
cluster, scope it with ``--nodes`` (a hostlist), ``--partition``, and/or
``--selector`` (repeatable ``key=value``, ANDed) — the three are unioned
together and resolved once, at up-time:

.. code-block:: bash

   spur k8s up --nodes "gpu[01-08]"
   spur k8s up --partition batch
   spur k8s up --selector zone=z1 --selector gpu=mi300

A scoped cluster's membership is frozen until you grow or shrink it with
``add-nodes`` / ``remove-nodes`` (see `Adding and removing worker nodes`_).

High-availability control plane
'''''''''''''''''''''''''''''''''

By default the cluster runs a single control-plane node (``control_plane_node``
in ``spur.conf``, or the first node). For HA, request 3 or 5 control planes:

.. code-block:: bash

   spur k8s up --replicas 3                                    # lowest-named 3 nodes
   spur k8s up --control-plane-nodes cp-1,cp-2,cp-3             # explicit set; overrides --replicas

``--control-plane-nodes`` (or a single ``--control-plane-node``) always wins
over ``--replicas``. The first control-plane node is the etcd bootstrap.

.. important::

   Every control-plane node — whether picked automatically, named with
   ``--control-plane-node``, or listed in ``--control-plane-nodes`` — **must be
   part of the cluster's node scope**. If you also pass ``--nodes``,
   ``--partition``, or ``--selector``, make sure the control-plane node(s) are
   included in that selection; otherwise ``spur k8s up`` is rejected with
   ``control-plane node <name> is not a registered node`` (explicit list) or
   ``control-plane node <name> is not among the selected cluster nodes``
   (auto-picked). Leave ``--nodes``/``--partition``/``--selector`` unset to
   scope the cluster to the whole inventory, which trivially satisfies this —
   any registered node is then a valid control-plane choice.

On a multi-control-plane mesh cluster there is no floating VIP — a VRRP address
cannot follow WireGuard cryptokey routing — so ``spur k8s up`` enables k0s
node-local load balancing (``nodeLocalLoadBalancing`` with ``EnvoyProxy``) on
every control plane. This gives each node a local Envoy that round-robins across
all controllers, so konnectivity has a cluster-wide ``:8132`` endpoint instead of
pinning every agent to a single controller. It is on automatically for any
multi-control-plane count (today 3 or 5) and off for a single control plane (no
balancing needed).

.. note::

   k0s does not hot-reload node-local load balancing, and Spur only writes a
   controller's ``k0s.yaml`` while bringing that controller up — an
   already-active control plane is never rewritten in place. A fresh
   ``spur k8s up --replicas 3`` is therefore unaffected (controllers render the
   setting before any worker joins), but an **existing** HA cluster cannot pick
   up the fix by restarting workers alone: reprovision the control plane with
   ``spur k8s down --reset`` followed by ``spur k8s up``.

Check status
~~~~~~~~~~~~~

.. code-block:: bash

   spur k8s status
   # phase: ready
   # control-plane: head-node
   #   head-node   controller  active   enabled=true
   #   node-2      worker      active   enabled=true
   #   ...

``phase`` moves ``down -> provisioning -> ready``. Per-node ``component_state`` is
queried live from each agent.

Networking / CNI
~~~~~~~~~~~~~~~~~

**kuberouter** (default) — the built-in k0s CNI. The control-plane API is advertised
on the node's primary interface and workers join over it. No mesh required. SPUR runs
kube-router with ``overlay-type=full``, so every pod packet travels in the IPIP tunnel
with the node address on the outside. kube-router's own default (``subnet``) sends pod
packets unencapsulated between nodes of the same subnet, and a cloud NIC (OCI VNIC, AWS
ENI, ...) drops a packet whose source is a pod address unless source/destination checking
is disabled on the interface. Full overlay works on any underlay, at the cost of the IPIP
header between same-subnet nodes. The tunnel is IP protocol 4 (IPIP) between nodes, and
kube-router peers over BGP (TCP 179); open both if a host or cloud firewall (OCI security
list, AWS security group, ...) default-denies them.

``pod_cidr`` and ``service_cidr`` from ``[cluster]`` apply to both CNIs.

.. note::

   Clusters built by a SPUR release before 0.12 with ``cni = "kuberouter"`` run on the k0s
   default CIDRs (``10.244.0.0/16`` pods, ``10.96.0.0/12`` services) in ``subnet`` overlay
   mode. The k0s config is rendered once per control plane when it first starts, so those
   nodes keep running as they are. A control plane added or re-joined later receives the
   ``[cluster]`` CIDRs, and k0s requires the same CIDRs on every control plane. To move such
   a cluster to the pinned CIDRs and full overlay, reprovision it (``spur k8s down --reset``
   then ``spur k8s up``), or set ``pod_cidr = "10.244.0.0/16"`` and
   ``service_cidr = "10.96.0.0/12"`` before adding control planes.

**calico** (``cni = "calico"``) — with the WireGuard mesh enabled
(``network.wg_enabled = true``), ``spur k8s up`` generates a k0s config that
advertises the API on the control-plane's **mesh IP** and runs Calico in
``bird`` (BGP, no overlay) mode, setting each worker's kubelet ``--node-ip`` to
its mesh IP so pod traffic routes over the WireGuard mesh. This requires the
``spur0`` mesh to be up first (``spur net join``); membership reconciliation
only maintains the peer set + ``AllowedIPs``, it does not create the tunnel.

The controller continuously reconciles the full-mesh membership to every node
(pruning peers for departed nodes), so a reboot, a WireGuard restart, or a
control-plane failover self-heals.

With the mesh disabled (``network.wg_enabled = false``), Calico instead
advertises the API on each node's real underlay address and runs in ``vxlan``
mode — Calico's own overlay, requiring no mesh. Standard VXLAN (UDP 4789)
between nodes; open that port if a host firewall default-denies it.

.. note::

   Like ``nodeLocalLoadBalancing`` above, this is rendered once per control
   plane and does not hot-reload: toggling ``network.wg_enabled`` after a
   cluster's nodes are already ``active`` does not retroactively switch them
   between ``bird`` and ``vxlan``. Reprovision (``spur k8s down --reset`` then
   ``spur k8s up``) to pick up the change.

Storage
~~~~~~~

k0s bundles no storage, so a plain cluster has no ``StorageClass`` and any
``PersistentVolumeClaim`` stays ``Pending``. By default Spur ships the
`local-path-provisioner <https://github.com/rancher/local-path-provisioner>`_
(``storage_provisioner = "local-path"``) as the cluster's **default**
StorageClass — RWO, node-local — so PVC workloads bind out of the box. The
control-plane agent writes the manifest into the k0s manifest-deployer directory,
which k0s applies automatically (no in-cluster client).

Local-path stores volumes under ``local_path_dir`` (default
``/var/lib/local-path-provisioner``, on the root filesystem). If PVCs will hold
much data — model caches, datasets — point it at a large scratch disk:

.. code-block:: ini

   [cluster]
   local_path_dir = "/mnt/scratch/local-path"

Set ``storage_provisioner = "none"`` to bring your own storage.

Adding and removing worker nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How a worker joins depends on how the cluster was scoped at ``spur k8s up``:

**Whole-inventory cluster** (``spur k8s up`` with no scope flags) — every
registered node is a member. Start ``spurd`` on the new node (registered to the
same controller) and, if using the mesh CNI, join it to the mesh first. On the
next reconcile tick it is assigned a role + mesh IP + pod CIDR and joins
automatically. No further command is needed.

**Scoped cluster** (``spur k8s up --nodes/--partition/--selector``) — membership
is frozen at up-time, so a newly-registered node stays *outside* the cluster until
you add it explicitly. Grow the cluster online, no ``down``/``--reset`` needed:

.. code-block:: bash

   spur k8s add-nodes --nodes gpu[09-12]        # or --partition <p> / --selector k=v
   spur k8s status                              # the new workers converge to active

Added nodes are workers; they are unioned into the member set and enrolled by the
reconcile loop exactly as an in-scope node is. Adding a node already in the
cluster is a no-op.

Remove a worker gracefully — cordon, drain (evict pods, PDB-aware), then stop and
``k0s reset`` the node:

.. code-block:: bash

   spur k8s remove-nodes --nodes gpu12
   spur k8s remove-nodes --nodes gpu12 --drain-timeout 180
   spur k8s remove-nodes --nodes gpu12 --force   # proceed past running jobs / a blocked drain

.. warning::

   ``remove-nodes`` is **destructive**: it runs ``k0s reset`` on the departing
   node, wiping its k0s state (etcd/kine data, pulled images, containerd state,
   certs). Re-adding the same node later re-downloads k0s and re-seeds state
   (~262 MB plus an image re-pull). For a temporary "stop scheduling here", use
   ``spur node drain`` (the SPUR-scheduling layer) instead — it does not touch k0s.

   ``--force`` only skips the running-jobs guard (the jobs keep running) and lets
   the drain proceed past a ``PodDisruptionBudget`` or its timeout — it can evict
   pods that would otherwise block. ``remove-nodes`` refuses a control-plane node,
   the last remaining worker (would empty the member set), and any node not in a
   scoped cluster.

``spur k8s remove-nodes`` is distinct from ``spur node remove``: the former is the
graceful, k0s-aware path for shrinking a running cluster (drain + reset); the
latter is inventory-only (it does not drain pods or stop k0s) and is for
decommissioning a host from SPUR entirely. Use ``k8s remove-nodes`` first, then
``node remove`` if the host is also leaving SPUR.

Tear down
~~~~~~~~~

.. code-block:: bash

   spur k8s down            # stop + disable the k0s unit on every node
   spur k8s down --reset    # also `k0s reset` (destructive: wipes cluster state)

``--reset`` removes ``/var/lib/k0s`` on every node, along with the spurd-owned
systemd unit and cached join token, but leaves the WireGuard mesh (``spur0``)
intact. Purging the join token matters: a token minted against the torn-down
cluster's CA would fail the next join with a ``kubernetes-ca`` verification
error. To switch the CNI, tear down with ``--reset`` and bring the cluster back
up with the new ``cni`` setting.

For Users
---------

Users do not need Spur access — they interact with the cluster through the
standard Kubernetes tooling.

Get a kubeconfig
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spur k8s kubeconfig > mine.conf
   export KUBECONFIG=$PWD/mine.conf
   kubectl get nodes

A bare ``spur k8s kubeconfig`` mints a ServiceAccount kubeconfig scoped to
**your own** SPUR user/account namespace — no admin access required. It prints
to stdout so it can be redirected to a file.

Access control
'''''''''''''''

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Command
     - Who
     - Result
   * - ``spur k8s kubeconfig``
     - anyone
     - Own scoped kubeconfig (namespace = own SPUR account).
   * - ``spur k8s kubeconfig --user <name>``
     - cluster admin
     - ``<name>``'s scoped kubeconfig. Requesting anyone but yourself needs admin.
   * - ``spur k8s kubeconfig --admin``
     - cluster admin
     - The k0s cluster-admin kubeconfig (full access). Mutually exclusive with ``--user``.

"Cluster admin" is ``root``, or a user the accounting layer marks as an admin
association (see :doc:`../admin-guide/accounting`). ``--admin`` additionally
requires ``[cluster] allow_admin_kubeconfig = true`` in ``spur.conf`` — it is
``false`` by default, since the admin check on ``caller`` is not yet backed by
authenticated identity. With it off, get the cluster-admin kubeconfig directly
on the control-plane node instead: ``k0s kubeconfig admin``.

Run a workload
~~~~~~~~~~~~~~~

Use ``kubectl`` normally:

.. code-block:: bash

   kubectl get nodes -o wide
   kubectl create deployment web --image=nginx
   kubectl run -it --rm probe --image=busybox -- sh

Request GPUs
~~~~~~~~~~~~~

GPU worker nodes advertise ``amd.com/gpu`` (containerd injects the devices from a
CDI spec spurd writes on join). Request them in a pod spec:

.. code-block:: yaml

   apiVersion: v1
   kind: Pod
   metadata:
     name: gpu-probe
   spec:
     restartPolicy: Never
     containers:
       - name: rocm
         image: rocm/dev-ubuntu-24.04:latest
         command: ["rocm-smi"]
         resources:
           limits:
             amd.com/gpu: 1

Command reference
-----------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Command
     - Purpose
   * - ``spur k8s up [--nodes <hostlist>] [--partition <p>] [--selector k=v] [--control-plane-node <h> | --control-plane-nodes <h1,h2,h3>] [--replicas 1|3|5]``
     - Provision + start the cluster (idempotent). Admin only. Control-plane
       node(s) must lie within the ``--nodes``/``--partition``/``--selector``
       scope (or leave that scope empty for the whole inventory).
   * - ``spur k8s add-nodes --nodes <hostlist> | --partition <p> | --selector k=v``
     - Add worker nodes to a running scoped cluster (no down/reset). Admin only.
   * - ``spur k8s remove-nodes --nodes <hostlist> [--drain-timeout <secs>] [--force]``
     - Drain + ``k0s reset`` + remove a worker (destructive; re-add re-seeds state). Admin only.
   * - ``spur k8s status``
     - Cluster phase + per-node component state.
   * - ``spur k8s kubeconfig [--user <name>] [--admin]``
     - Print a kubeconfig (redirect to a file). Bare = own scope; ``--user``/``--admin`` need admin.
   * - ``spur k8s down [--reset]``
     - Stop the cluster; ``--reset`` also wipes k0s state. Admin only.
   * - ``spur k8s install-k0s [--version <tag>|latest] [--path <p>] [--force]``
     - Install the k0s binary on this node (local; run as root).

Configuration reference (``[cluster]``)
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Key
     - Default
     - Meaning
   * - ``enabled``
     - ``false``
     - Enable Spur-managed k0s. When off, spurd never touches systemd/k0s.
   * - ``distro``
     - ``k0s``
     - Kubernetes distribution SPUR manages. Only ``k0s`` is supported today.
   * - ``control_plane_node``
     - (first node)
     - Hostname of the k0s control plane.
   * - ``control_plane_replicas``
     - ``1``
     - HA control-plane count (1, 3, or 5). Overridden per-invocation by ``spur k8s up --replicas``.
   * - ``pod_cidr``
     - ``10.42.0.0/16``
     - Pod network; per-node /24s are carved from it.
   * - ``service_cidr``
     - ``10.43.0.0/16``
     - Service network.
   * - ``cni``
     - ``kuberouter``
     - ``kuberouter`` or ``calico`` (``bird`` over the mesh if enabled, else ``vxlan``).
   * - ``cni_mtu``
     - ``1450``
     - Calico MTU emitted into the generated k0s config (leaves WireGuard headroom).
   * - ``storage_provisioner``
     - ``local-path``
     - Storage Spur ships as the default StorageClass (``local-path`` or ``none``).
   * - ``local_path_dir``
     - ``/var/lib/local-path-provisioner``
     - On-node directory local-path stores PVs in; point at a big disk for data-heavy PVCs.
   * - ``k0s_version``
     - pinned
     - k0s release to install/run (a tag or ``latest``).
   * - ``k0s_binary``
     - ``/usr/local/bin/k0s``
     - Install path for the k0s binary.
   * - ``k8s_provisioning_timeout_secs``
     - ``600``
     - Seconds a node may stay non-``active`` during provisioning before the cluster is marked ``degraded``.
   * - ``allow_admin_kubeconfig``
     - ``false``
     - Allow ``spur k8s kubeconfig --admin`` to serve the cluster-admin kubeconfig over RPC.
