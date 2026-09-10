Upgrading Spur
==============

This page covers upgrading Spur binaries, whether on a single host or across a whole
cluster. It describes the ``spur self-update`` self-updater, re-running ``install.sh``,
and the Ansible playbooks that upgrade a running cluster with or without an outage.

.. note::

   There are two upgrade scopes: **upgrading the binaries on a single host** and
   **upgrading a whole cluster**. There is **no in-process hot-swap** that preserves
   running jobs inside a single daemon process. ``spur self-update`` only swaps the
   binaries on disk — it does **not** restart the daemons. The drain-aware, jobs-preserving
   cluster path is the Ansible ``rolling_upgrade.yml`` playbook (see below).

.. important::

   From this release ``spurctld`` opens a new listener on port **6823** for
   ``/livez`` and ``/readyz``, bound to every interface. Nothing else changes,
   and no upgrade step is needed, but a host whose firewall is built from a fixed
   port list gains a port that no rule covers. Review the rule, or set
   ``probes.enabled = false`` in ``spur.conf``.

Single-Host: ``spur self-update``
---------------------------------

``spur self-update`` downloads the latest release and replaces the ``spur``, ``spurctld``,
and ``spurd`` binaries on the current host. It is a single-host convenience: it does not
restart daemons and gives no drain or quorum protection, so it is not a substitute for the
cluster playbooks below.

Check whether an update is available:

.. code-block:: bash

   spur version --check

.. code-block:: text

   update available: 0.3.0 → v0.3.1
   Run `spur self-update` to install.

Install the update:

.. code-block:: bash

   spur self-update

``spur update`` is an alias for ``spur self-update``. Add ``--nightly`` to pull from the
nightly channel instead of the latest stable release:

.. code-block:: bash

   spur self-update --nightly

The updater downloads the release tarball, verifies it against its published SHA256
checksum, then replaces each binary atomically: the current binary is renamed to
``<name>.spur-old`` as a backup, the new binary is copied into place, and the backups are
deleted on success. If any copy fails, the ``.spur-old`` backup is restored. The install
directory is auto-detected as wherever the running ``spur`` binary already lives.

After a successful update the CLI prints:

.. code-block:: text

   Updated spur to v0.3.1
   Note: Restart running daemons (spurctld, spurd) to use the new version.

.. warning::

   ``spur self-update`` never restarts daemons. Running ``spurctld`` and ``spurd``
   processes keep executing the old binary until you restart them yourself:

   .. code-block:: bash

      sudo systemctl restart spurctld spurd

The ``[update]`` config block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optional ``[update]`` block in ``spur.conf`` controls the daemon startup update check.
Its fields are:

.. list-table::
   :header-rows: 1

   * - Field
     - Default
     - Effect
   * - ``check_on_startup``
     - ``true``
     - Check the GitHub releases API when the daemon starts and log if an update exists.
   * - ``auto_update``
     - ``false``
     - Download and install an available update automatically. Even when ``true``, the
       daemon is **never** auto-restarted.
   * - ``channel``
     - ``"stable"``
     - Release channel to check: ``"stable"`` or ``"nightly"``.
   * - ``cache_dir``
     - ``"/var/cache/spur"``
     - Directory for the update-check cache (1-hour TTL).

.. note::

   Even with ``auto_update = true``, a new binary on disk does not take effect until the
   daemon is restarted. The config block applies to ``spurctld``; ``spurd`` never
   auto-installs updates.

See :doc:`/admin-guide/configuration` for the full ``[update]`` field reference.

Single-Host: Re-running ``install.sh``
--------------------------------------

Re-running the one-line installer upgrades an existing single-host install in place. It
downloads the requested release, verifies its checksum, and copies the binaries over the
install directory (default ``~/.local/bin``):

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash

Pass ``nightly`` or a pinned ``vX.Y.Z`` to select a specific release, or set
``INSTALL_DIR`` to install elsewhere:

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash -s -- nightly
   curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash -s -- v0.3.1

Like ``spur self-update``, ``install.sh`` does not manage systemd units or restart
daemons. Restart ``spurctld`` and ``spurd`` yourself after re-installing.

Cluster Upgrades with Ansible
-----------------------------

For a multi-node cluster, the Ansible toolkit is the recommended upgrade path. Two
playbooks are supported; both reuse the same install, config, and health-check roles as
``deploy.yml``, so their behavior stays consistent.

Rebuild all three binaries from the same source tree together — they share a Raft
write-ahead-log schema, and mixing binaries from different builds can leave a controller
unable to parse a log written by a differently-versioned peer:

.. code-block:: bash

   cargo build --release -p spur-cli -p spurctld -p spurd

Binaries roll out by content, not version string: Ansible compares checksums, so an
unchanged re-run is a near no-op.

Full convergence (``deploy.yml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``deploy.yml`` is the simplest upgrade: it re-installs binaries and restarts every daemon
on every host in one play. This causes a **brief cluster-wide outage** — in-flight jobs are
disrupted, and in an HA setup all controllers bounce together, briefly losing the Raft
leader. It is non-destructive by default (state is preserved). Use it for topology changes
or when a short blip is acceptable:

.. code-block:: bash

   ansible-playbook playbooks/deploy.yml -i inventory/hosts.ini -e spur_binary_src=/path/to/target/release

.. note::

   Spur 0.3.0 has no online Raft membership change. Adding, removing, or reordering a
   **controller** fails early unless you also pass ``-e spur_wipe_state=true`` (a Raft
   reinit that wipes state). Compute agents are not Raft members, so they can be added or
   removed freely without a wipe.

Rolling upgrade (``rolling_upgrade.yml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rolling_upgrade.yml`` is the seamless, no-full-outage path: it upgrades one host at a
time so the cluster keeps scheduling and running jobs throughout.

.. code-block:: bash

   ansible-playbook playbooks/rolling_upgrade.yml -i inventory/hosts.ini -e spur_binary_src=/path/to/target/release

The playbook proceeds in order:

1. **Guard rails.** Abort if ``spur_wipe_state=true`` (never wipe Raft mid-upgrade), if
   ``spur_transport=wireguard`` (not yet supported by this playbook), or if the cluster is
   not already healthy (``spur nodes`` must return success).
2. **Upgrade controllers one at a time** (``serial: 1``, no failures tolerated), preserving
   Raft quorum. Each controller's binaries are force-reinstalled and the daemon restarted;
   the "wait for Raft leader" step is the health gate before moving to the next controller.
   The existing ``spur.conf`` is preserved unless you pass ``-e spur_overwrite_conf=true``.
3. **Drain and upgrade agents in batches.** For each agent: drain it from the controller
   (``spur node drain <node> --reason "ansible rolling upgrade"``), wait until its state is
   ``DRAINED`` or ``DOWN`` (running jobs finish first — drain never force-kills), swap the
   binary, restart ``spurd``, wait for the node to re-register, then resume it
   (``scontrol update NodeName=<node> State=RESUME``).
4. **Verify.** Submit a real test job to confirm the upgraded cluster schedules work.

The rolling upgrade is controlled with these ``-e`` flags:

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Effect
   * - ``spur_binary_src=<dir>``
     - *(unset)*
     - Directory of pre-built ``spur``/``spurctld``/``spurd`` binaries to roll out.
       Unset → install the published release via ``install.sh`` (``spur_version``).
   * - ``spur_rolling_batch_size=<N>``
     - ``1``
     - Agents upgraded per batch. Controllers are always upgraded one at a time regardless.
   * - ``spur_ignore_unreachable_agents=true``
     - ``false``
     - Skip agents unreachable over SSH instead of aborting.
   * - ``spur_skip_busy_agents=true``
     - ``false``
     - Leave a still-busy node on its current binary and continue, rather than aborting.
   * - ``spur_force_upgrade_busy_agents=true``
     - ``false``
     - Kill running jobs and containers on a busy node and upgrade it anyway. Affected jobs
       are marked ``NODE_FAIL``.
   * - ``spur_overwrite_conf=true``
     - ``false``
     - Re-render ``spur.conf`` from the Ansible template. Needed only when inventory
       variables changed; otherwise the existing config is preserved.

.. note::

   A larger ``spur_rolling_batch_size`` upgrades faster but drains more capacity at once. A
   single-controller cluster still has a short outage while its own controller restarts —
   true zero-downtime requires an HA quorum of 3 or more controllers.

Safe Upgrade Order
~~~~~~~~~~~~~~~~~~~

Follow this order for any cluster upgrade:

1. **Rebuild all three binaries together** from the same source tree — they share a Raft
   WAL schema and must stay version-matched.
2. **Upgrade controllers before agents.** Both playbooks do this automatically, one
   controller at a time to preserve quorum.
3. **Drain agents before swapping binaries.** The rolling playbook drains automatically; a
   running job blocks the swap unless you force it.
4. **Never wipe state during an upgrade.** Keep the default ``spur_wipe_state=false``.
   Wiping resets the Raft job-id counter and destroys accounting history.
5. **HA membership is fixed at init** in Spur 0.3.0. You can roll new binaries onto the
   existing controller set freely, but changing *which* hosts are controllers requires
   ``deploy.yml -e spur_wipe_state=true``.
6. **Only roll forward across an accounting schema migration.** The first upgraded
   controller applies pending PostgreSQL migrations on startup, so upgrade controllers
   before agents and let each migration finish before the next controller starts. One
   such migration widens the ``jobs`` job-id columns to 64-bit: it rewrites the table
   under an ``ACCESS EXCLUSIVE`` lock, so its duration scales with the row count and
   accounting writes block until it completes. **Rolling back to a pre-migration
   controller is not supported** — an older controller reads the widened columns as
   32-bit and its accounting queries fail against a migrated database. Take a database
   backup before upgrading if you need a recovery path.
7. **Roll forward, not back.** The Raft log gains entries — new job states, pending
   reasons, operations — as Spur evolves, and a controller replaying a log written by a
   newer build cannot parse them. A newer controller reads older logs fine, so the
   supported recovery from a bad upgrade is to roll forward, not to reinstall the
   previous version over a log the new one has already written.

Behavior Changes Between Releases
---------------------------------

Some releases change how an existing ``spur.conf`` behaves without that file being
edited. Review these before rolling binaries out.

.. _cgroup-upgrade-notes:

Job resource enforcement (``[cgroup]``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The release introducing the ``[cgroup]`` section changed what ``spurd`` writes for
a config that has **no** ``[cgroup]`` section. The first two can affect jobs that
ran fine before the upgrade; the rest relax an existing bound.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Control file
     - Before
     - After
     - Effect
   * - ``memory.max``
     - unset for ``--mem-per-cpu`` jobs
     - the memory the scheduler allocated
     - **A ``--mem-per-cpu`` job that ran unbounded is now capped, and is
       OOM-killed if it overruns.** Only the per-node request was read before,
       which those jobs do not set.
   * - ``memory.high``
     - unset
     - equal to ``memory.max``
     - **Memory-heavy jobs stall in reclaim before the OOM kill**, costing
       throughput.
   * - ``cpu.max``
     - CFS quota sized from ``--cpus-per-task``
     - unset
     - Relaxed: the cpuset is the CPU bound, as in Slurm.
   * - ``cpuset.cpus``
     - sized from ``--cpus-per-task``
     - sized from the node allocation
     - Relaxed: a multi-task job gets all the cores it was granted, not one
       task's worth.

To keep the previous behavior, put this in ``spur.conf`` on every compute node
**before** restarting ``spurd``:

.. code-block:: toml

   [cgroup]
   cpu_quota = true  # restore the CFS quota

``spurd`` reads ``[cgroup]`` only at startup, so this has to be in place before the
restart — ``scontrol reconfigure`` will not apply it afterwards.

.. note::

   Neither memory change has a knob of its own: ``memory.max`` and ``memory.high``
   are both written whenever ``constrain_ram_space`` is on, matching Slurm's
   ``cgroup/v2`` plugin. Setting ``constrain_ram_space = false`` restores the old
   behaviour but drops the memory ceiling for *every* job, not just the
   ``--mem-per-cpu`` ones. Prefer raising ``allowed_ram_percent`` (for example
   ``125``), which gives jobs headroom above their allocation and starts reclaim
   there, over turning memory constraints off.

Rolling back is safe with the section left in place: older binaries do not know
``[cgroup]`` and ignore it.

Job device access (``constrain_devices``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The release introducing ``constrain_devices`` attaches a default-deny BPF device
filter to every job cgroup, and it defaults to **on** — including for a
``spur.conf`` with no ``[cgroup]`` section, or one written before the field
existed.

**Effect:** a batch payload can open only the device nodes its allocation granted,
plus the base pseudo-devices and a built-in host-infrastructure set. A job that had
been reaching a GPU it was not allocated — by re-exporting ``ROCR_VISIBLE_DEVICES``,
or because it never read the variable — now fails that ``open`` with ``EPERM``.

**A job can also be denied a host device node its allocation never covered**, and
that is the risk to plan the rollout around. Some device nodes belong to the node
rather than to any job, so no allocation hands them out and nothing puts them in the
allow-list. The main case is RDMA/InfiniBand verbs under ``/dev/infiniband/`` —
needed by MPI, and by NCCL or RCCL over InfiniBand, which makes this reach
**non-GPU** jobs. These are allowed by default, so the common cases keep working with
no configuration; see :ref:`device-filter-implicit-allow` for the full implicit set.

The vendor control nodes a GPU runtime initializes through, such as ``/dev/nvidiactl``
and ``/dev/nvidia-uvm``, reach a job through its allocation's CDI device edits. A node
configured through GRES rather than a CDI spec does not list them as devices, so a GPU
job there fails with ``EPERM`` on those nodes until they are named in
``extra_device_paths``.

A site whose jobs open some other host device node will still see ``EPERM``. Name
those paths rather than turning the filter off:

.. code-block:: toml

   [cgroup]
   extra_device_paths = ["/dev/some-site-device"]

Roll the agent out node by node and watch job output for ``EPERM`` on device paths.
To defer the change entirely:

.. code-block:: toml

   [cgroup]
   constrain_devices = false

Installing the filter needs ``CAP_BPF`` and ``CAP_NET_ADMIN`` (or ``CAP_SYS_ADMIN``);
a cgroup-device program is a net-admin program type, so ``CAP_BPF`` alone is not enough.
An agent without them logs ``device filter not installed`` and runs the job with no
device isolation, so an unprivileged ``spurd`` behaves as before — unless ``required = true``, which
turns that degradation into a refused launch.

See Also
--------

- :doc:`ansible`
- :doc:`uninstalling`
- :doc:`/admin-guide/configuration`
