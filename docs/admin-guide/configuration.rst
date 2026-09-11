Configuration Reference (spur.conf)
===================================

``spur.conf`` is a TOML file describing controller, node, accounting, scheduling,
and network settings. The default location is ``/etc/spur/spur.conf`` (the Ansible
layout installs it at ``<spur_home>/etc/spur.conf``). Only ``cluster_name`` is
required; every section has a default and may be omitted, and unknown keys are
silently ignored. The controller validates the file on load.

The sections below are grouped by subsystem. Every field lists its type, default,
and meaning.

.. note::

   ``spurctld`` reads every section of ``spur.conf``. ``spurd`` reads the same file
   but only for local agent settings (``[hooks]``, ``[devices]``, ``rlimits.memlock``,
   ``[cgroup]``, ``[cluster]``, and ``[mpi]``); its identity and networking come from
   CLI flags.
   Node CPU, memory, and GRES are reported by each agent when it registers, not
   declared here — ``[[nodes]]`` only overlays scheduling policy onto nodes that
   have already registered.

Minimal configuration
----------------------

A working single-node configuration needs a cluster name and one partition; nodes
join by registering, so the ``[[nodes]]`` block below is optional and only tags
them for ``--constraint`` matching. Accounting, WireGuard, and the k0s cluster
manager are all off unless explicitly configured.

.. code-block:: toml

   cluster_name = "mi300x-cluster"

   [controller]
   listen_addr = "[::]:6817"
   state_dir = "/var/spool/spur"
   max_batch_requeue = 5

   [scheduler]
   plugin = "backfill"
   interval_secs = 1
   max_jobs_per_cycle = 10000
   fairshare_halflife_days = 14

   [accounting]
   database_url = "postgresql://spur:spur@localhost/spur"

   [auth]
   plugin = "none"

   [[partitions]]
   name = "gpu"
   default = true
   state = "UP"
   nodes = "mi300,mi300-2"
   max_time = "7-00:00:00"
   default_time = "1:00:00"
   min_nodes = 1
   priority_tier = 1

   [[nodes]]
   names = "mi300,mi300-2"
   features = ["mi300x", "rocm6"]
   weight = 1

The full annotated example — including label selectors, account restrictions, and
the k0s cluster manager — lives at ``examples/spur.conf`` in the repository.

.. _reload-scope:

Applying configuration changes
------------------------------

After editing ``spur.conf``, ``scontrol reconfigure`` re-reads the file and applies
it to the running controller, which makes the file authoritative: runtime-only
changes made with ``scontrol update`` are overwritten by the file's values.
``spurctld`` does not reload on ``SIGHUP`` — ``scontrol reconfigure`` is the only
trigger.

Not every field can be applied to a running daemon. Each section below records what
its fields need in a **Reload** column — or, where every field in a section shares
the same scope, in a single ``Reload:`` line above the table:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Reload
     - Meaning
   * - Live
     - Applied by ``scontrol reconfigure``; no restart needed.
   * - Restart
     - Read once when ``spurctld`` starts. ``reconfigure`` re-reads the value but
       does not apply it; restart the controller.
   * - Agent restart
     - Consumed by ``spurd`` on each compute node. ``reconfigure`` reaches only the
       controller, so restart ``spurd`` on every node.
   * - Client
     - Read by the ``spur`` CLI from the submitting host on each invocation. Neither
       ``reconfigure`` nor a daemon restart applies.
   * - Not implemented
     - Parsed and validated, but no code consumes it. Setting it has no effect.

The restart-only set mirrors Slurm, where ports, ``StateSaveLocation``, ``AuthType``,
and the plugin set also require a daemon restart.

**Leader-only, in an HA cluster.** ``reconfigure`` is handled by the Raft leader and
swaps only that controller's in-memory config; no Raft log entry carries the new
file, so followers keep the config they loaded at startup until they restart (in
Kubernetes they re-read the same ConfigMap). ``[[partitions]]`` is the exception —
partition changes replicate through the write-ahead log — but a follower re-derives
node features and weight from its own pre-reconfigure ``[[nodes]]`` blocks. Do not
rely on reconfigured non-partition state surviving an immediate failover; roll the
controllers to converge them.

.. warning::

   A partition removed from ``spur.conf`` is skipped rather than deleted when it
   still has active jobs. ``scontrol reconfigure`` reports success either way, and
   the skip is recorded only in the controller log. Drain a partition before
   removing it from the file.

Top-level keys
--------------

.. list-table::
   :header-rows: 1
   :widths: 18 18 14 14 36

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``cluster_name``
     - string
     - **(required)**
     - Live
     - Cluster name. An empty value fails to load with ``missing required field:
       cluster_name``. Reported by ``scontrol show config``, which reads it from
       the live controller rather than the login node's local config. Changing it
       live re-labels metrics mid-series; prefer a restart.
   * - ``licenses``
     - table<string, integer>
     - ``{}``
     - Live
     - Cluster-wide license pool, e.g. ``{ fluent = 20, comsol = 5 }``. Jobs
       consume licenses via ``--licenses``. Availability is derived as total minus
       in-use, so changing a total cannot strand a running job's holding.

``[controller]``
----------------

Controller daemon (``spurctld``) network endpoints, state storage, job-ID range,
and Raft high-availability topology.

.. list-table::
   :header-rows: 1
   :widths: 24 12 18 14 32

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``listen_addr``
     - string
     - ``"[::]:6817"``
     - Restart
     - gRPC listen address serving ``SlurmController`` and ``SlurmAccounting``.
   * - ``rest_addr``
     - string
     - ``"[::]:6820"``
     - Restart
     - REST API listen address.
   * - ``hosts``
     - [string]
     - ``["localhost"]``
     - Client
     - Controller hostname(s); the first is primary. The CLI builds failover
       endpoints from these hosts plus the port of ``listen_addr``. Read by the
       CLI on each invocation, not by ``spurctld``.
   * - ``state_dir``
     - string
     - ``"/var/spool/spur"``
     - Not implemented
     - Ignored. The controller always uses its ``--state-dir`` flag, which
       defaults to ``/var/spool/spur``.
   * - ``max_job_id``
     - integer
     - ``999999999``
     - Not implemented
     - Intended as the job-ID wrap point. No code consumes it. Job IDs are 32-bit
       unsigned and stored as 64-bit in the accounting database, so ids above
       ``i32::MAX`` are recorded correctly; the counter still wraps to zero past
       ``u32::MAX`` and would re-issue ids that collide with existing rows.
   * - ``first_job_id``
     - integer
     - ``1``
     - Restart
     - Job ID assigned to the first submitted job.
   * - ``peers``
     - [string]
     - ``[]``
     - Restart
     - Raft HA peers as ``"host:port"``. Empty means single-node. The list must be
       identically ordered on every controller — node IDs derive from position.
       Example: ``["node1:6821", "node2:6821", "node3:6821"]``.
   * - ``node_id``
     - integer
     - none
     - Restart
     - This controller's Raft ID. Normally unset (single-node always uses ``1``).
       When set it must fall in ``1..=peers.len()`` and equal this host's position
       in ``peers``.
   * - ``raft_listen_addr``
     - string
     - ``"[::]:6821"``
     - Restart
     - Internal Raft gRPC listen address, separate from the client API.
   * - ``heartbeat_timeout_secs``
     - integer
     - none
     - Restart
     - Seconds without a heartbeat before a node is marked Down. Unset by
       default; the controller applies a 90-second fallback when absent.
   * - ``max_batch_requeue``
     - integer
     - ``5``
     - Live
     - Maximum automatic requeues (excluding preemption) before a job is held with
       ``JobHoldMaxRequeue``. Must be ``>= 1``; ``0`` is a validation error.
   * - ``max_launch_backoff_secs``
     - integer
     - ``300``
     - Live
     - Upper bound on the exponential backoff applied before retrying a failed
       job launch.
   * - ``hold_on_prolog_fail``
     - bool
     - ``true``
     - Live
     - Hold a job whose ``prolog_slurmctld`` hook fails instead of requeuing it.
   * - ``terminal_job_retention_secs``
     - integer
     - ``3600``
     - Live
     - How long a completed job stays in controller memory before eviction.
       Accounting rows in PostgreSQL are unaffected.
   * - ``dispatch_reject_cooldown_secs``
     - integer
     - ``30``
     - Live
     - How long a node is skipped for dispatch after rejecting a launch or
       failing to be reached.
   * - ``agent_connect_timeout_secs``
     - integer
     - ``5``
     - Live
     - Budget for establishing a controller-to-agent connection. Range 0-600.
       ``0`` falls back to the operating system's TCP timeout, which is
       typically around two minutes and is not configurable from here.
   * - ``agent_keepalive_interval_secs``
     - integer
     - ``10``
     - Live
     - HTTP/2 ping interval on an open agent connection. Range 0-600. ``0``
       disables keepalive entirely. Pings are sent while a request is in flight, so
       this is what detects a node that accepted the connection and then went
       silent — total detection time is roughly this plus
       ``agent_keepalive_timeout_secs``. Lower it for faster detection at the
       cost of more ping traffic per node; raise it if an agent
       implementation objects to frequent pings.
   * - ``agent_keepalive_timeout_secs``
     - integer
     - ``10``
     - Live
     - How long to wait for a ping response before dropping the connection.
       Range 1-600 whenever keepalive is on; ``0`` is rejected because it
       marks every ping overdue the moment it is sent. Ignored entirely when
       ``agent_keepalive_interval_secs`` is ``0``.
   * - ``dispatch_timeout_secs``
     - integer
     - ``300``
     - Live
     - Ceiling on a single launch, allocation-register, or multi-node PMIx
       prepare RPC to an agent. The agent runs the node prolog and any
       container image unpack before it answers, so this must exceed your
       slowest prolog. Range 0-86400;
       ``0`` disables it. A node that has died or become unreachable is
       detected sooner by channel keepalive, when keepalive is enabled; an
       agent that is still running but whose launch never completes is
       bounded only by this value. A node that exceeds it is skipped for new
       dispatch for the same span, and is not marked down, so it still
       appears available in ``sinfo`` while being skipped.
   * - ``job_info_visibility``
     - string
     - ``redacted``
     - Live
     - How much of another user's job an identified non-owner (non-admin) may
       read via ``get_job`` / ``get_job_steps``. ``redacted`` (default) shows
       identity, state, timing, and account but blanks the working directory,
       command, submit line, stdio paths, comment, the allocated, requested,
       and planned node lists, and both the allocated and requested resource
       detail (``ReqTRES``, ``Features``, and the per-node minima);
       ``owner_only`` returns ``NOT_FOUND`` for other users' jobs; ``full`` is the
       legacy behaviour where every field is visible to any caller. Owners and
       admins always see the full record. Scoping applies only to identified
       callers — under ``auth.mode = required``, or when a credential is
       presented under ``permissive``; with authentication disabled or no
       credential presented, the full record is returned (so no-auth deployments
       and internal consumers are unaffected).

``[accounting]``
----------------

PostgreSQL-backed accounting, fairshare, and QOS enforcement. Accounting runs
in-process inside ``spurctld`` (served on port 6817) — there is no separate
``slurmdbd``.

.. list-table::
   :header-rows: 1
   :widths: 24 10 20 14 32

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``database_url``
     - string
     - ``""``
     - Restart
     - PostgreSQL connection string. A non-empty value enables accounting; empty
       disables it entirely. Example: ``"postgresql://spur:spur@localhost/spur"``.
       The controller connects in the background and retries with backoff, so an
       unreachable database at startup does not stop it from connecting later.
   * - ``fairshare_refresh_secs``
     - integer
     - ``300``
     - Restart
     - How often (seconds) to refresh fairshare and QOS caches from the database.
       The interval is baked into the refresh loops when they are spawned.
   * - ``grp_wall_window_days``
     - integer
     - ``14``
     - Restart
     - Trailing window over which a QOS's wall-clock consumption is measured for
       the ``grpwall`` limit. Must be between ``1`` and ``3650``; a zero window
       would measure nothing and silently stop every ``grpwall`` budget applying,
       so it is rejected at startup. Independent of
       ``scheduler.fairshare_halflife_days``: that fades usage for priority
       scoring, this is a hard budget cutoff.
   * - ``default_qos``
     - string
     - ``""``
     - Live
     - Cluster-wide fallback QOS, applied at submit when a job resolves to no QOS
       (the analog of Slurm's ``normal``). Must name an existing QOS; empty means
       no fallback.
   * - ``require_qos``
     - bool
     - ``false``
     - Live
     - Reject at submit any job that still has no QOS after the resolution chain.
       Mirrors Slurm's ``AccountingStorageEnforce=qos``.
   * - ``require_association``
     - bool
     - ``false``
     - Live
     - Reject at submit any job whose user resolves to no account: no
       ``--account`` given and no default account on file. Unconditional, like
       ``require_qos``. Mirrors Slurm's ``AccountingStorageEnforce=associations``.
   * - ``txn_retention_days``
     - integer
     - unset
     - Restart
     - Delete admin audit-log (``txn``) rows older than this many days. Unset (the
       default) or ``0`` disables purging (rows kept forever, matching Slurm's
       default purge-off behavior); a positive value enables it. See :doc:`accounting`.

See :doc:`accounting` for how ``default_qos`` and ``require_qos`` interact with the
per-job QOS resolution chain, and how ``require_association`` interacts with the
per-job account resolution chain.

``[scheduler]``
---------------

Scheduling loop cadence, per-cycle limits, and fairshare decay.

.. list-table::
   :header-rows: 1
   :widths: 28 10 16 14 32

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``plugin``
     - string
     - ``"backfill"``
     - Restart
     - Scheduler plugin name. Backfill is the only implemented scheduler; this
       value is a display label reported by ``sdiag`` and does not select an
       algorithm.
   * - ``interval_secs``
     - integer
     - ``1``
     - Restart
     - How often (seconds) the scheduler runs. The loop cadence and the
       preemption requeue hold are fixed at startup; the launch-backoff base
       does re-read this value live.
   * - ``max_jobs_per_cycle``
     - integer
     - ``10000``
     - Restart
     - Maximum number of jobs evaluated per scheduling cycle.
   * - ``fairshare_halflife_days``
     - integer
     - ``14``
     - Restart
     - Fairshare usage decay half-life, in days.
   * - ``default_time_limit_minutes``
     - integer
     - ``0``
     - Live
     - Cluster-wide fallback wall-time (minutes) for a job that sets no ``-t`` and
       lands on a partition with no ``DefaultTime``. ``0`` disables the fallback,
       leaving such jobs unbounded. Set > 0 to bound otherwise-unlimited jobs.
       When enabled, a ``-t``-less job on a partition that has a finite ``MaxTime``
       but no ``DefaultTime`` defaults to that partition's ``MaxTime`` (for a
       multi-partition request, the smallest ``MaxTime`` among them), not this
       flat value. Prior to this release the setting was inert (never applied);
       it now takes effect, and its default changed from ``60`` to ``0`` so
       ``-t``-less jobs stay unbounded exactly as before. A site that had set it
       expecting an effect will now see that effect. A job the partitions leave
       unbounded still takes its QOS or association ``MaxWall`` when one is set —
       see :ref:`maxwall-default`.
   * - ``enforce_part_limits``
     - string
     - ``NO``
     - Live
     - Whether partition wall-time limits are enforced at submit. ``NO`` admits
       over-limit jobs and lets them pend with a ``PartitionTimeLimit`` reason.
       ``ALL`` rejects unless the job fits every requested partition; ``ANY``
       rejects only when it fits none. Mirrors Slurm's ``EnforcePartLimits``.
   * - ``complete_wait_secs``
     - integer
     - ``300``
     - Live
     - Maximum seconds a job may sit in COMPLETING before it is force-finished.
   * - ``max_user_priority``
     - integer
     - ``1000``
     - Live
     - Highest base priority a non-admin may request, at submit (``--priority``)
       or via ``scontrol update``. Requests above this are clamped down, not
       rejected; at submit the clamp is returned to the caller as a warning, while
       on the ``scontrol update`` path (which has no response field) it is only
       logged. Defaults to the base priority (``1000``), so a non-admin can lower
       but not raise priority, matching Slurm, where boosting priority is
       operator-only. Raise it to grant users a band above the baseline. The
       ceiling applies only to identified non-admin callers: admins are exempt, and
       so are callers with no verified identity (``auth.mode = disabled``, or
       ``permissive`` with no credential), where the cluster trusts the client as
       before.
   * - ``inactive_limit_secs``
     - integer
     - ``0``
     - Live
     - Reap an interactive allocation (``salloc``/``srun``) whose client has sent
       no keepalive for this many seconds, freeing the nodes. ``0`` (the default)
       disables reaping. Mirrors Slurm's ``InactiveLimit``. Once enabled, *every*
       interactive allocation is subject to reaping regardless of client version:
       a client too old to send keepalives is reaped once idle past the limit, so
       upgrade all ``spur`` CLI clients before enabling this. Must be at least
       twice the client keepalive interval (60 seconds); smaller non-zero values
       are rejected at startup so a live client is never reaped between pings.
   * - ``resv_overrun_minutes``
     - integer
     - ``0``
     - Live
     - Grace minutes after a reservation ends before its still-running jobs are
       cancelled.
   * - ``preempt_type``
     - string
     - ``"none"``
     - Live
     - Controls cross-QOS preemption eligibility. ``"none"`` (default) applies no
       QOS-level restrictions — any job with a sufficient priority gap may preempt
       any other. ``"qos_priority"`` enforces the per-QOS ``preempt`` allow-list:
       a pending job may only preempt a running job when the pending job's QOS
       explicitly lists the running job's QOS name in its ``preempt`` field. An
       empty allow-list means the QOS may not preempt anything. Mirrors Slurm's
       ``PreemptType=preempt/qos``. See :doc:`accounting` for the QOS
       ``preempt`` field.
   * - ``preempt_exempt_time``
     - integer
     - ``0``
     - Live
     - Cluster-wide minimum number of seconds a job must have been running before
       it becomes eligible for preemption. ``0`` (default) means a job is
       immediately eligible. Can be overridden per-partition (``preempt_exempt_time``
       in ``[[partitions]]``) and per-QOS (``preemptexempttime`` via
       ``sacctmgr``); the most specific value wins (QOS > partition > global).
       Mirrors Slurm's ``PreemptExemptTime``.

.. note::

   A pending job that needs more nodes than currently have free capacity is
   re-evaluated every scheduling cycle, but does not hold a reservation on
   any node while it waits. In practice this means: as soon as any node the
   job could use becomes free, that capacity is available to any other
   pending job that fits it right now — including one submitted after the
   larger job and with lower priority. A large multi-node job can therefore
   sit pending indefinitely behind a steady stream of smaller jobs, even
   though it has higher priority than every one of them individually,
   because none of them is ever compared against it directly; each is only
   checked against whatever capacity is free at that moment.

   The reliable way to guarantee a high-priority job is not indefinitely
   delayed by lower-priority ones is preemption, not priority alone.
   Priority only affects the order pending jobs are considered for
   available capacity — it does not reclaim capacity already given to a
   running job. Configure:

   - A priority gap of more than 2× between the jobs that must run and the
     jobs they need to be able to displace (via base ``--priority``, QOS
     ``priority``, or a combination — see :doc:`accounting` for how the
     effective priority gap is computed).
   - ``preempt_type`` (and, if a running job should not simply be killed,
     ``preemptmode=requeue`` or ``suspend`` on its QOS) so that gap actually
     triggers preemption instead of only affecting scheduling order.

   With both in place, a high-priority job that cannot find free capacity
   will preempt a lower-priority running job holding the capacity it needs,
   rather than waiting for it to finish on its own.

.. note::

   Backfill protection for a large multi-node job (holding a node against
   smaller jobs until the job it's waiting on can start) is only as good as
   the wall-time information available for the jobs already running. A job
   with no wall-time is assumed to run for up to a year, but that number is
   only a placeholder used to size the *reservation itself*; it is not
   compared against how long an incoming job actually needs the node, so a
   fully unbounded cluster gets little practical protection from this
   mechanism — a large job can still be starved by a continuous stream of
   equally unbounded smaller jobs. Setting ``default_time_limit_minutes``
   (or a per-partition ``DefaultTime``/``MaxTime``) so jobs carry real
   wall-time information makes backfill reservations meaningful. For a job
   that must not be starved by anything wall-time can't bound, use
   preemption (``preempt_type``, above) instead.

``[auth]``
----------

How client requests are authenticated.

.. list-table::
   :header-rows: 1
   :widths: 20 10 16 14 40

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``plugin``
     - string
     - ``"jwt"``
     - Restart
     - Constrains startup only: ``"munge"`` and any unrecognised value are
       rejected rather than silently ignored, and ``"none"`` with
       ``mode = "required"`` is refused as contradictory. Nothing reads it
       afterwards — whether a presented credential is verified is decided by
       ``mode`` and ``jwt_key`` alone, so ``plugin = "none"`` does **not** turn
       verification off.
   * - ``mode``
     - string
     - ``"permissive"``
     - Restart
     - ``"disabled"`` ignores credentials entirely, even valid ones.
       ``"permissive"`` verifies a credential that is presented — an invalid or
       malformed one is always refused — but allows a request that carries none.
       ``"required"`` refuses every request without a valid credential.
   * - ``jwt_key``
     - string
     - none
     - Restart
     - Signing key for user credentials (``spur token user``) and node admission
       tokens, given as a file path or inline value. Deliberately not reloadable:
       swapping it live would immediately invalidate every outstanding token.
   * - ``allow_root_jobs``
     - bool
     - ``false``
     - Agent restart
     - Permit jobs to run as UID 0. Consumed by ``spurd`` at its own startup.

.. warning::

   Under the default ``mode = "permissive"``, a caller that presents no
   credential is unauthenticated, and the username it asserts in the request is
   taken at face value. Identity-dependent decisions — job ownership, reservation
   management, job-info visibility — are then only as trustworthy as the network.
   Set ``mode = "required"`` (with a ``jwt_key``) to make them enforceable, and
   restrict the controller port at the network layer either way. ``spurctld``
   warns at startup whenever it binds a non-loopback address without
   ``required``.

.. note::

   When ``jwt_key`` is unset, admission tokens are signed with a well-known
   built-in key and are therefore forgeable; set an explicit key before enabling
   token admission (``[admission] mode = "token"``).

.. _privileged-operations:

Privileged operations
~~~~~~~~~~~~~~~~~~~~~

The control-plane mutations that define cluster tenancy — partitions, node state
and labels (``scontrol update NodeName=``, ``spur node drain``, ``spur node
remove``), ``reconfigure``, admission tokens, reservations, and the accounting
account/user/QOS records — require a **cluster admin**. A caller with
a verified non-admin identity is refused with ``PermissionDenied``. A caller with
*no* verified identity is allowed, so that ``disabled`` and credential-less
``permissive`` deployments keep working; under ``mode = "required"`` every caller
is authenticated, so the bar binds everyone.

Admin means one of the following:

* a credential minted with ``spur token user --admin``;
* a credential for the user ``root``;
* an accounting admin level of ``Admin`` (see :doc:`accounting`).

Only the first counts for the accounting service's own mutations: it holds no
association cache and does not special-case ``root``, so it accepts the token
claim alone. The controller RPCs honour all three.

Reservations are the one exception, and are stricter in two ways. An
unidentified caller is **not** waved through, and membership of ``sudo`` or
``wheel`` also qualifies. Creating, updating, or deleting a reservation is
allowed when any of these holds:

* the caller is a cluster admin, as above;
* the caller resolves to UID 0 on the controller;
* the caller is a member of the ``sudo`` or ``wheel`` group, as the controller's
  own NSS resolves the name — so a site using LDAP or SSSD grants this by group
  membership, and a controller with no shared user directory can only resolve
  local accounts.

Anyone else is refused, as is a name the controller cannot resolve at all — a
caller it cannot vouch for is denied, not allowed. The rule mirrors the one the
CLI applies locally before it ever connects, which is the point — a client that
does not go through the CLI cannot skip it. The cost is that a ``sudo``/``wheel``
operator must be resolvable *on the controller*, not only on the login node.

Ownership is not part of this decision. The creator is recorded as ``Owner`` for
attribution and shown by ``scontrol show reservation``, but any operator may
update or delete any reservation, matching Slurm's operator semantics.

.. warning::

   Without a credential the controller can only check the username the client
   asserted, so under ``permissive`` this stops an unprivileged client but not a
   deliberately crafted request. ``mode = "required"`` is what makes it a
   boundary, because there the name comes from the verified credential.

``[[partitions]]``
------------------

An array of tables — one ``[[partitions]]`` block per partition (queue). Membership
is the union of the ``nodes`` hostlist pattern and the ``selector`` label match.

**Reload: Live** for every field below. Partitions are the only section that also
replicates to follower controllers, because ``reconfigure`` applies them through the
write-ahead log rather than the in-memory config swap. A partition still running
jobs is skipped rather than deleted (see :ref:`reload-scope`).

Time values for ``max_time`` and ``default_time`` accept ``minutes``,
``minutes:seconds``, ``hours:minutes:seconds``, ``days-hours``,
``days-hours:minutes``, and ``days-hours:minutes:seconds``, as well as values
with suffixes and ``INFINITE`` / ``UNLIMITED``. Partition limits round up to whole
minutes. Seconds fields carry into whole minutes before rounding: ``0:0:90`` is
two minutes, and ``2-0:0:90`` is two days and two minutes.

.. warning::

   Two-field colon values now mean minutes:seconds, not hours:minutes. When
   upgrading from the previous interpretation, replace values intended as
   hours:minutes with explicit ``HH:MM:SS`` in partition configuration and job
   ``--time`` arguments. For example, use ``30:00:00`` for thirty hours;
   ``30:00`` now means thirty minutes. Bare ``days-hours`` values such as
   ``2-12`` now produce a finite duration of two days and twelve hours.

.. list-table::
   :header-rows: 1
   :widths: 22 18 20 40

   * - Field
     - Type
     - Default
     - Description
   * - ``name``
     - string
     - **(required)**
     - Partition (queue) name.
   * - ``default``
     - bool
     - ``false``
     - Mark this as the cluster default partition.
   * - ``state``
     - string
     - ``"UP"``
     - Partition state, parsed case-insensitively: ``UP``, ``DOWN``, ``DRAIN``;
       anything else becomes Inactive.
   * - ``nodes``
     - string
     - ``""``
     - Hostlist pattern of member nodes, e.g. ``"gpu[001-008]"`` or
       ``"mi300,mi300-2"``.
   * - ``selector``
     - table<string, string>
     - ``{}``
     - Label selector; a node joins if it matches **all** key=value pairs. Unioned
       with ``nodes``.
   * - ``max_time``
     - string
     - UNLIMITED
     - Maximum wall time. Slurm format: ``"72:00:00"``, ``"7-00:00:00"``, ``"60"``
       (minutes), or ``INFINITE`` / ``UNLIMITED``. Suffixed durations are also
       accepted: ``"1h"``, ``"90m"``, ``"1h40m"``, ``"2d12h"``, ``"30s"``.
   * - ``default_time``
     - string
     - UNLIMITED
     - Default wall time for jobs that omit ``--time``. Same format as ``max_time``.
   * - ``max_nodes``
     - integer
     - none
     - Maximum nodes per job.
   * - ``min_nodes``
     - integer
     - ``1``
     - Minimum nodes per job.
   * - ``allow_accounts``
     - [string]
     - ``[]``
     - Accounts permitted to submit to this partition (allow-list).
   * - ``deny_accounts``
     - [string]
     - ``[]``
     - Accounts denied submission to this partition (deny-list).
   * - ``priority_tier``
     - integer
     - ``0``
     - Priority ranking for this partition. Jobs on a higher-tier partition are
       treated as more urgent than jobs on a lower-tier partition, even if their
       raw submitted priority is the same. This allows a "premium" partition to
       bump jobs off a "standard" partition without the admin manually adjusting
       job priorities. A job that spans multiple partitions inherits the highest
       tier among them.
   * - ``preempt_mode``
     - string
     - ``"off"``
     - What the scheduler does to a running job when a higher-priority job needs
       its node.

       ``"cancel"`` — the running job is stopped and removed from the queue.
       ``"requeue"`` — the running job is stopped and put back in the queue;
       it will start again automatically once a node is free.
       ``"suspend"`` — the running job is paused (not stopped). It keeps its
       node allocation and continues automatically once the higher-priority job
       finishes. Because the node stays occupied, any other job that also needs
       that node exclusively will have to wait until the paused job either
       finishes or is cancelled.
       ``"off"`` (default) — running jobs in this partition are never kicked
       out. The scheduler will wait for a free slot instead of preempting.

       A job's QOS can change what happens to *that specific job* when it is
       kicked out (see ``preemptmode`` in :doc:`accounting`). The partition
       field is the on/off switch: preemption is only attempted at all when
       this is set to something other than ``"off"``.
   * - ``preempt_exempt_time``
     - integer or null
     - ``null`` (inherit global)
     - Per-partition override for the minimum seconds a job must have been running
       before it is eligible for preemption. Overrides ``scheduler.preempt_exempt_time``
       for jobs in this partition. Can be further overridden per-QOS by the QOS's
       ``preemptexempttime`` field. Can also be set at runtime without restart via
       ``scontrol update PartitionName=<name> PreemptExemptTime=<secs>``;
       use ``scontrol update PartitionName=<name> ClearPreemptExemptTime=yes``
       to revert to the global default.

``[[nodes]]``
-------------

An array of tables overlaying scheduling policy onto nodes. Match nodes by hostlist
pattern (``names``) or by label (``selector``); an entry applies if **either**
matches, and the first matching entry wins.

.. list-table::
   :header-rows: 1
   :widths: 20 16 12 16 36

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``names``
     - string
     - ``""``
     - Live
     - Hostlist pattern, e.g. ``"gpu[001-008]"``, or the literal ``ALL``. Optional
       when ``selector`` is used.
   * - ``selector``
     - table<string, string>
     - ``{}``
     - Live
     - Apply this entry to nodes matching **all** key=value pairs.
   * - ``cpus``
     - integer
     - ``0``
     - Not implemented
     - CPU count. Reported by the agent at registration; this value is ignored.
   * - ``memory_mb``
     - integer
     - ``0``
     - Not implemented
     - Memory in MB. Reported by the agent at registration; this value is ignored.
   * - ``gres``
     - [string]
     - ``[]``
     - Not implemented
     - Generic resources. Reported by the agent at registration; declare local
       GRES pools under ``[devices]`` on the node instead.
   * - ``features``
     - [string]
     - ``[]``
     - Live
     - Node features/tags for ``--constraint`` matching. A node matching no entry
       has its features cleared.
   * - ``address``
     - string
     - none
     - Live
     - Fallback address used until the agent registers one. It never overrides an
       address an agent has already reported.
   * - ``weight``
     - integer
     - ``1``
     - Live
     - Scheduling weight; higher is preferred. Reset to ``1`` for a node matching
       no entry.

.. note::

   ``[[nodes]]`` is not a node roster. A node joins the cluster when ``spurd``
   registers with the controller, so adding a block here does not create a node,
   and removing one does not remove a node — it only clears that node's features
   and weight. Remove a node with ``spur node remove <node>``, which takes a
   :ref:`cluster admin <privileged-operations>`. This differs from Slurm, where
   ``NodeName=`` lines in ``slurm.conf`` define the roster.

``[network]``
-------------

WireGuard mesh networking and the agent port.

.. list-table::
   :header-rows: 1
   :widths: 24 10 18 16 32

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``wg_enabled``
     - bool
     - ``false``
     - Not implemented
     - Intended to enable WireGuard mesh networking. No code reads it; the mesh
       is driven by ``[cluster] enabled`` instead.
   * - ``wg_cidr``
     - string
     - ``"10.44.0.0/16"``
     - Restart
     - CIDR for WireGuard address allocation. Validated as an IPv4 CIDR when
       ``[cluster]`` is enabled.
   * - ``wg_interface``
     - string
     - ``"spur0"``
     - Not implemented
     - Superseded by the ``SPUR_WG_INTERFACE`` environment variable read by
       ``spurd``, which defaults to ``spur0``.
   * - ``wg_port``
     - integer
     - ``51820``
     - Not implemented
     - Intended as the WireGuard listen port. No code reads it.
   * - ``agent_port``
     - integer
     - ``6818``
     - Not implemented
     - Each agent advertises its own port at registration, and the controller
       falls back to ``6818`` when it does not. Set the agent's port through its
       ``--listen`` address instead.
   * - ``reject_loopback_comm_addr``
     - bool
     - ``false``
     - Live
     - Reject a node registration whose advertised address is a loopback
       address, which would otherwise make the node unreachable from the
       controller.

``[logging]``
-------------

**Reload: Not implemented** for every field below. The section is parsed but no
daemon reads it.

.. list-table::
   :header-rows: 1
   :widths: 20 14 20 46

   * - Field
     - Type
     - Default
     - Description
   * - ``level``
     - string
     - ``"info"``
     - Intended log level. Use the ``--log-level`` flag or the ``RUST_LOG``
       environment variable instead.
   * - ``format``
     - string
     - ``"text"``
     - Intended log format. Output format is not configurable.
   * - ``file``
     - string
     - none
     - Intended log file path. Logging to a file is not implemented; daemons log
       to stderr, so redirect via the service manager (for example systemd's
       journal) instead.

``[rlimits]``
-------------

POSIX ``RLIMIT_*`` values ``spurd`` applies to job steps at launch.

**Reload: Agent restart.**

.. list-table::
   :header-rows: 1
   :widths: 18 12 22 48

   * - Field
     - Type
     - Default
     - Description
   * - ``memlock``
     - string
     - ``"unlimited"``
     - ``RLIMIT_MEMLOCK`` for job processes. ``"unlimited"`` (also ``""`` or
       ``"0"``) sets ``RLIM_INFINITY``; ``"inherit"`` leaves whatever ``spurd``
       inherited; a byte-count string (e.g. ``"1073741824"`` for 1 GiB) sets a fixed
       cap. An invalid value errors at parse time.

.. note::

   ``memlock = "unlimited"`` lets RDMA and NCCL workloads pin memory out of the box.
   Lower it only when a hard cap is required.

``[mpi]``
---------

PMIx plugin settings for ``--mpi=pmix`` jobs (batch launch and ``srun`` steps).

Plugin loading happens on the node, so those fields need an agent restart; the
per-step directory and timeouts are sent by the controller with each dispatch and
are reloadable.

.. list-table::
   :header-rows: 1
   :widths: 26 10 18 16 30

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``plugin_dir``
     - string
     - ``"/usr/lib/spur"``
     - Agent restart
     - Directory searched for the PMIx plugin when ``pmix_plugin`` is unset.
   * - ``pmix_plugin``
     - string
     - ``""``
     - Agent restart
     - Explicit path to the PMIx plugin. When empty, the plugin resolves to
       ``<plugin_dir>/spur_mpi_pmix.so``.
   * - ``pmix_min_version``
     - string
     - ``"4.1.0"``
     - Agent restart
     - Minimum PMIx library version accepted when loading the plugin.
   * - ``pmix_tmpdir``
     - string
     - ``"/tmp/spur-pmix"``
     - Live
     - Base directory for per-step PMIx scratch (namespace and rank state).
   * - ``modex_connect_timeout_secs``
     - integer
     - ``5``
     - Live
     - Timeout for a step's initial connection to the PMIx modex.
   * - ``modex_fence_timeout_secs``
     - integer
     - ``120``
     - Live
     - Timeout for a collective fence across the step's ranks.
   * - ``modex_verify_timeout_secs``
     - integer
     - ``30``
     - Live
     - Timeout for post-fence modex verification.

.. note::

   ``plugin_dir`` defaults to ``/usr/lib/spur``. Tarball installs
   (``INSTALL_DIR=/opt/spur/bin``) put ``spur_mpi_pmix.so`` in
   ``/opt/spur/lib/spur`` — set ``plugin_dir`` to that directory or install the
   ``.so`` under the default path. Create ``pmix_tmpdir`` on every agent
   (``chmod 1777``). Ubuntu ``libpmix2t64`` needs ``PMIX_MCA_gds=hash`` on
   ``spurd`` and in ``$HOME/spur/mpi/env.sh``. Full MPI bring-up, including
   Open MPI TCP interface pinning on multi-NIC nodes, is in
   :doc:`/deployment/native-host`.

``[update]``
------------

Startup update checks and optional auto-download.

**Reload: Restart** for every field below — the controller configures its update
checker once at startup.

.. note::

   These fields apply to ``spurctld`` only. ``spurd`` runs its own startup update
   check with built-in defaults and does not read this section, so
   ``check_on_startup = false`` does not stop agents from checking.

.. list-table::
   :header-rows: 1
   :widths: 22 12 22 44

   * - Field
     - Type
     - Default
     - Description
   * - ``check_on_startup``
     - bool
     - ``true``
     - Check for updates on daemon startup.
   * - ``auto_update``
     - bool
     - ``false``
     - Automatically download and install updates.
   * - ``channel``
     - string
     - ``"stable"``
     - Release channel: ``"stable"`` or ``"nightly"``.
   * - ``cache_dir``
     - string
     - ``"/var/cache/spur"``
     - Directory for the update-check cache file.

.. note::

   Daemons never auto-restart, even with ``auto_update = true``. A downloaded update
   takes effect on the next manual restart.

``[admission]``
---------------

Controls which nodes may register with the controller.

**Reload: Live.**

.. list-table::
   :header-rows: 1
   :widths: 16 12 16 56

   * - Field
     - Type
     - Default
     - Description
   * - ``mode``
     - string
     - ``"open"``
     - Node admission mode. ``open`` lets any node register; ``token`` requires a
       registering ``spurd`` to present a valid admission token.

See :doc:`accounting` for managing admission tokens with ``spur token``.

``[devices]``
-------------

GPU and generic-resource discovery.

**Reload: Agent restart** for every field below, including each
``[[devices.gres]]`` entry — the device registry is built once when ``spurd``
starts.

.. list-table::
   :header-rows: 1
   :widths: 20 16 16 48

   * - Field
     - Type
     - Default
     - Description
   * - ``auto_detect``
     - bool
     - ``true``
     - Discover GPUs from AMD KFD sysfs when the CDI cache is empty (AMD only).
   * - ``cdi_spec_dirs``
     - [string]
     - ``[]``
     - Extra directories to scan for CDI specs, beyond ``/etc/cdi`` and
       ``/var/run/cdi``.
   * - ``gres``
     - [table]
     - ``[]``
     - File-based or countable GRES pools; see below.

Each ``[[devices.gres]]`` entry uses Slurm GRES syntax with fields ``name``
(required), ``type``, ``file``, ``multiple_files``, ``count``, ``cores``, ``links``,
and ``flags`` ([string]). Examples:

.. code-block:: toml

   [[devices.gres]]
   name = "gpu"
   file = "/dev/dri/renderD[128-129]"
   flags = ["amd_gpu_env"]

   [[devices.gres]]
   name = "bandwidth"
   type = "lustre"
   count = 4096
   flags = ["count_only"]

``[isolation]``
---------------

Job isolation layers.

.. warning::

   **Reload: Not implemented** for every field in this section. ``spurd`` does not
   read ``[isolation]``, so none of these values changes any behaviour — including
   setting one to ``false`` to disable a layer. Do not treat this section as a
   security control. The table below records the intended meaning of each field
   and how the corresponding behaviour is actually selected today.

.. list-table::
   :header-rows: 1
   :widths: 18 12 14 56

   * - Field
     - Type
     - Default
     - Intended meaning / actual behaviour
   * - ``setuid``
     - bool
     - ``true``
     - Run jobs as the submitting user's UID/GID. Always applied when ``spurd``
       runs as root; not gated by this field.
   * - ``namespaces``
     - bool
     - ``true``
     - PID and mount namespace isolation. Applied whenever ``spurd`` runs as
       root, except for multi-rank ``--mpi=pmix`` wrappers which stay in the host
       namespace; not gated by this field.
   * - ``seccomp``
     - bool
     - ``true``
     - seccomp-BPF syscall filter. Opt-in via the ``SPUR_SECCOMP=1``
       environment variable on ``spurd`` and **off** unless that is set.
   * - ``landlock``
     - bool
     - ``true``
     - Landlock filesystem access control. Actually opt-in via the
       ``SPUR_LANDLOCK=1`` environment variable on ``spurd`` and **off** unless
       that is set.

``[cgroup]``
------------

cgroup-v2 resource enforcement that ``spurd`` applies to native-host jobs. Every
process the agent starts for a job — the batch payload, ``srun`` steps, ``spur
exec``, and interactive attach — runs in a cgroup at
``/sys/fs/cgroup/spur/job_<id>``, and the limits are derived from the **per-node
budget the controller allocated** — not from the ``--cpus-per-task`` / ``--mem``
the user requested. Kubernetes jobs are unaffected: there the kubelet owns the
cgroups.

.. warning::

   These settings bound a job only when ``spurd`` runs as root, and with the
   default ``required = false`` a host that cannot apply a constraint — missing
   ``CAP_NET_ADMIN`` for the device filter, say — logs a warning and runs the work
   unconstrained. Set ``required = true`` to make that case fail closed instead.
   See :ref:`cgroup-containment-gaps` for what remains even then: per-step
   granularity, and the site-supplied task hooks that run outside the job cgroup.

.. note::

   Upgrading a cluster whose ``spur.conf`` has no ``[cgroup]`` section changes
   what gets enforced — see :ref:`cgroup-upgrade-notes`.

.. list-table::
   :header-rows: 1
   :widths: 20 8 10 14 48

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``enabled``
     - bool
     - ``true``
     - Agent restart
     - Master switch. When ``false``, no cgroup is created and no limit is
       applied.
   * - ``required``
     - bool
     - ``false``
     - Agent restart
     - Refuse the work when a requested constraint cannot be applied, instead of
       warning and running it unconstrained. This gates a batch launch, the
       registration of an interactive allocation (``salloc``, standalone
       ``srun``), and a step that did not end up in its job's cgroup. See the
       note below.
   * - ``constrain_cores``
     - bool
     - ``true``
     - Agent restart
     - Pin the job to its allocated cores via ``cpuset.cpus``. With
       ``cpu_quota`` off this is the only CPU bound, so a job whose allocation
       yields no cores runs CPU-unconstrained and ``spurd`` logs a warning.
   * - ``cpu_quota``
     - bool
     - ``false``
     - Agent restart
     - Additionally cap CPU time with a CFS quota (``cpu.max``). Off because the
       cpuset already bounds whole-core allocations; enabling it adds a hard
       throttle on top.
   * - ``constrain_ram_space``
     - bool
     - ``true``
     - Agent restart
     - Cap memory via ``memory.max`` and ``memory.high``. Costs some per-node job
       throughput — see the note below.
   * - ``allowed_ram_percent``
     - int
     - ``100``
     - Agent restart
     - Hard ceiling (``memory.max``) as a percentage of the allocated memory.
       ``memory.high`` stays at 100% of the allocation, so the default makes the
       two equal and a value above 100 opens a soft-throttle band. Must be at
       least 1. Ignored when ``constrain_ram_space`` is ``false``, where it
       reverts to 100%.
   * - ``constrain_swap``
     - bool
     - ``false``
     - Agent restart
     - Bound swap via ``memory.swap.max``. Off by default, as in Slurm: while
       off, ``memory.max`` bounds resident memory only and a job that outgrows
       ``--mem`` swaps instead of being killed. Turning it on with
       ``constrain_ram_space`` off still bounds memory — see below.
   * - ``allowed_swap_percent``
     - int
     - ``0``
     - Agent restart
     - Swap allowance as a percentage of the allocated memory. ``0`` means no
       swap. Not capped at 100 — a swap-rich node may grant more swap than RAM,
       and Slurm places no upper bound on ``AllowedSwapSpace`` either.
   * - ``min_ram_mb``
     - int
     - ``30``
     - Agent restart
     - Floor for the memory ceilings, in MiB. Guards against a tiny ``--mem``
       creating a cgroup so small the job dies during its own startup.
   * - ``oom_kill_job``
     - bool
     - ``true``
     - Agent restart
     - On OOM, kill every process in the job (``memory.oom.group``) instead of
       letting the kernel pick one.
   * - ``constrain_devices``
     - bool
     - ``true``
     - Agent restart
     - Restrict the job to the device nodes its allocation granted, using a
       cgroup-v2 BPF device filter. Default-deny: a job allocated no GPUs can open
       only the nodes listed under :ref:`device-filter-implicit-allow` below, and
       opening an unallocated GPU fails with ``EPERM`` whatever the job sets
       ``ROCR_VISIBLE_DEVICES`` to. Installing the filter needs ``CAP_BPF`` and
       ``CAP_NET_ADMIN`` (or ``CAP_SYS_ADMIN``) — ``CAP_BPF`` alone is insufficient for a
       cgroup-device program; without them the job runs with no device isolation.
   * - ``extra_device_paths``
     - [string]
     - ``[]``
     - Agent restart
     - Additional device node paths **every** job on this node may open, on top of
       its allocation and the implicit set below. The escape hatch for a site
       device the defaults miss, short of turning the filter off. A path that is
       not a device node is ignored, so an entry for hardware this node lacks is
       harmless. Read once at agent startup, like the rest of ``[cgroup]``.

A job submitted without ``--mem`` has no memory budget, so ``memory.max``,
``memory.high``, and ``memory.swap.max`` are all left at the kernel default.
Constraining swap to zero while memory stayed unlimited would be incoherent.

The two ``constrain_*`` memory switches interact, matching Slurm. Constraining
swap on its own does **not** leave RAM unlimited: the RAM ceiling becomes the
combined RAM+swap total, so the sum a job can reach stays bounded.

.. list-table::
   :header-rows: 1
   :widths: 18 16 22 22 22

   * - ``constrain_ram_space``
     - ``constrain_swap``
     - ``memory.max``
     - ``memory.high``
     - ``memory.swap.max``
   * - ``false``
     - ``false``
     - unset
     - unset
     - unset
   * - ``true``
     - ``false``
     - ``allowed_ram_percent``\ % of the allocation
     - the allocation
     - unset
   * - ``false``
     - ``true``
     - allocation + ``allowed_swap_percent``\ %
     - the allocation
     - ``0``
   * - ``true``
     - ``true``
     - ``allowed_ram_percent``\ % of the allocation
     - the allocation
     - ``allowed_swap_percent``\ % of the allocation

Every ceiling is floored at ``min_ram_mb``, and ``memory.high`` never exceeds
``memory.max``.

.. _device-filter-implicit-allow:

What the device filter allows without an allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A job's allow-list is its allocated device nodes plus two fixed sets. Both are
granted to every job, including one that requested no GPUs, so they are part of the
isolation boundary and worth knowing:

- **Base pseudo-devices**, which virtually every program needs: ``/dev/null``,
  ``/dev/zero``, ``/dev/full``, ``/dev/random``, ``/dev/urandom``, ``/dev/tty``,
  ``/dev/console``, ``/dev/ptmx`` and ``/dev/pts/*``.
- **Host infrastructure** — device nodes shared by the whole node rather than handed
  out per job, so no allocation can ever grant them: ``/dev/fuse`` (used by
  Apptainer and Singularity, which run *inside* the batch job), everything under
  ``/dev/infiniband/`` (RDMA verbs, for MPI and for NCCL or RCCL over InfiniBand),
  and the MIG capability nodes under ``/dev/nvidia-caps/``. Every one is resolved by
  path when the filter is built, so a node without that hardware grants nothing extra.

The vendor control nodes a GPU runtime initializes through — ``/dev/nvidiactl``,
``/dev/nvidia-uvm`` and the like — are **not** in this set. They reach a job through
its allocation's CDI device edits, the same path as the per-GPU compute nodes, so a
job with no GPU never gets them. A node configured through GRES rather than a CDI
spec does not enumerate them as devices, so name them in ``extra_device_paths`` there.

**The per-GPU compute nodes are not in that set.** ``/dev/nvidia<N>``, ``/dev/kfd``
and the ``/dev/dri`` render and card nodes reach a job only through its allocation —
that gating is the entire point of the filter. Putting one of them in
``extra_device_paths`` hands every job on the node a GPU.

When enforcement fails
~~~~~~~~~~~~~~~~~~~~~~

By default every step degrades to a warning: a controller that could not be
delegated, a rejected control-file write, a cpuset that did not apply, a device
filter that could not be attached, or a process that could not join the cgroup
all leave the job **running unconstrained**. Grep the
agent log for these to find silently-unenforced nodes:

.. code-block:: text

   failed to delegate cgroup controller
   failed to write cgroup control file
   cpuset not applied; job runs without a CPU bound
   device filter not installed; job runs without device isolation
   allocation registered without cgroup enforcement
   failed to join cgroup; job runs without resource limits

Set ``required = true`` to refuse the work instead. A batch launch fails, an
interactive allocation is refused at registration (so ``salloc`` and standalone
``srun`` error out rather than handing back an unenforceable allocation), and a
step that did not join its job's cgroup is killed and refused. Each fails rather
than running outside its limits, which is the right trade on a shared node where
the limits are the isolation boundary. It also means a host that cannot enforce
stops accepting work, so roll it out only once the agent runs as root and the
cgroup tree is confirmed writable.

.. note::

   **Memory constraints cost throughput.** ``memory.high`` makes the kernel
   reclaim against a job approaching its ceiling rather than failing the
   allocation outright. At the default ``allowed_ram_percent = 100`` that
   pressure starts at the same point the OOM kill would, so a job that overruns
   its budget slows down before it dies. Sites that would rather have headroom
   than reclaim stalls should raise ``allowed_ram_percent`` (e.g. ``125``) rather
   than turn ``constrain_ram_space`` off.

Verify what a running job actually got:

.. code-block:: bash

   cat /sys/fs/cgroup/spur/job_1234/cpuset.cpus       # allocated cores
   cat /sys/fs/cgroup/spur/job_1234/memory.max        # hard ceiling, bytes
   cat /sys/fs/cgroup/spur/job_1234/memory.high       # reclaim threshold
   cat /sys/fs/cgroup/spur/job_1234/memory.swap.max   # swap ceiling
   bpftool cgroup show /sys/fs/cgroup/spur/job_1234   # attached device filter

Migrating from ``cgroup.conf``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The memory ceilings are computed exactly as Slurm's ``cgroup/v2`` plugin computes
them, so a site's existing percentages carry over directly.

.. list-table::
   :header-rows: 1
   :widths: 34 34 32

   * - Slurm ``cgroup.conf``
     - Spur ``[cgroup]``
     - Notes
   * - ``ConstrainCores``
     - ``constrain_cores``
     -
   * - ``ConstrainDevices``
     - ``constrain_devices``
     - Both deny by ``major:minor``. Slurm takes the allow-list from
       ``gres.conf``; Spur takes it from the device registry's injection plan,
       plus the implicit set above and ``extra_device_paths``.
   * - ``ConstrainRAMSpace``
     - ``constrain_ram_space``
     -
   * - ``AllowedRAMSpace``
     - ``allowed_ram_percent``
     - Integer only; Slurm accepts ``101.5``.
   * - ``ConstrainSwapSpace``
     - ``constrain_swap``
     -
   * - ``AllowedSwapSpace``
     - ``allowed_swap_percent``
     - Integer only. Values above 100 carry over unchanged.
   * - ``MinRAMSpace``
     - ``min_ram_mb``
     -
   * - ``OOMKillStep``
     - ``oom_kill_job``
     - Applies to the whole job; Spur has no per-step cgroups yet.

Deliberate differences from Slurm:

- **Spur constrains cores, RAM, and devices by default.** Every Slurm
  ``Constrain*`` defaults to ``no``, so a job that overran ``--mem`` under a stock
  Slurm configuration will be reclaimed against, or killed, on Spur, and one that
  reached a GPU it was not allocated now gets ``EPERM``. Swap is the exception and
  stays off, matching Slurm: bounding it turns a job that completed slowly into
  one that is killed outright.
- **``allowed_ram_percent = 0`` is rejected**, where Slurm would accept it and
  floor every job at ``MinRAMSpace``. That silently caps a whole cluster at
  30 MiB per job, so Spur treats it as the typo it almost certainly is. Every
  other percentage carries over from ``cgroup.conf`` unchanged.
- **Whole-job OOM kill by default.** Slurm's default kills one process and lets
  the step keep running; set ``oom_kill_job = false`` for that behaviour.
- **No CFS quota.** Slurm never writes ``cpu.max``; neither does Spur by default.
  ``cpu_quota`` exists for sites that want one.
- **No ``--mem`` means unbounded.** Slurm substitutes the node's configured
  ``RealMemory``; Spur has no such per-node figure and leaves the ceilings unset.
- **No plugin selectors.** ``CgroupPlugin``, ``TaskPlugin``, and ``ProctrackType``
  have no Spur equivalent — the mechanism is not pluggable.
- **Raising ``allowed_ram_percent`` above 100 over-commits the node** and can
  trigger a system-wide OOM, the same warning Slurm gives for ``AllowedRAMSpace``.

``[metrics]``
-------------

OpenMetrics HTTP export from ``spurctld``.

.. list-table::
   :header-rows: 1
   :widths: 22 10 18 14 36

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``enabled``
     - bool
     - ``true``
     - Restart
     - Start the metrics HTTP server.
   * - ``listen_addr``
     - string
     - ``"[::]:6822"``
     - Restart
     - Metrics HTTP listen address; the port is used when ``bind = "loopback"``.
   * - ``bind``
     - string
     - ``"loopback"``
     - Restart
     - ``loopback`` binds ``127.0.0.1:<port>``; ``all`` uses ``listen_addr`` as-is.
   * - ``high_cardinality``
     - bool
     - ``false``
     - Live
     - Serve the per-job/user/account metrics route. While ``false`` that route
       returns 404. High cardinality on a busy cluster; enable deliberately.

``[rest_api]``
--------------

.. list-table::
   :header-rows: 1
   :widths: 16 10 14 14 46

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - Restart
     - Start the Slurm-compatible REST server (default port 6820). Off by
       default: the REST surface performs no authentication, so enabling it on a
       reachable address exposes unauthenticated job submission. Enable it only
       behind an authenticating proxy or on a loopback interface.

``[hooks]``
-----------

Prolog/epilog and job-submit scripts. Each field is an optional fully-qualified
path; unset means no hook. The prolog/epilog fields map one-to-one to Slurm's
parameters.

Reload scope follows whichever process executes the hook: controller hooks are
live, node hooks need an agent restart, and ``srun`` hooks are read from the
submitting host on each invocation.

.. list-table::
   :header-rows: 1
   :widths: 22 24 32 22

   * - Spur field
     - Slurm equivalent
     - Runs on
     - Reload
   * - ``prolog``
     - ``Prolog``
     - compute node, before job launch
     - Agent restart
   * - ``epilog``
     - ``Epilog``
     - compute node, at job termination
     - Agent restart
   * - ``prolog_slurmctld``
     - ``PrologSlurmctld``
     - controller, at allocation
     - Live
   * - ``epilog_slurmctld``
     - ``EpilogSlurmctld``
     - controller, at termination
     - Live
   * - ``task_prolog``
     - ``TaskProlog``
     - compute node, before each step
     - Agent restart
   * - ``task_epilog``
     - ``TaskEpilog``
     - compute node, after each step
     - Agent restart
   * - ``srun_prolog``
     - ``SrunProlog``
     - srun node, before step dispatch
     - Client
   * - ``srun_epilog``
     - ``SrunEpilog``
     - srun node, after step completion
     - Client
   * - ``job_submit``
     - ``JobSubmitPlugin``
     - controller, at submit
     - Live
   * - ``job_submit_lua``
     - ``job_submit.lua``
     - controller, at submit
     - Live

.. note::

   ``reconfigure`` validates ``job_submit`` and ``job_submit_lua`` before swapping
   the config, so a broken submit hook is rejected rather than applied — the
   previous configuration stays in place and the command reports an error.

``[notifications]``
-------------------

Job-event notification transports.

**Reload: Live** for every field below.

.. list-table::
   :header-rows: 1
   :widths: 22 12 16 50

   * - Field
     - Type
     - Default
     - Description
   * - ``webhook_url``
     - string
     - none
     - URL to POST job-event notifications to.
   * - ``smtp_command``
     - string
     - none
     - SMTP command for mail, e.g. ``"/usr/sbin/sendmail -t"``.
   * - ``from_address``
     - string
     - none
     - From address, e.g. ``"spur@cluster.local"``.

``[power]``
-----------

Idle-node suspend and resume.

.. list-table::
   :header-rows: 1
   :widths: 26 10 14 14 36

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``suspend_timeout_secs``
     - integer
     - none
     - Restart
     - Idle seconds before a node is suspended. Unset disables power management
       entirely. Read once when the power-management loop starts, so
       ``reconfigure`` can neither enable, disable, nor retime it.
   * - ``suspend_command``
     - string
     - none
     - Live
     - Suspend command; ``{node}`` is replaced with the node name, e.g.
       ``"systemctl suspend"``.
   * - ``resume_command``
     - string
     - none
     - Live
     - Resume command; ``{node}`` is replaced, e.g. ``"ipmitool chassis power on"``.

.. note::

   Because ``suspend_timeout_secs`` is restart-only, turning power management on
   for the first time requires a controller restart. Once running, the suspend and
   resume commands can be changed live.

Kubernetes modes
----------------

Spur has two distinct, mutually exclusive Kubernetes modes. ``[kubernetes]`` lets
Spur run **inside** an existing cluster and accept ``SpurJob`` CRDs; ``[cluster]``
lets Spur **own** and provision a k0s cluster.

``[kubernetes]``
~~~~~~~~~~~~~~~~

.. warning::

   **Reload: Not implemented** for every field in this section. The
   ``spur-k8s-operator`` binary does not read ``spur.conf`` at all; it takes its
   node selector from the ``--node-selector`` flag, its namespace from the
   ``POD_NAMESPACE`` environment variable, and its credentials from the ambient
   kubeconfig or in-cluster service account. Configure the operator through its
   deployment manifest, not here.

.. list-table::
   :header-rows: 1
   :widths: 26 12 30 32

   * - Field
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - Enable K8s integration (accept ``SpurJob`` CRDs).
   * - ``kubeconfig``
     - string
     - none
     - Path to a kubeconfig; empty uses in-cluster config.
   * - ``namespace``
     - string
     - ``"spur"``
     - Namespace for ``SpurJob`` CRDs and Pods.
   * - ``node_label_selector``
     - string
     - ``"spur.amd.com/managed=true"``
     - Label selector for nodes in the Spur pool.

``[cluster]``
~~~~~~~~~~~~~

Spur-managed k0s cluster. When disabled (the default), ``spurd`` never touches
systemd or k0s.

This section is split across both daemons: the controller reads the network and
control-plane fields at startup, while ``spurd`` reads the on-node fields at its
own startup. Only ``allow_admin_kubeconfig`` is reloadable.

.. list-table::
   :header-rows: 1
   :widths: 26 10 20 16 28

   * - Field
     - Type
     - Default
     - Reload
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - Restart
     - Enable the Spur-managed k0s cluster. Requires restarting the controller
       **and** every agent.
   * - ``distro``
     - string
     - ``"k0s"``
     - Not implemented
     - Intended to select a Kubernetes distribution. Validated as ``"k0s"`` on
       load but otherwise unused; k0s is always the distribution.
   * - ``pod_cidr``
     - string
     - ``"10.42.0.0/16"``
     - Restart
     - Pod network CIDR. Prefix must be ``<= /24`` (per-node /24 carving).
   * - ``service_cidr``
     - string
     - ``"10.43.0.0/16"``
     - Restart
     - Service network CIDR.
   * - ``cni``
     - string
     - ``"kuberouter"``
     - Restart
     - CNI mode: ``"kuberouter"`` (k0s default) or ``"calico"`` (bird native routing
       over the mesh).
   * - ``cni_mtu``
     - integer
     - ``1450``
     - Restart
     - CNI MTU, leaving headroom for WireGuard overhead.
   * - ``control_plane_node``
     - string
     - none
     - Restart
     - Hostname running the k0s control plane; empty picks one from inventory.
   * - ``control_plane_replicas``
     - integer
     - ``1``
     - Restart
     - Number of control-plane members to provision.
   * - ``k8s_provisioning_timeout_secs``
     - integer
     - ``600``
     - Restart
     - How long a node may stay in provisioning before it is marked Degraded.
   * - ``allow_admin_kubeconfig``
     - bool
     - ``false``
     - Live
     - Allow the controller to hand out a cluster-admin kubeconfig. Reloadable,
       so it can be turned off without a restart.
   * - ``k0s_version``
     - string
     - pinned
     - Agent restart
     - k0s release the agent installs.
   * - ``k0s_binary``
     - string
     - built-in path
     - Agent restart
     - Path to the ``k0s`` binary on the node.
   * - ``storage_provisioner``
     - string
     - ``"local-path"``
     - Agent restart
     - ``"local-path"`` ships a default node-local StorageClass; ``"none"`` disables
       it. Other values are rejected.
   * - ``local_path_dir``
     - string
     - ``/var/lib/local-path-provisioner``
     - Agent restart
     - On-node directory for local-path PVs. Must be absolute and free of quotes,
       backslashes, whitespace, and control characters.

See :doc:`/deployment/managed-kubernetes` for provisioning a Spur-owned cluster.

``[federation]``, ``[topology]``, ``[burst_buffer]``
----------------------------------------------------

``[federation]``
   **Reload: Live.** Peer clusters for cross-cluster job routing. Each
   ``[[federation.clusters]]`` entry has ``name`` (string) and ``address``
   (string, e.g. ``"http://peer-ctrl:6817"``). Defaults to no peers.

``[topology]``
   **Reload: Restart.** Optional switch-hierarchy configuration for locality-aware
   scheduling.
   ``plugin`` (string, default ``"none"``) selects the model: ``"tree"`` for a
   switch hierarchy, ``"block"`` for fixed-size blocks, or ``"none"`` to disable.
   In tree mode, each ``[[topology.switches]]`` entry has ``name`` (string),
   ``nodes`` (hostlist pattern for a leaf switch), and ``switches`` (comma-separated
   child switch names for an aggregation switch). In block mode, ``block_size``
   (integer) sets the number of nodes per block. Defaults to no topology.

``[burst_buffer]``
   **Reload: Live.** Burst-buffer capacity. ``total_gb`` (integer, default ``0``)
   sets total capacity
   in GiB; jobs reserve via ``--bb capacity=NNN``. ``0`` disables burst buffers, and
   requesting jobs stay pending with ``BurstBufferResources``.

Validation
----------

The controller validates ``spur.conf`` on load and refuses to start on error:

- ``cluster_name`` must be non-empty.
- ``controller.max_batch_requeue`` must be ``>= 1``.
- When ``[cluster]`` is enabled:

  - ``distro`` must be ``"k0s"``.
  - ``network.wg_cidr``, ``cluster.pod_cidr``, and ``cluster.service_cidr`` must be
    valid IPv4 CIDRs, and ``pod_cidr`` must be ``<= /24``.
  - The three CIDRs must not overlap.
  - ``storage_provisioner`` must be ``local-path`` or ``none``.
  - ``local_path_dir`` must be absolute and clean when the local-path provisioner is
    used.

Environment overrides
---------------------

.. note::

   Config-file fields are **not** overridable by environment variables.
   ``SPUR_CONTROLLER_ADDR`` is a CLI-level override that sets the controller address
   for client commands (``sacctmgr``, ``scontrol``, ``spur token``); it does not
   affect any ``spur.conf`` field.

See Also
--------

- :doc:`accounting`
- :doc:`/deployment/ansible`
- :doc:`/deployment/native-host`
