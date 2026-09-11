Running Jobs in Containers
==========================

Spur can run a job inside an OCI container image, backed by a squashfs snapshot
of the image. You import an image once with ``spur image``, then pass container
flags to ``spur submit`` (Slurm ``sbatch``) or ``spur run`` (Slurm ``srun``) to
run your job inside it. This page covers importing images, running container
jobs, and executing extra commands inside a running one.

Importing Images
----------------

``spur image import <ref>`` fetches an image and packs it into a zstd squashfs
``.sqsh`` file (this needs ``squashfs-tools`` installed). The reference can be a
registry image or a local container source:

.. code-block:: bash

   spur image import ubuntu:22.04                 # registry
   spur image import registry.example.com/pytorch:latest # registry with host
   spur image import docker://myapp:latest        # registry via docker transport
   spur image import dockerd://myapp:latest        # local Docker daemon
   spur image import podman://myapp:latest         # local Podman storage

List, export, and remove imported images:

.. code-block:: bash

   spur image list
   spur image export mycontainer -o /tmp/mycontainer.sqsh
   spur image remove mycontainer

Images are stored in the first of these directories that is usable:
``$SPUR_IMAGE_DIR``, then ``/var/spool/spur/images``, then ``~/.spur/images``.
At submit time Spur resolves a bare image name (e.g. ``busybox``) to an absolute
``.sqsh`` path in these directories when it finds one — which works when the
login and compute nodes share a filesystem. Otherwise the name is passed through
for the node agent to resolve.

Running a Container Job
-----------------------

Add container flags to ``sbatch`` or ``srun``. The job's script (or command)
runs inside the container.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Flag
     - Description
   * - ``--container-image <ref|/path.sqsh>``
     - Image to run in. A registry reference, a stored name, or a ``.sqsh`` path.
   * - ``--container-mounts <src>:<dst>[:ro]``
     - Bind-mount a host path into the container. Repeatable; append ``:ro`` for
       read-only.
   * - ``--container-workdir <dir>``
     - Working directory inside the container.
   * - ``--container-name <name>``
     - Persist the container across jobs. Required for ``spur image export``.
   * - ``--container-readonly``
     - Not yet implemented — rejected at submission. (A read-only container
       root is not enforced yet; the flag is refused rather than silently
       ignored.)
   * - ``--container-mount-home``
     - Mount your home directory into the container.
   * - ``--container-env KEY=VAL``
     - Set an environment variable inside the container. Repeatable.
   * - ``--container-entrypoint <cmd>``
     - Shell command run inside the container immediately before the job script
       (``<cmd> && <script>``). It does not replace the image's ENTRYPOINT.
   * - ``--container-remap-root``
     - Map the job user to root inside the container.

A GPU training job with a read-only data mount:

.. code-block:: bash

   sbatch --container-image registry.example.com/pytorch:latest \
          --container-mounts /data:/data:ro \
          --gres=gpu:8 train.sh

An interactive shell in a container with your home directory mounted:

.. code-block:: bash

   srun --container-image ubuntu:22.04 --container-mount-home bash

Environment Inside the Container
--------------------------------

A container job starts from the image's own environment — the ``config.Env`` that
``docker run`` would apply — and layers the job's environment on top of it. Spur
records the image config at ``spur image import`` time, so images imported by an
older Spur have nothing recorded and keep starting from the job environment
alone; re-import them to pick up their environment.

Later entries win:

1. the image's ``config.Env``
2. the job environment (``--export``, which defaults to ``ALL``)
3. ``--container-env KEY=VAL``
4. admin ``environ.d`` files and hook variables

``PATH`` and ``LD_LIBRARY_PATH`` are the exception: instead of being replaced,
the image's entries and the job's are joined, image first, so that the programs
and libraries shipped in the image stay reachable while additions from the
submitting host still apply. An explicit ``--container-env PATH=...`` replaces
the result outright.

This is what lets an image's own tooling run without spelling out its paths:

.. code-block:: bash

   # torchrun lives on the image's PATH (/opt/venv/bin), not the host's
   sbatch --container-image registry.example.com/pytorch:latest train.sh

Containerized Job Steps
-----------------------

``srun`` steps run inside a container too, not just batch jobs. The container
flags above apply to a step's ``srun`` and behave as follows:

- **Step inside a containerized allocation.** When the batch job was submitted
  with ``--container-image`` (``sbatch --container-image ...``), a nested
  ``srun`` step enters that already-running container — it shares the job's
  image, mounts, and GPU devices. This is the standard Slurm + Pyxis pattern
  used by multi-node launchers.

- **Step with its own image.** A standalone ``srun --container-image`` (or an
  ``salloc`` allocation followed by ``srun --container-image``) creates a fresh
  container for that step. Different steps in one allocation can use different
  images and mounts.

Allocate first, then run steps from inside the allocation shell. By default a
single step fans out one container per allocated node; ``-N`` or ``-w`` limits
container creation or entry to the selected step nodes:

.. code-block:: bash

   salloc -N2 --gpus-per-node=8
   # ...now inside the allocation shell:
   srun --container-image trainer.sqsh <command>   # one container per node
   srun -N1 --container-image trainer.sqsh <command>   # one allocated node
   srun -w node002 --container-image trainer.sqsh <command>   # exact node

.. note::

   ``-N`` and ``-w`` only choose which allocated nodes run the step. They do not
   resize the job-level CPU, memory, or GPU cgroup on those nodes.

GPU allocation, bind mounts, and cancellation apply per step: a cancelled step
tears down its container and cleans up its rootfs without affecting the rest of
the allocation.

Interactive steps (``srun --pty``) run inside the container too — you get a real
terminal inside the container rather than on the host. ``--pty`` still ignores
``--nodes`` and uses only the first ``-w`` name. The step uses its own
``--container-image`` when given one, otherwise a nested ``srun --pty`` inherits
the image from its containerized ``sbatch`` job:

.. code-block:: bash

   salloc -N1 --gpus-per-node=8
   # ...now inside the allocation shell:
   srun --pty --container-image trainer.sqsh bash   # interactive shell in the container

   # or attach an interactive shell to a running containerized job
   # (enters its container via nsenter):
   sbatch -N1 --container-image trainer.sqsh long_job.sh
   srun --jobid=<id> --overlap --pty bash            # same container, interactive

A ``--pty`` step nested inside a containerized ``sbatch`` job enters
the parent's running container (like any other nested step) rather than building
a fresh one.

Exec Into a Running Container Job
---------------------------------

``spur exec <job_id> <command...>`` runs a one-shot command inside a job's
running container. Output is buffered — this is not an interactive terminal.

.. code-block:: bash

   spur exec 1024 rocm-smi
   spur exec 1024 -- ls -la /workspace

.. tip::

   For an interactive session inside a running job, use ``srun --jobid <id>
   --overlap`` or :doc:`spur attach <interactive>` instead — ``spur exec`` has no
   TTY mode.

See Also
--------

- :doc:`submitting-jobs`
- :doc:`interactive`
