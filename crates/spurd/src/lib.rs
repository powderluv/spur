// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Library half of the `spurd` node agent. Split out so `spur-stepd` (the
//! per-job supervisor binary) can reuse `executor`/`container`/`stepd`
//! without linking spurd's controller-RPC, k0s, or mesh-networking surface.

pub mod agent_server;
pub mod auth_middleware;
pub mod cluster;
pub mod container;
pub(crate) mod device_cgroup;
pub mod executor;
pub mod job_entry;
pub(crate) mod job_lifecycle;
pub mod landlock;
pub mod mpi_plugin;
pub mod privdrop;
pub mod pty;
pub mod reporter;
pub mod seccomp;
pub mod stepd;
