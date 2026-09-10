// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use spur_core::config::SlurmConfig;

pub fn load_spur_config() -> SlurmConfig {
    let path_str = std::env::var("SPUR_CONF").unwrap_or_else(|_| "/etc/spur/spur.conf".to_string());
    let path = std::path::Path::new(&path_str);
    match SlurmConfig::load_from_file(path) {
        Ok(config) => config,
        Err(_) => SlurmConfig {
            cluster_name: "spur".into(),
            controller: Default::default(),
            accounting: Default::default(),
            scheduler: Default::default(),
            auth: Default::default(),
            partitions: Vec::new(),
            nodes: Vec::new(),
            network: Default::default(),
            logging: Default::default(),
            kubernetes: Default::default(),
            cluster: Default::default(),
            notifications: Default::default(),
            power: Default::default(),
            federation: Default::default(),
            topology: None,
            isolation: Default::default(),
            licenses: Default::default(),
            burst_buffer: Default::default(),
            update: Default::default(),
            probes: Default::default(),
            metrics: Default::default(),
            rest_api: Default::default(),
            hooks: Default::default(),
            devices: Default::default(),
            admission: Default::default(),
            rlimits: Default::default(),
            cgroup: Default::default(),
            mpi: Default::default(),
        },
    }
}
