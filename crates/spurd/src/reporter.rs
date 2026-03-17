use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Context;
use spur_core::resource::{GpuLinkType, GpuResource, ResourceSet};
use spur_proto::proto::slurm_controller_client::SlurmControllerClient;
use spur_proto::proto::{RegisterAgentRequest, ResourceSet as ProtoResourceSet};
use tracing::{debug, info, warn};

use crate::gpu;

/// Discovers and reports node resources to the controller.
pub struct NodeReporter {
    pub hostname: String,
    pub controller_addr: String,
    pub resources: ResourceSet,
    pub free_memory_mb: AtomicU64,
    pub cpu_load: AtomicU64,
}

impl NodeReporter {
    pub fn new(hostname: String, controller_addr: String, resources: ResourceSet) -> Self {
        Self {
            hostname,
            controller_addr,
            resources,
            free_memory_mb: AtomicU64::new(0),
            cpu_load: AtomicU64::new(0),
        }
    }

    /// Register with the controller.
    pub async fn register(&self) -> anyhow::Result<()> {
        let mut client = SlurmControllerClient::connect(self.controller_addr.clone())
            .await
            .context("failed to connect to spurctld for registration")?;

        let resp = client
            .register_agent(RegisterAgentRequest {
                hostname: self.hostname.clone(),
                resources: Some(resource_to_proto(&self.resources)),
                version: env!("CARGO_PKG_VERSION").into(),
            })
            .await
            .context("registration failed")?;

        let inner = resp.into_inner();
        if inner.accepted {
            info!("registered with controller");
        } else {
            anyhow::bail!("controller rejected registration: {}", inner.message);
        }

        Ok(())
    }

    /// Periodic heartbeat loop.
    pub async fn heartbeat_loop(&self) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            interval.tick().await;

            // Update system metrics
            let (load, free_mem) = read_system_metrics();
            self.cpu_load.store(load as u64, Ordering::Relaxed);
            self.free_memory_mb.store(free_mem, Ordering::Relaxed);

            // Send heartbeat via unary RPC (streaming deferred to Phase 5)
            match SlurmControllerClient::connect(self.controller_addr.clone()).await {
                Ok(mut client) => {
                    // Use GetNode as a lightweight heartbeat for now
                    // Full streaming heartbeat in Phase 5
                    match client
                        .get_node(spur_proto::proto::GetNodeRequest {
                            name: self.hostname.clone(),
                        })
                        .await
                    {
                        Ok(_) => debug!(
                            load,
                            free_mem,
                            "heartbeat sent"
                        ),
                        Err(e) => warn!(error = %e, "heartbeat failed"),
                    }
                }
                Err(e) => warn!(error = %e, "heartbeat connection failed"),
            }
        }
    }
}

/// Discover local node resources from sysfs / /proc.
pub fn discover_resources() -> ResourceSet {
    let cpus = discover_cpus();
    let memory_mb = discover_memory_mb();
    let gpus = gpu::discover_gpus();

    ResourceSet {
        cpus,
        memory_mb,
        gpus,
        ..Default::default()
    }
}

/// Count online CPUs from sysfs.
fn discover_cpus() -> u32 {
    // Try /sys/devices/system/cpu/online first
    if let Ok(online) = std::fs::read_to_string("/sys/devices/system/cpu/online") {
        if let Some(count) = parse_cpu_range(&online.trim()) {
            return count;
        }
    }

    // Fallback: count /proc/cpuinfo processors
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        return cpuinfo
            .lines()
            .filter(|l| l.starts_with("processor"))
            .count() as u32;
    }

    // Last resort
    num_cpus()
}

/// Parse "0-191" or "0-63,128-191" into a total count.
fn parse_cpu_range(s: &str) -> Option<u32> {
    let mut count = 0u32;
    for part in s.split(',') {
        let part = part.trim();
        if let Some((start_s, end_s)) = part.split_once('-') {
            let start: u32 = start_s.parse().ok()?;
            let end: u32 = end_s.parse().ok()?;
            count += end - start + 1;
        } else {
            let _: u32 = part.parse().ok()?;
            count += 1;
        }
    }
    Some(count)
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

/// Read total memory from /proc/meminfo.
fn discover_memory_mb() -> u64 {
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let rest = rest.trim();
                if let Some(kb_str) = rest.strip_suffix("kB") {
                    if let Ok(kb) = kb_str.trim().parse::<u64>() {
                        return kb / 1024;
                    }
                }
            }
        }
    }
    0
}

/// Read current load average and free memory.
fn read_system_metrics() -> (u32, u64) {
    let load = read_load_avg();
    let free_mem = read_free_memory_mb();
    (load, free_mem)
}

fn read_load_avg() -> u32 {
    if let Ok(loadavg) = std::fs::read_to_string("/proc/loadavg") {
        if let Some(first) = loadavg.split_whitespace().next() {
            if let Ok(load) = first.parse::<f64>() {
                return (load * 100.0) as u32;
            }
        }
    }
    0
}

fn read_free_memory_mb() -> u64 {
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let rest = rest.trim();
                if let Some(kb_str) = rest.strip_suffix("kB") {
                    if let Ok(kb) = kb_str.trim().parse::<u64>() {
                        return kb / 1024;
                    }
                }
            }
        }
    }
    0
}

pub fn resource_to_proto(r: &ResourceSet) -> ProtoResourceSet {
    ProtoResourceSet {
        cpus: r.cpus,
        memory_mb: r.memory_mb,
        gpus: r
            .gpus
            .iter()
            .map(|g| spur_proto::proto::GpuResource {
                device_id: g.device_id,
                gpu_type: g.gpu_type.clone(),
                memory_mb: g.memory_mb,
                peer_gpus: g.peer_gpus.clone(),
                link_type: match g.link_type {
                    GpuLinkType::XGMI => spur_proto::proto::GpuLinkType::GpuLinkXgmi as i32,
                    GpuLinkType::NVLink => spur_proto::proto::GpuLinkType::GpuLinkNvlink as i32,
                    GpuLinkType::PCIe => spur_proto::proto::GpuLinkType::GpuLinkPcie as i32,
                },
            })
            .collect(),
        generic: r.generic.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpu_range() {
        assert_eq!(parse_cpu_range("0-191"), Some(192));
        assert_eq!(parse_cpu_range("0-63,128-191"), Some(128));
        assert_eq!(parse_cpu_range("0"), Some(1));
        assert_eq!(parse_cpu_range("0-3"), Some(4));
    }

    #[test]
    fn test_discover_cpus() {
        let cpus = discover_cpus();
        assert!(cpus > 0);
    }

    #[test]
    fn test_discover_memory() {
        let mem = discover_memory_mb();
        assert!(mem > 0);
    }
}
