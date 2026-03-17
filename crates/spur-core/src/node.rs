use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::resource::ResourceSet;

/// Node states matching Slurm's model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NodeState {
    Idle,
    Allocated,
    Mixed,
    Down,
    Drain,
    Draining,
    Error,
    Unknown,
}

impl NodeState {
    pub fn display(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Allocated => "allocated",
            Self::Mixed => "mixed",
            Self::Down => "down",
            Self::Drain => "drained",
            Self::Draining => "draining",
            Self::Error => "error",
            Self::Unknown => "unknown",
        }
    }

    /// Short suffix used in sinfo (e.g., "idle", "alloc", "mix").
    pub fn short(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Allocated => "alloc",
            Self::Mixed => "mix",
            Self::Down => "down",
            Self::Drain => "drain",
            Self::Draining => "drng",
            Self::Error => "err",
            Self::Unknown => "unk",
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(self, Self::Idle | Self::Mixed)
    }
}

impl std::fmt::Display for NodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display())
    }
}

/// A compute node in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    pub state: NodeState,
    pub state_reason: Option<String>,
    pub partitions: Vec<String>,

    pub total_resources: ResourceSet,
    pub alloc_resources: ResourceSet,

    pub arch: String,
    pub os: String,
    pub cpu_load: u32,
    pub free_memory_mb: u64,

    pub boot_time: Option<DateTime<Utc>>,
    pub last_busy: Option<DateTime<Utc>>,
    pub agent_start_time: Option<DateTime<Utc>>,
    pub last_heartbeat: Option<DateTime<Utc>>,

    /// Agent address for gRPC communication.
    pub address: Option<String>,
    /// Agent gRPC listen port.
    pub port: u16,
    /// WireGuard public key (for mesh setup).
    pub wg_pubkey: Option<String>,
    /// Agent version.
    pub version: Option<String>,
}

impl Node {
    pub fn new(name: String, resources: ResourceSet) -> Self {
        Self {
            name,
            state: NodeState::Unknown,
            state_reason: None,
            partitions: Vec::new(),
            total_resources: resources,
            alloc_resources: ResourceSet::default(),
            arch: String::new(),
            os: String::new(),
            cpu_load: 0,
            free_memory_mb: 0,
            boot_time: None,
            last_busy: None,
            agent_start_time: None,
            last_heartbeat: None,
            address: None,
            port: 6818,
            wg_pubkey: None,
            version: None,
        }
    }

    /// Available (unallocated) resources.
    pub fn available_resources(&self) -> ResourceSet {
        self.total_resources.subtract(&self.alloc_resources)
    }

    /// Whether this node can accept new work.
    pub fn is_schedulable(&self) -> bool {
        self.state.is_available()
    }

    /// Update state based on allocation level.
    pub fn update_state_from_alloc(&mut self) {
        if self.state == NodeState::Down
            || self.state == NodeState::Drain
            || self.state == NodeState::Draining
            || self.state == NodeState::Error
        {
            return; // Don't override admin states
        }

        if self.alloc_resources.cpus == 0 && self.alloc_resources.gpus.is_empty() {
            self.state = NodeState::Idle;
        } else if self.alloc_resources.cpus >= self.total_resources.cpus {
            self.state = NodeState::Allocated;
        } else {
            self.state = NodeState::Mixed;
        }
    }
}
