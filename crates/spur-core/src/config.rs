use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

use crate::partition::{Partition, PartitionState, PreemptMode};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse TOML: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("missing required field: {0}")]
    MissingField(String),
    #[error("invalid value for {field}: {value}")]
    InvalidValue { field: String, value: String },
}

/// Top-level configuration (slurm.conf equivalent, in TOML).
///
/// We support reading both our native TOML format and the traditional
/// Slurm key=value format for migration purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlurmConfig {
    pub cluster_name: String,

    #[serde(default)]
    pub controller: ControllerConfig,

    #[serde(default)]
    pub accounting: AccountingConfig,

    #[serde(default)]
    pub scheduler: SchedulerConfig,

    #[serde(default)]
    pub auth: AuthConfig,

    #[serde(default)]
    pub partitions: Vec<PartitionConfig>,

    #[serde(default)]
    pub nodes: Vec<NodeConfig>,

    #[serde(default)]
    pub network: NetworkConfig,

    #[serde(default)]
    pub logging: LoggingConfig,

    #[serde(default)]
    pub kubernetes: KubernetesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// gRPC listen address for the controller.
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,
    /// REST API listen address.
    #[serde(default = "default_rest_addr")]
    pub rest_addr: String,
    /// Hostname(s) for HA. First is primary.
    #[serde(default = "default_hosts")]
    pub hosts: Vec<String>,
    /// State save location.
    #[serde(default = "default_state_dir")]
    pub state_dir: String,
    /// Max job ID before wrapping.
    #[serde(default = "default_max_job_id")]
    pub max_job_id: u32,
    /// First job ID.
    #[serde(default = "default_one")]
    pub first_job_id: u32,
}

fn default_listen_addr() -> String {
    "[::]:6817".into()
}
fn default_rest_addr() -> String {
    "[::]:6820".into()
}
fn default_hosts() -> Vec<String> {
    vec!["localhost".into()]
}
fn default_state_dir() -> String {
    "/var/spool/spur".into()
}
fn default_max_job_id() -> u32 {
    999_999_999
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            listen_addr: "[::]:6817".into(),
            rest_addr: "[::]:6820".into(),
            hosts: vec!["localhost".into()],
            state_dir: "/var/spool/spur".into(),
            max_job_id: 999_999_999,
            first_job_id: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountingConfig {
    /// Address of the accounting daemon.
    #[serde(default = "default_accounting_host")]
    pub host: String,
    /// PostgreSQL connection string.
    #[serde(default = "default_database_url")]
    pub database_url: String,
    /// How long to keep completed job records.
    #[serde(default = "default_purge_days")]
    pub purge_after_days: u32,
}

fn default_accounting_host() -> String {
    "localhost:6819".into()
}
fn default_database_url() -> String {
    "postgresql://spur:spur@localhost/spur".into()
}
fn default_purge_days() -> u32 {
    365
}

impl Default for AccountingConfig {
    fn default() -> Self {
        Self {
            host: "localhost:6819".into(),
            database_url: "postgresql://spur:spur@localhost/spur".into(),
            purge_after_days: 365,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler plugin name.
    #[serde(default = "default_scheduler_plugin")]
    pub plugin: String,
    /// How often to run the scheduler (seconds).
    #[serde(default = "default_one")]
    pub interval_secs: u32,
    /// Max jobs to evaluate per cycle.
    #[serde(default = "default_max_jobs")]
    pub max_jobs_per_cycle: u32,
    /// Fair-share decay half-life (days).
    #[serde(default = "default_halflife")]
    pub fairshare_halflife_days: u32,
    /// Default job time limit (minutes), if not set per-partition.
    #[serde(default = "default_time_limit")]
    pub default_time_limit_minutes: u32,
}

fn default_scheduler_plugin() -> String {
    "backfill".into()
}
fn default_max_jobs() -> u32 {
    10000
}
fn default_halflife() -> u32 {
    14
}
fn default_time_limit() -> u32 {
    60
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            plugin: "backfill".into(),
            interval_secs: 1,
            max_jobs_per_cycle: 10000,
            fairshare_halflife_days: 14,
            default_time_limit_minutes: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Auth plugin: "jwt", "munge", "none".
    pub plugin: String,
    /// JWT secret key (file path or inline).
    pub jwt_key: Option<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            plugin: "jwt".into(),
            jwt_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    pub name: String,
    #[serde(default)]
    pub default: bool,
    #[serde(default = "default_partition_state")]
    pub state: String,
    pub nodes: String,
    pub max_time: Option<String>,
    pub default_time: Option<String>,
    pub max_nodes: Option<u32>,
    #[serde(default = "default_one")]
    pub min_nodes: u32,
    #[serde(default)]
    pub allow_accounts: Vec<String>,
    #[serde(default)]
    pub allow_groups: Vec<String>,
    #[serde(default)]
    pub priority_tier: u32,
    #[serde(default)]
    pub preempt_mode: String,
}

fn default_partition_state() -> String {
    "UP".into()
}
fn default_one() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Hostlist pattern for this node definition.
    pub names: String,
    pub cpus: u32,
    pub memory_mb: u64,
    #[serde(default)]
    pub gres: Vec<String>,
    #[serde(default)]
    pub features: Vec<String>,
    /// Override address (if different from hostname).
    pub address: Option<String>,
}

/// Network / WireGuard mesh configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Enable WireGuard mesh networking between nodes.
    #[serde(default)]
    pub wg_enabled: bool,
    /// CIDR block for WireGuard address allocation (default: 10.44.0.0/16).
    #[serde(default = "default_wg_cidr")]
    pub wg_cidr: String,
    /// WireGuard interface name (default: spur0).
    #[serde(default = "default_wg_interface")]
    pub wg_interface: String,
    /// WireGuard listen port (default: 51820).
    #[serde(default = "default_wg_port")]
    pub wg_port: u16,
    /// Agent gRPC listen port (default: 6818).
    #[serde(default = "default_agent_port")]
    pub agent_port: u16,
}

fn default_wg_cidr() -> String {
    "10.44.0.0/16".into()
}
fn default_wg_interface() -> String {
    "spur0".into()
}
fn default_wg_port() -> u16 {
    51820
}
fn default_agent_port() -> u16 {
    6818
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            wg_enabled: false,
            wg_cidr: "10.44.0.0/16".into(),
            wg_interface: "spur0".into(),
            wg_port: 51820,
            agent_port: 6818,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub file: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "text".into(),
            file: None,
        }
    }
}

/// Kubernetes integration configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    /// Enable K8s integration.
    #[serde(default)]
    pub enabled: bool,
    /// Path to kubeconfig file. If empty, uses in-cluster config.
    #[serde(default)]
    pub kubeconfig: Option<String>,
    /// K8s namespace for SpurJob CRDs and Pods.
    #[serde(default = "default_k8s_namespace")]
    pub namespace: String,
    /// Label selector for K8s nodes to include in the Spur pool.
    #[serde(default = "default_k8s_node_selector")]
    pub node_label_selector: String,
}

fn default_k8s_namespace() -> String {
    "spur".into()
}

fn default_k8s_node_selector() -> String {
    "spur.ai/managed=true".into()
}

impl Default for KubernetesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            kubeconfig: None,
            namespace: "spur".into(),
            node_label_selector: "spur.ai/managed=true".into(),
        }
    }
}

impl SlurmConfig {
    /// Load from a TOML file.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load from a TOML string.
    pub fn from_str(s: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(s)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.cluster_name.is_empty() {
            return Err(ConfigError::MissingField("cluster_name".into()));
        }
        Ok(())
    }

    /// Convert partition configs to Partition structs.
    pub fn build_partitions(&self) -> Vec<Partition> {
        self.partitions
            .iter()
            .map(|pc| Partition {
                name: pc.name.clone(),
                state: match pc.state.to_uppercase().as_str() {
                    "UP" => PartitionState::Up,
                    "DOWN" => PartitionState::Down,
                    "DRAIN" => PartitionState::Drain,
                    _ => PartitionState::Inactive,
                },
                is_default: pc.default,
                nodes: pc.nodes.clone(),
                max_time_minutes: pc.max_time.as_ref().and_then(|t| parse_time_minutes(t)),
                default_time_minutes: pc.default_time.as_ref().and_then(|t| parse_time_minutes(t)),
                max_nodes: pc.max_nodes,
                min_nodes: pc.min_nodes,
                priority_tier: pc.priority_tier,
                preempt_mode: match pc.preempt_mode.to_lowercase().as_str() {
                    "cancel" => PreemptMode::Cancel,
                    "requeue" => PreemptMode::Requeue,
                    "suspend" => PreemptMode::Suspend,
                    _ => PreemptMode::Off,
                },
                ..Default::default()
            })
            .collect()
    }
}

/// Parse a time string like "72:00:00", "4-00:00:00", "INFINITE", "60" (minutes).
pub fn parse_time_minutes(s: &str) -> Option<u32> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("INFINITE") || s.eq_ignore_ascii_case("UNLIMITED") {
        return None; // No limit
    }

    // days-hours:minutes:seconds
    if let Some((days, rest)) = s.split_once('-') {
        let days: u32 = days.parse().ok()?;
        let hms = parse_hms(rest)?;
        return Some(days * 24 * 60 + hms);
    }

    // hours:minutes:seconds or hours:minutes or just minutes
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        1 => parts[0].parse().ok(),
        2 => {
            let h: u32 = parts[0].parse().ok()?;
            let m: u32 = parts[1].parse().ok()?;
            Some(h * 60 + m)
        }
        3 => Some(parse_hms(s)?),
        _ => None,
    }
}

/// Parse a time string to total seconds (not minutes).
///
/// Same Slurm-compatible format as `parse_time_minutes` but with second
/// granularity: "N" → N minutes, "H:MM" → hours+minutes, "H:MM:SS" → exact.
pub fn parse_time_seconds(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("INFINITE") || s.eq_ignore_ascii_case("UNLIMITED") {
        return None;
    }

    // days-hours:minutes:seconds
    if let Some((days, rest)) = s.split_once('-') {
        let days: u64 = days.parse().ok()?;
        return Some(days * 86400 + parse_hms_seconds(rest)?);
    }

    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        1 => {
            // Just minutes → convert to seconds
            let mins: u64 = parts[0].parse().ok()?;
            Some(mins * 60)
        }
        2 => {
            // HH:MM → hours and minutes (no seconds)
            let h: u64 = parts[0].parse().ok()?;
            let m: u64 = parts[1].parse().ok()?;
            Some(h * 3600 + m * 60)
        }
        3 => {
            // HH:MM:SS
            let h: u64 = parts[0].parse().ok()?;
            let m: u64 = parts[1].parse().ok()?;
            let sec: u64 = parts[2].parse().ok()?;
            Some(h * 3600 + m * 60 + sec)
        }
        _ => None,
    }
}

fn parse_hms_seconds(s: &str) -> Option<u64> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        2 => {
            let h: u64 = parts[0].parse().ok()?;
            let m: u64 = parts[1].parse().ok()?;
            Some(h * 3600 + m * 60)
        }
        3 => {
            let h: u64 = parts[0].parse().ok()?;
            let m: u64 = parts[1].parse().ok()?;
            let sec: u64 = parts[2].parse().ok()?;
            Some(h * 3600 + m * 60 + sec)
        }
        _ => None,
    }
}

fn parse_hms(s: &str) -> Option<u32> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 && parts.len() != 2 {
        return None;
    }
    let h: u32 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    let s: u32 = if parts.len() == 3 {
        parts[2].parse().ok()?
    } else {
        0
    };
    Some(h * 60 + m + if s > 0 { 1 } else { 0 }) // Round up seconds
}

/// Format minutes as D-HH:MM:SS or HH:MM:SS.
pub fn format_time(total_minutes: Option<u32>) -> String {
    match total_minutes {
        None => "UNLIMITED".into(),
        Some(mins) => {
            let days = mins / (24 * 60);
            let hours = (mins % (24 * 60)) / 60;
            let minutes = mins % 60;
            if days > 0 {
                format!("{}-{:02}:{:02}:00", days, hours, minutes)
            } else {
                format!("{:02}:{:02}:00", hours, minutes)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_time() {
        assert_eq!(parse_time_minutes("60"), Some(60));
        assert_eq!(parse_time_minutes("1:30"), Some(90));
        assert_eq!(parse_time_minutes("72:00:00"), Some(4320));
        assert_eq!(parse_time_minutes("1-00:00:00"), Some(1440));
        assert_eq!(parse_time_minutes("INFINITE"), None);
    }

    #[test]
    fn test_parse_time_seconds() {
        // "N" → N minutes in seconds
        assert_eq!(parse_time_seconds("1"), Some(60));
        assert_eq!(parse_time_seconds("60"), Some(3600));
        // "H:MM" → exact seconds
        assert_eq!(parse_time_seconds("1:30"), Some(5400)); // 1h30m
                                                            // "H:MM:SS" → exact seconds (the key case)
        assert_eq!(parse_time_seconds("0:00:10"), Some(10));
        assert_eq!(parse_time_seconds("0:01:30"), Some(90));
        assert_eq!(parse_time_seconds("1:00:00"), Some(3600));
        // days-HH:MM:SS
        assert_eq!(parse_time_seconds("1-00:00:00"), Some(86400));
        assert_eq!(parse_time_seconds("7-00:00:00"), Some(604800));
        // limits
        assert_eq!(parse_time_seconds("INFINITE"), None);
        assert_eq!(parse_time_seconds("UNLIMITED"), None);
    }

    #[test]
    fn test_format_time() {
        assert_eq!(format_time(Some(90)), "01:30:00");
        assert_eq!(format_time(Some(1500)), "1-01:00:00");
        assert_eq!(format_time(None), "UNLIMITED");
    }

    #[test]
    fn test_load_config() {
        let toml = r#"
cluster_name = "test-cluster"

[controller]
listen_addr = "[::]:6817"
rest_addr = "[::]:6820"
hosts = ["ctrl1", "ctrl2"]
state_dir = "/var/spool/spur"
max_job_id = 999999999
first_job_id = 100

[scheduler]
plugin = "backfill"
interval_secs = 2

[[partitions]]
name = "gpu"
default = true
nodes = "gpu[001-008]"
max_time = "72:00:00"

[[partitions]]
name = "cpu"
nodes = "cpu[001-064]"
max_time = "168:00:00"

[[nodes]]
names = "gpu[001-008]"
cpus = 128
memory_mb = 512000
gres = ["gpu:mi300x:8"]

[[nodes]]
names = "cpu[001-064]"
cpus = 256
memory_mb = 1024000
"#;

        let config = SlurmConfig::from_str(toml).unwrap();
        assert_eq!(config.cluster_name, "test-cluster");
        assert_eq!(config.partitions.len(), 2);
        assert_eq!(config.nodes.len(), 2);
        assert!(config.partitions[0].default);

        let parts = config.build_partitions();
        assert_eq!(parts[0].name, "gpu");
        assert_eq!(parts[0].max_time_minutes, Some(4320));
    }
}
