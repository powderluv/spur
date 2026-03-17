//! Accounting data models: accounts, users, QOS, associations, TRES.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trackable RESource types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TresType {
    Cpu,
    Memory,    // MB
    Energy,    // Joules
    Node,
    Gpu,
    Billing,   // Weighted composite
}

impl TresType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Memory => "mem",
            Self::Energy => "energy",
            Self::Node => "node",
            Self::Gpu => "gres/gpu",
            Self::Billing => "billing",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "mem" | "memory" => Some(Self::Memory),
            "energy" => Some(Self::Energy),
            "node" => Some(Self::Node),
            "gres/gpu" | "gpu" => Some(Self::Gpu),
            "billing" => Some(Self::Billing),
            _ => None,
        }
    }
}

/// TRES usage/allocation record.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TresRecord {
    pub values: HashMap<TresType, u64>,
}

impl TresRecord {
    pub fn new() -> Self {
        Self { values: HashMap::new() }
    }

    pub fn set(&mut self, tres: TresType, value: u64) {
        self.values.insert(tres, value);
    }

    pub fn get(&self, tres: TresType) -> u64 {
        self.values.get(&tres).copied().unwrap_or(0)
    }

    pub fn add(&mut self, other: &TresRecord) {
        for (k, v) in &other.values {
            *self.values.entry(*k).or_insert(0) += v;
        }
    }

    /// Format as "cpu=N,mem=N,gres/gpu=N" string.
    pub fn format(&self) -> String {
        let mut parts: Vec<String> = self.values.iter()
            .filter(|(_, v)| **v > 0)
            .map(|(k, v)| format!("{}={}", k.name(), v))
            .collect();
        parts.sort();
        parts.join(",")
    }

    /// Parse from "cpu=N,mem=N" string.
    pub fn parse(s: &str) -> Self {
        let mut rec = Self::new();
        for part in s.split(',') {
            let part = part.trim();
            if let Some((key, val)) = part.split_once('=') {
                if let (Some(tres), Ok(v)) = (TresType::from_name(key.trim()), val.trim().parse::<u64>()) {
                    rec.set(tres, v);
                }
            }
        }
        rec
    }
}

/// An account in the accounting hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub name: String,
    pub description: String,
    pub organization: String,
    pub parent: Option<String>,
    pub fairshare_weight: u32,
    /// Resource limits for all jobs under this account.
    pub limits: AccountLimits,
}

/// Per-account resource limits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccountLimits {
    /// Max running jobs across all users in this account.
    pub max_running_jobs: Option<u32>,
    /// Max submitted (pending + running) jobs.
    pub max_submit_jobs: Option<u32>,
    /// Max TRES per job.
    pub max_tres_per_job: Option<TresRecord>,
    /// Max total TRES across all running jobs.
    pub grp_tres: Option<TresRecord>,
    /// Max wall time per job (minutes).
    pub max_wall_minutes: Option<u32>,
}

/// A user-account association.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Association {
    pub user: String,
    pub account: String,
    pub partition: Option<String>,
    pub fairshare_weight: u32,
    pub limits: AccountLimits,
    pub is_default: bool,
}

/// Quality of Service definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qos {
    pub name: String,
    pub description: String,
    pub priority: i32,
    pub preempt_mode: QosPreemptMode,
    pub limits: QosLimits,
    /// Usage factor — multiplier for fair-share usage accounting.
    /// 0.0 = don't charge, 1.0 = normal, 2.0 = double charge.
    pub usage_factor: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QosPreemptMode {
    #[default]
    Off,
    Cancel,
    Requeue,
    Suspend,
}

impl QosPreemptMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cancel" => Self::Cancel,
            "requeue" => Self::Requeue,
            "suspend" => Self::Suspend,
            _ => Self::Off,
        }
    }
}

/// Per-QOS resource limits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QosLimits {
    pub max_jobs_per_user: Option<u32>,
    pub max_submit_jobs_per_user: Option<u32>,
    pub max_tres_per_job: Option<TresRecord>,
    pub max_tres_per_user: Option<TresRecord>,
    pub grp_tres: Option<TresRecord>,
    pub max_wall_minutes: Option<u32>,
    pub grp_wall_minutes: Option<u32>,
}

impl Default for Account {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            organization: String::new(),
            parent: None,
            fairshare_weight: 1,
            limits: AccountLimits::default(),
        }
    }
}

impl Default for Qos {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            priority: 0,
            preempt_mode: QosPreemptMode::Off,
            limits: QosLimits::default(),
            usage_factor: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tres_format_parse() {
        let mut rec = TresRecord::new();
        rec.set(TresType::Cpu, 64);
        rec.set(TresType::Memory, 256000);
        rec.set(TresType::Gpu, 8);

        let formatted = rec.format();
        assert!(formatted.contains("cpu=64"));
        assert!(formatted.contains("gres/gpu=8"));

        let parsed = TresRecord::parse(&formatted);
        assert_eq!(parsed.get(TresType::Cpu), 64);
        assert_eq!(parsed.get(TresType::Gpu), 8);
    }

    #[test]
    fn test_tres_add() {
        let mut a = TresRecord::new();
        a.set(TresType::Cpu, 10);
        let mut b = TresRecord::new();
        b.set(TresType::Cpu, 20);
        b.set(TresType::Gpu, 4);
        a.add(&b);
        assert_eq!(a.get(TresType::Cpu), 30);
        assert_eq!(a.get(TresType::Gpu), 4);
    }

    #[test]
    fn test_qos_preempt_mode() {
        assert_eq!(QosPreemptMode::from_str("cancel"), QosPreemptMode::Cancel);
        assert_eq!(QosPreemptMode::from_str("off"), QosPreemptMode::Off);
        assert_eq!(QosPreemptMode::from_str("unknown"), QosPreemptMode::Off);
    }
}
