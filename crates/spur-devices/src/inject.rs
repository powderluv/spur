// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::cdi::spec::{ContainerEdits, DeviceNode, Mount};
use crate::registry::entry::DeviceEntry;

#[derive(Clone, Debug, Default)]
pub struct InjectActions {
    pub gpu_visibility_vars: HashSet<String>,
}

impl InjectActions {
    pub fn merge(&mut self, other: &Self) {
        self.gpu_visibility_vars
            .extend(other.gpu_visibility_vars.iter().cloned());
    }
}

#[derive(Clone, Debug, Default)]
pub struct MergedEdits {
    pub device_edits: ContainerEdits,
    pub shared_edits: ContainerEdits,
    pub inject_actions: InjectActions,
}

impl MergedEdits {
    pub fn all_device_paths(&self) -> Vec<&str> {
        self.device_edits
            .device_nodes
            .iter()
            .chain(self.shared_edits.device_nodes.iter())
            .map(|dn| dn.effective_host_path())
            .collect()
    }

    pub fn all_env(&self) -> Vec<&str> {
        self.device_edits
            .env
            .iter()
            .chain(self.shared_edits.env.iter())
            .map(|s| s.as_str())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindMountPlan {
    pub source: String,
    pub target: String,
    pub readonly: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceNodePlan {
    pub host_path: String,
    pub container_path: String,
    pub dev_type: Option<String>,
    pub major: Option<i64>,
    pub minor: Option<i64>,
    pub permissions: String,
    pub uid: Option<u32>,
    pub gid: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HookPlan {
    pub hook_name: String,
    pub path: String,
    pub args: Vec<String>,
    pub env: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContainerInjectionPlan {
    pub device_nodes: Vec<DeviceNodePlan>,
    pub mounts: Vec<BindMountPlan>,
    pub env: HashMap<String, String>,
    pub hooks: Vec<HookPlan>,
    pub additional_gids: Vec<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HostInjectionPlan {
    pub env: HashMap<String, String>,
    pub visible_devices: Vec<String>,
    pub device_paths: Vec<String>,
}

/// Merge container edits and inject actions from resolved registry entries.
pub fn merge_edits_from_devices(devices: &[&DeviceEntry]) -> MergedEdits {
    let mut result = MergedEdits::default();
    let mut shared_applied: HashSet<String> = HashSet::new();

    for dev in devices {
        result.device_edits.append(&dev.device_edits);

        if !shared_applied.contains(&dev.kind) && !dev.shared_edits.is_empty() {
            result.shared_edits.append(&dev.shared_edits);
            shared_applied.insert(dev.kind.clone());
        }

        result.inject_actions.merge(&dev.inject_actions);
    }

    result
}

pub(crate) fn device_ids_for_gres(devices: &[&DeviceEntry], gres_name: &str) -> Vec<u32> {
    devices
        .iter()
        .filter(|d| d.gres_name == gres_name)
        .map(|d| d.device_id)
        .collect()
}

fn format_device_id_list(ids: &[u32]) -> String {
    ids.iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Set job-scoped device env vars from resolved registry entries.
pub(crate) fn apply_spur_job_device_env(
    env: &mut HashMap<String, String>,
    devices: &[&DeviceEntry],
) {
    let gpu_ids = device_ids_for_gres(devices, "gpu");
    if gpu_ids.is_empty() {
        return;
    }
    env.insert("SPUR_JOB_GPUS".into(), format_device_id_list(&gpu_ids));
}

pub fn build_injection_plans(
    devices: &[&DeviceEntry],
    job_uid: u32,
    job_gid: u32,
) -> (HostInjectionPlan, ContainerInjectionPlan) {
    let edits = merge_edits_from_devices(devices);
    let gpu_ids = device_ids_for_gres(devices, "gpu");
    let mut host_plan = HostInjector::plan(&edits, &gpu_ids);
    let mut container_plan = ContainerInjector::plan(&edits, &gpu_ids, job_uid, job_gid);
    apply_spur_job_device_env(&mut host_plan.env, devices);
    apply_spur_job_device_env(&mut container_plan.env, devices);
    (host_plan, container_plan)
}

pub fn gpu_visibility_for_vendor(vendor: Option<&str>) -> HashSet<String> {
    match vendor {
        Some("amd") => HashSet::from(["ROCR_VISIBLE_DEVICES".into()]),
        Some("nvidia") => HashSet::from(["CUDA_VISIBLE_DEVICES".into()]),
        Some("intel") => HashSet::from(["ZE_AFFINITY_MASK".into()]),
        _ => HashSet::new(),
    }
}

/// Keep the first device node per container path (per-device edits before shared edits).
fn unique_device_nodes<'a>(
    device_nodes: impl IntoIterator<Item = &'a DeviceNode>,
    shared_nodes: impl IntoIterator<Item = &'a DeviceNode>,
) -> Vec<&'a DeviceNode> {
    let mut seen = HashSet::new();
    device_nodes
        .into_iter()
        .chain(shared_nodes)
        .filter(|dn| seen.insert(dn.path.clone()))
        .collect()
}

/// Keep the first mount per container target path (per-device edits before shared edits).
fn unique_mounts<'a>(
    device_mounts: impl IntoIterator<Item = &'a Mount>,
    shared_mounts: impl IntoIterator<Item = &'a Mount>,
) -> Vec<&'a Mount> {
    let mut seen = HashSet::new();
    device_mounts
        .into_iter()
        .chain(shared_mounts)
        .filter(|m| seen.insert(m.container_path.clone()))
        .collect()
}

pub struct ContainerInjector;

impl ContainerInjector {
    pub fn plan(
        edits: &MergedEdits,
        device_ids: &[u32],
        job_uid: u32,
        job_gid: u32,
    ) -> ContainerInjectionPlan {
        let mut plan = ContainerInjectionPlan::default();

        for dn in unique_device_nodes(
            &edits.device_edits.device_nodes,
            &edits.shared_edits.device_nodes,
        ) {
            plan.device_nodes.push(DeviceNodePlan {
                host_path: dn.effective_host_path().to_owned(),
                container_path: dn.path.clone(),
                dev_type: dn.r#type.clone(),
                major: dn.major,
                minor: dn.minor,
                permissions: dn.permissions.clone().unwrap_or_else(|| "rw".into()),
                uid: dn.uid.or(Some(job_uid)),
                gid: dn.gid.or(Some(job_gid)),
            });
        }

        let mut mount_list: Vec<BindMountPlan> =
            unique_mounts(&edits.device_edits.mounts, &edits.shared_edits.mounts)
                .into_iter()
                .map(|m| {
                    let readonly = m
                        .options
                        .as_ref()
                        .is_some_and(|opts| opts.iter().any(|o| o == "ro"));
                    BindMountPlan {
                        source: m.host_path.clone(),
                        target: m.container_path.clone(),
                        readonly,
                    }
                })
                .collect();
        mount_list.sort_by_key(|m| m.target.matches('/').count());
        plan.mounts = mount_list;

        for env_str in edits.all_env() {
            if let Some((key, value)) = env_str.split_once('=') {
                plan.env.insert(key.to_owned(), value.to_owned());
            }
        }

        apply_gpu_visibility(&mut plan.env, &edits.inject_actions, device_ids);

        for hook in edits
            .device_edits
            .hooks
            .iter()
            .chain(edits.shared_edits.hooks.iter())
        {
            plan.hooks.push(HookPlan {
                hook_name: hook.hook_name.clone(),
                path: hook.path.clone(),
                args: hook.args.clone().unwrap_or_default(),
                env: hook.env.clone().unwrap_or_default(),
            });
        }

        let mut gids: Vec<u32> = edits
            .device_edits
            .additional_gids
            .iter()
            .chain(edits.shared_edits.additional_gids.iter())
            .copied()
            .filter(|g| *g > 0)
            .collect();
        gids.sort_unstable();
        gids.dedup();
        plan.additional_gids = gids;

        debug!(
            device_nodes = plan.device_nodes.len(),
            mounts = plan.mounts.len(),
            env_vars = plan.env.len(),
            hooks = plan.hooks.len(),
            "built container injection plan"
        );

        plan
    }
}

pub struct HostInjector;

impl HostInjector {
    pub fn plan(edits: &MergedEdits, device_ids: &[u32]) -> HostInjectionPlan {
        let mut plan = HostInjectionPlan::default();

        for env_str in edits.all_env() {
            if let Some((key, value)) = env_str.split_once('=') {
                plan.env.insert(key.to_owned(), value.to_owned());
            }
        }

        apply_gpu_visibility(&mut plan.env, &edits.inject_actions, device_ids);

        plan.visible_devices = edits
            .device_edits
            .device_nodes
            .iter()
            .map(|dn| dn.effective_host_path().to_owned())
            .filter(|p| Path::new(p).exists())
            .collect();

        plan.device_paths = edits
            .all_device_paths()
            .into_iter()
            .map(|s| s.to_owned())
            .collect();

        debug!(
            env_vars = plan.env.len(),
            visible_devices = plan.visible_devices.len(),
            "built host injection plan"
        );

        plan
    }
}

fn apply_gpu_visibility(
    env: &mut HashMap<String, String>,
    actions: &InjectActions,
    device_ids: &[u32],
) {
    let value = format_device_id_list(device_ids);
    for var in &actions.gpu_visibility_vars {
        env.insert(var.clone(), value.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdi::spec::{ContainerEdits, DeviceNode, Hook, Mount};

    fn amd_inject_actions() -> InjectActions {
        InjectActions {
            gpu_visibility_vars: HashSet::from(["ROCR_VISIBLE_DEVICES".into()]),
        }
    }

    fn make_amd_edits() -> MergedEdits {
        MergedEdits {
            device_edits: ContainerEdits {
                device_nodes: vec![
                    DeviceNode {
                        path: "/dev/dri/renderD128".into(),
                        host_path: None,
                        r#type: Some("c".into()),
                        major: Some(226),
                        minor: Some(128),
                        file_mode: None,
                        permissions: Some("rw".into()),
                        uid: None,
                        gid: None,
                    },
                    DeviceNode {
                        path: "/dev/dri/renderD129".into(),
                        host_path: None,
                        r#type: Some("c".into()),
                        major: Some(226),
                        minor: Some(129),
                        file_mode: None,
                        permissions: Some("rw".into()),
                        uid: None,
                        gid: None,
                    },
                ],
                env: vec!["ROCR_VISIBLE_DEVICES=0,1".into()],
                ..Default::default()
            },
            shared_edits: ContainerEdits {
                device_nodes: vec![DeviceNode {
                    path: "/dev/kfd".into(),
                    host_path: None,
                    r#type: Some("c".into()),
                    major: None,
                    minor: None,
                    file_mode: None,
                    permissions: Some("rw".into()),
                    uid: None,
                    gid: None,
                }],
                mounts: vec![Mount {
                    host_path: "/opt/rocm/lib".into(),
                    container_path: "/opt/rocm/lib".into(),
                    r#type: None,
                    options: Some(vec![
                        "ro".into(),
                        "nosuid".into(),
                        "nodev".into(),
                        "bind".into(),
                    ]),
                }],
                ..Default::default()
            },
            inject_actions: amd_inject_actions(),
        }
    }

    #[test]
    fn test_container_plan_device_nodes() {
        let edits = make_amd_edits();
        let plan = ContainerInjector::plan(&edits, &[0, 1], 1000, 1000);

        assert_eq!(plan.device_nodes.len(), 3);
        assert_eq!(plan.device_nodes[0].host_path, "/dev/dri/renderD128");
        assert_eq!(plan.device_nodes[1].host_path, "/dev/dri/renderD129");
        assert_eq!(plan.device_nodes[2].host_path, "/dev/kfd");
    }

    #[test]
    fn test_container_plan_uid_gid_fallback() {
        let edits = make_amd_edits();
        let plan = ContainerInjector::plan(&edits, &[0], 1000, 1000);

        assert_eq!(plan.device_nodes[0].uid, Some(1000));
        assert_eq!(plan.device_nodes[0].gid, Some(1000));
    }

    #[test]
    fn test_container_plan_mounts_sorted() {
        let edits = make_amd_edits();
        let plan = ContainerInjector::plan(&edits, &[0], 1000, 1000);

        assert_eq!(plan.mounts.len(), 1);
        assert_eq!(plan.mounts[0].source, "/opt/rocm/lib");
        assert!(plan.mounts[0].readonly);
    }

    #[test]
    fn test_container_plan_env_vars_amd() {
        let edits = make_amd_edits();
        let plan = ContainerInjector::plan(&edits, &[0, 1], 1000, 1000);

        assert_eq!(plan.env.get("ROCR_VISIBLE_DEVICES").unwrap(), "0,1");
        assert!(!plan.env.contains_key("SPUR_JOB_GPUS"));
        assert!(!plan.env.contains_key("CUDA_VISIBLE_DEVICES"));
    }

    fn gpu_device_entry(device_id: u32) -> DeviceEntry {
        use crate::registry::entry::{DeviceCapability, DeviceEntry};
        DeviceEntry {
            index: device_id,
            gres_name: "gpu".into(),
            resource_type: None,
            kind: "amd.com/gpu".into(),
            device_id,
            capability: DeviceCapability::Injectable,
            memory_mb: 0,
            numa_node: None,
            cores: None,
            links: None,
            link_type: None,
            pci_bdf: None,
            vendor: Some("amd".into()),
            inject_actions: InjectActions::default(),
            capacity: 1,
            allocated: 0,
            device_edits: ContainerEdits::default(),
            shared_edits: ContainerEdits::default(),
            device_paths: Vec::new(),
        }
    }

    #[test]
    fn test_apply_spur_job_device_env_gpu_only() {
        let gpu_devices: Vec<DeviceEntry> = vec![gpu_device_entry(0), gpu_device_entry(1)];
        let gpu_refs: Vec<&DeviceEntry> = gpu_devices.iter().collect();
        let mut env = HashMap::new();
        apply_spur_job_device_env(&mut env, &gpu_refs);
        assert_eq!(env.get("SPUR_JOB_GPUS").unwrap(), "0,1");
    }

    #[test]
    fn test_apply_spur_job_device_env_skips_non_gpu() {
        let mut nic = gpu_device_entry(2);
        nic.gres_name = "nic".into();
        let devices = [nic];
        let refs: Vec<&DeviceEntry> = devices.iter().collect();
        let mut env = HashMap::new();
        apply_spur_job_device_env(&mut env, &refs);
        assert!(!env.contains_key("SPUR_JOB_GPUS"));
    }

    #[test]
    fn test_container_plan_nvidia_env_vars() {
        let edits = MergedEdits {
            device_edits: ContainerEdits::default(),
            shared_edits: ContainerEdits::default(),
            inject_actions: InjectActions {
                gpu_visibility_vars: HashSet::from(["CUDA_VISIBLE_DEVICES".into()]),
            },
        };
        let plan = ContainerInjector::plan(&edits, &[0, 1], 1000, 1000);

        assert_eq!(plan.env.get("CUDA_VISIBLE_DEVICES").unwrap(), "0,1");
        assert!(!plan.env.contains_key("ROCR_VISIBLE_DEVICES"));
    }

    #[test]
    fn test_container_plan_dedup_device_nodes() {
        let edits = MergedEdits {
            device_edits: ContainerEdits {
                device_nodes: vec![DeviceNode {
                    path: "/dev/kfd".into(),
                    host_path: None,
                    r#type: None,
                    major: None,
                    minor: None,
                    file_mode: None,
                    permissions: None,
                    uid: None,
                    gid: None,
                }],
                ..Default::default()
            },
            shared_edits: ContainerEdits {
                device_nodes: vec![DeviceNode {
                    path: "/dev/kfd".into(),
                    host_path: None,
                    r#type: None,
                    major: None,
                    minor: None,
                    file_mode: None,
                    permissions: None,
                    uid: None,
                    gid: None,
                }],
                ..Default::default()
            },
            inject_actions: InjectActions::default(),
        };

        let plan = ContainerInjector::plan(&edits, &[], 0, 0);
        assert_eq!(plan.device_nodes.len(), 1);
    }

    #[test]
    fn test_container_plan_hooks() {
        let mut edits = make_amd_edits();
        edits.shared_edits.hooks.push(Hook {
            hook_name: "createContainer".into(),
            path: "/usr/bin/vendor-hook".into(),
            args: Some(vec!["vendor-hook".into(), "update-ldcache".into()]),
            env: None,
            timeout: None,
        });

        let plan = ContainerInjector::plan(&edits, &[0], 1000, 1000);
        assert_eq!(plan.hooks.len(), 1);
        assert_eq!(plan.hooks[0].hook_name, "createContainer");
        assert_eq!(plan.hooks[0].path, "/usr/bin/vendor-hook");
    }

    #[test]
    fn test_container_plan_additional_gids() {
        let mut edits = make_amd_edits();
        edits.shared_edits.additional_gids = vec![44, 109, 44];

        let plan = ContainerInjector::plan(&edits, &[0], 1000, 1000);
        assert_eq!(plan.additional_gids, vec![44, 109]);
    }

    #[test]
    fn test_host_plan_env_vars_amd() {
        let edits = make_amd_edits();
        let plan = HostInjector::plan(&edits, &[0, 1]);

        assert_eq!(plan.env.get("ROCR_VISIBLE_DEVICES").unwrap(), "0,1");
        assert!(!plan.env.contains_key("SPUR_JOB_GPUS"));
        assert!(!plan.env.contains_key("CUDA_VISIBLE_DEVICES"));
    }

    #[test]
    fn test_host_plan_hides_gpus_when_no_devices() {
        let edits = MergedEdits {
            inject_actions: amd_inject_actions(),
            ..Default::default()
        };
        let plan = HostInjector::plan(&edits, &[]);
        assert_eq!(plan.env.get("ROCR_VISIBLE_DEVICES").unwrap(), "");
    }

    #[test]
    fn test_host_plan_visible_devices() {
        let edits = make_amd_edits();
        let plan = HostInjector::plan(&edits, &[0, 1]);

        assert!(plan
            .device_paths
            .contains(&"/dev/dri/renderD128".to_string()));
        assert!(plan
            .device_paths
            .contains(&"/dev/dri/renderD129".to_string()));
        assert!(plan.device_paths.contains(&"/dev/kfd".to_string()));
    }

    #[test]
    fn test_apply_gpu_visibility() {
        let mut env = HashMap::new();
        let actions = InjectActions {
            gpu_visibility_vars: HashSet::from([
                "CUDA_VISIBLE_DEVICES".into(),
                "ROCR_VISIBLE_DEVICES".into(),
                "ZE_AFFINITY_MASK".into(),
                "GPU_DEVICE_ORDINAL".into(),
            ]),
        };
        apply_gpu_visibility(&mut env, &actions, &[0, 1]);

        assert_eq!(env.get("CUDA_VISIBLE_DEVICES").unwrap(), "0,1");
        assert_eq!(env.get("ROCR_VISIBLE_DEVICES").unwrap(), "0,1");
        assert_eq!(env.get("ZE_AFFINITY_MASK").unwrap(), "0,1");
        assert_eq!(env.get("GPU_DEVICE_ORDINAL").unwrap(), "0,1");
    }

    #[test]
    fn test_apply_gpu_visibility_no_devices() {
        let mut env = HashMap::new();
        let actions = InjectActions {
            gpu_visibility_vars: HashSet::from(["ROCR_VISIBLE_DEVICES".into()]),
        };
        apply_gpu_visibility(&mut env, &actions, &[]);
        assert_eq!(env.get("ROCR_VISIBLE_DEVICES").unwrap(), "");
    }

    #[test]
    fn test_apply_gpu_visibility_no_actions() {
        let mut env = HashMap::new();
        apply_gpu_visibility(&mut env, &InjectActions::default(), &[0]);
        assert!(env.is_empty());
    }

    #[test]
    fn test_gpu_visibility_for_vendor() {
        assert!(gpu_visibility_for_vendor(Some("amd")).contains("ROCR_VISIBLE_DEVICES"));
        assert!(gpu_visibility_for_vendor(Some("nvidia")).contains("CUDA_VISIBLE_DEVICES"));
        assert!(gpu_visibility_for_vendor(Some("intel")).contains("ZE_AFFINITY_MASK"));
        assert!(gpu_visibility_for_vendor(None).is_empty());
    }

    #[test]
    fn test_inject_actions_merge() {
        let mut a = InjectActions {
            gpu_visibility_vars: HashSet::from(["ROCR_VISIBLE_DEVICES".into()]),
        };
        let b = InjectActions {
            gpu_visibility_vars: HashSet::from(["CUDA_VISIBLE_DEVICES".into()]),
        };
        a.merge(&b);
        assert_eq!(a.gpu_visibility_vars.len(), 2);
        assert!(a.gpu_visibility_vars.contains("ROCR_VISIBLE_DEVICES"));
        assert!(a.gpu_visibility_vars.contains("CUDA_VISIBLE_DEVICES"));
    }
}
