//! Native container support for Spur.
//!
//! Implements Enroot-like rootless containers using Linux user namespaces
//! and mount namespaces. No daemon, no Docker, no external runtime needed.
//!
//! Image format: squashfs (same as Enroot). Import OCI/Docker images with
//! `spur image import`.
//!
//! GPU passthrough:
//! - AMD: bind-mount /dev/kfd + /dev/dri/renderD* + ROCm libraries
//! - NVIDIA: bind-mount /dev/nvidia* + libnvidia-container or driver libs

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{bail, Context};
use tokio::process::Command;
use tracing::{debug, info, warn};

/// Where squashfs images and container rootfs are stored.
const IMAGE_DIR: &str = "/var/spool/spur/images";
const CONTAINER_DIR: &str = "/var/spool/spur/containers";

/// A parsed bind mount specification.
#[derive(Debug)]
pub struct BindMount {
    pub source: String,
    pub target: String,
    pub readonly: bool,
}

/// Container configuration for a job.
#[derive(Debug)]
pub struct ContainerConfig {
    pub image: String,
    pub mounts: Vec<BindMount>,
    pub workdir: Option<String>,
    pub name: Option<String>,
    pub readonly: bool,
    pub gpu_devices: Vec<u32>,
    pub environment: HashMap<String, String>,
}

/// Resolve image reference to a rootfs path.
///
/// Supports:
/// - Absolute path to squashfs file
/// - Image name (looked up in IMAGE_DIR)
/// - docker:// URI (must be pre-imported with `spur image import`)
pub fn resolve_image(image: &str) -> anyhow::Result<PathBuf> {
    // Absolute path to squashfs
    let path = Path::new(image);
    if path.is_absolute() && path.exists() {
        return Ok(path.to_path_buf());
    }

    // Check image dir
    let image_path = PathBuf::from(IMAGE_DIR).join(format!("{}.sqsh", sanitize_name(image)));
    if image_path.exists() {
        return Ok(image_path);
    }

    // Try without .sqsh extension
    let image_path = PathBuf::from(IMAGE_DIR).join(sanitize_name(image));
    if image_path.exists() {
        return Ok(image_path);
    }

    bail!(
        "container image '{}' not found. Import it first with: spur image import {}",
        image,
        image
    )
}

/// Create a container rootfs from a squashfs image.
///
/// Named containers are persistent; unnamed containers use job-specific paths
/// and are cleaned up after the job.
pub fn setup_rootfs(image_path: &Path, job_id: u32, name: Option<&str>) -> anyhow::Result<PathBuf> {
    let rootfs = if let Some(name) = name {
        PathBuf::from(CONTAINER_DIR).join(sanitize_name(name))
    } else {
        PathBuf::from(CONTAINER_DIR).join(format!("job_{}", job_id))
    };

    // If named container already exists, reuse it
    if rootfs.exists() && name.is_some() {
        debug!(path = %rootfs.display(), "reusing named container");
        return Ok(rootfs);
    }

    std::fs::create_dir_all(&rootfs)
        .with_context(|| format!("failed to create container rootfs at {}", rootfs.display()))?;

    // Extract squashfs to rootfs
    let status = std::process::Command::new("unsquashfs")
        .args([
            "-f", // force overwrite
            "-d", // destination
            rootfs.to_str().unwrap(),
            image_path.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .context("failed to run unsquashfs — is squashfs-tools installed?")?;

    if !status.success() {
        bail!("unsquashfs failed for image {}", image_path.display());
    }

    info!(rootfs = %rootfs.display(), "container rootfs created");
    Ok(rootfs)
}

/// Build the wrapper script that launches a job inside a container.
///
/// Uses `unshare` to create user + mount namespaces, then `chroot` into the
/// rootfs. This approach works without root — requires
/// `sysctl kernel.unprivileged_userns_clone=1` (default on most distros).
pub fn build_container_launch_script(
    config: &ContainerConfig,
    rootfs: &Path,
    inner_script_path: &str,
    job_id: u32,
) -> anyhow::Result<String> {
    let mut script = String::new();
    script.push_str("#!/bin/bash\nset -e\n\n");

    // Create mount points for bind mounts inside rootfs
    let rootfs_str = rootfs.to_string_lossy();

    // Ensure key directories exist in rootfs
    script.push_str(&format!(
        "mkdir -p {}/dev {}/proc {}/sys {}/tmp\n",
        rootfs_str, rootfs_str, rootfs_str, rootfs_str
    ));

    // GPU device bind mounts
    let gpu_mounts = build_gpu_mounts(config, &rootfs_str);
    script.push_str(&gpu_mounts);

    // User-specified bind mounts
    for mount in &config.mounts {
        let target = format!("{}{}", rootfs_str, mount.target);
        script.push_str(&format!("mkdir -p \"{}\"\n", target));
        let ro_flag = if mount.readonly { ",ro" } else { "" };
        script.push_str(&format!(
            "mount --bind \"{}\" \"{}\"\n",
            mount.source, target
        ));
        if mount.readonly {
            script.push_str(&format!("mount -o remount,bind,ro \"{}\"\n", target));
        }
    }

    // Copy the job script into the rootfs
    let container_script = format!("{}/tmp/spur_job_{}.sh", rootfs_str, job_id);
    script.push_str(&format!(
        "cp \"{}\" \"{}\"\nchmod +x \"{}\"\n",
        inner_script_path, container_script, container_script
    ));

    // Build environment exports
    for (key, value) in &config.environment {
        // Escape single quotes in values
        let escaped = value.replace('\'', "'\\''");
        script.push_str(&format!("export {}='{}'\n", key, escaped));
    }

    // Determine workdir inside container
    let workdir = config.workdir.as_deref().unwrap_or("/tmp");

    // Launch with unshare for namespace isolation, then chroot
    // unshare -m gives us a private mount namespace
    // We use chroot instead of pivot_root for simplicity (works without root
    // if we're in a user namespace, or if running as root)
    script.push_str(&format!(
        "\n# Enter container\nexec unshare --mount --map-root-user bash -c '\n\
         mount -t proc proc {rootfs}/proc\n\
         mount -t sysfs sys {rootfs}/sys\n\
         mount -t devtmpfs dev {rootfs}/dev 2>/dev/null || mount --bind /dev {rootfs}/dev\n\
         mount -t tmpfs tmpfs {rootfs}/tmp\n\
         cp /tmp/spur_job_{job_id}.sh {rootfs}/tmp/spur_job_{job_id}.sh 2>/dev/null || true\n\
         chroot {rootfs} /bin/bash -c \"cd {workdir} && /tmp/spur_job_{job_id}.sh\"\n\
         '\n",
        rootfs = rootfs_str,
        job_id = job_id,
        workdir = workdir,
    ));

    Ok(script)
}

/// Generate mount commands for GPU device passthrough.
fn build_gpu_mounts(config: &ContainerConfig, rootfs: &str) -> String {
    let mut script = String::new();

    // Always try to bind-mount /dev/dri if it exists (for GPU access)
    script.push_str(&format!(
        "if [ -d /dev/dri ]; then\n  mkdir -p {rootfs}/dev/dri\n  mount --bind /dev/dri {rootfs}/dev/dri\nfi\n",
        rootfs = rootfs
    ));

    // AMD: /dev/kfd is needed for ROCm
    script.push_str(&format!(
        "if [ -e /dev/kfd ]; then\n  touch {rootfs}/dev/kfd 2>/dev/null\n  mount --bind /dev/kfd {rootfs}/dev/kfd\nfi\n",
        rootfs = rootfs
    ));

    // AMD: bind-mount ROCm libraries if present
    for rocm_path in &["/opt/rocm", "/opt/rocm/lib"] {
        script.push_str(&format!(
            "if [ -d {rp} ]; then\n  mkdir -p {rootfs}{rp}\n  mount --bind {rp} {rootfs}{rp}\nfi\n",
            rp = rocm_path,
            rootfs = rootfs
        ));
    }

    // NVIDIA: bind-mount nvidia device files
    script.push_str(&format!(
        "for dev in /dev/nvidia*; do\n  if [ -e \"$dev\" ]; then\n    touch {rootfs}/$dev 2>/dev/null\n    mount --bind $dev {rootfs}/$dev\n  fi\ndone\n",
        rootfs = rootfs
    ));

    // NVIDIA: bind-mount driver libraries if present
    for nvidia_path in &[
        "/usr/lib/x86_64-linux-gnu/libnvidia",
        "/usr/lib64/libnvidia",
    ] {
        let dir = Path::new(nvidia_path)
            .parent()
            .unwrap_or(Path::new("/usr/lib"))
            .display();
        script.push_str(&format!(
            "if ls {np}* 1>/dev/null 2>&1; then\n  mkdir -p {rootfs}{dir}\n  for lib in {np}*; do\n    mount --bind $lib {rootfs}$lib\n  done\nfi\n",
            np = nvidia_path,
            rootfs = rootfs,
            dir = dir
        ));
    }

    // Set GPU visibility environment variables
    if !config.gpu_devices.is_empty() {
        let gpu_list: String = config
            .gpu_devices
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        script.push_str(&format!(
            "export ROCR_VISIBLE_DEVICES={gpu_list}\n\
             export CUDA_VISIBLE_DEVICES={gpu_list}\n\
             export GPU_DEVICE_ORDINAL={gpu_list}\n"
        ));
    }

    script
}

/// Parse a bind mount spec like "/src:/dst:ro" into a BindMount.
pub fn parse_mount(spec: &str) -> anyhow::Result<BindMount> {
    let parts: Vec<&str> = spec.split(':').collect();
    match parts.len() {
        2 => Ok(BindMount {
            source: parts[0].to_string(),
            target: parts[1].to_string(),
            readonly: false,
        }),
        3 => Ok(BindMount {
            source: parts[0].to_string(),
            target: parts[1].to_string(),
            readonly: parts[2].contains("ro"),
        }),
        _ => bail!("invalid mount spec '{}' — expected /src:/dst[:ro]", spec),
    }
}

/// Clean up an unnamed container rootfs.
pub fn cleanup_rootfs(job_id: u32) {
    let rootfs = PathBuf::from(CONTAINER_DIR).join(format!("job_{}", job_id));
    if rootfs.exists() {
        if let Err(e) = std::fs::remove_dir_all(&rootfs) {
            warn!(
                path = %rootfs.display(),
                error = %e,
                "failed to clean up container rootfs"
            );
        } else {
            debug!(path = %rootfs.display(), "container rootfs cleaned up");
        }
    }
}

/// Import a Docker/OCI image to squashfs format.
///
/// Uses skopeo + mksquashfs, or falls back to enroot if available.
pub async fn import_image(uri: &str) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(IMAGE_DIR)?;

    let name = sanitize_name(uri);
    let output_path = PathBuf::from(IMAGE_DIR).join(format!("{}.sqsh", name));

    if output_path.exists() {
        info!(image = %uri, path = %output_path.display(), "image already imported");
        return Ok(output_path);
    }

    // Try enroot first (handles docker:// URIs natively)
    if which("enroot") {
        info!(image = %uri, "importing with enroot");
        let status = Command::new("enroot")
            .args(["import", "--output", output_path.to_str().unwrap(), uri])
            .status()
            .await
            .context("failed to run enroot import")?;
        if status.success() {
            return Ok(output_path);
        }
        warn!("enroot import failed, trying manual method");
    }

    // Manual: skopeo copy → OCI dir → mksquashfs
    let tmp_dir = PathBuf::from(CONTAINER_DIR).join(format!("import_{}", name));
    let oci_dir = tmp_dir.join("oci");
    let rootfs_dir = tmp_dir.join("rootfs");
    std::fs::create_dir_all(&oci_dir)?;
    std::fs::create_dir_all(&rootfs_dir)?;

    // Normalize URI for skopeo
    let skopeo_src = if uri.starts_with("docker://") {
        uri.to_string()
    } else {
        format!("docker://{}", uri)
    };

    info!(image = %uri, "downloading image with skopeo");
    let status = Command::new("skopeo")
        .args(["copy", &skopeo_src, &format!("oci:{}", oci_dir.display())])
        .status()
        .await
        .context("failed to run skopeo — is skopeo installed?")?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        bail!("skopeo copy failed for {}", uri);
    }

    // Extract OCI layers to rootfs using umoci
    info!("extracting OCI layers with umoci");
    let status = Command::new("umoci")
        .args([
            "unpack",
            "--image",
            &format!("{}:latest", oci_dir.display()),
            rootfs_dir.to_str().unwrap(),
        ])
        .status()
        .await
        .context("failed to run umoci — is umoci installed?")?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        bail!("umoci unpack failed");
    }

    // Pack rootfs into squashfs
    let rootfs_content = rootfs_dir.join("rootfs");
    info!("creating squashfs image");
    let status = Command::new("mksquashfs")
        .args([
            rootfs_content.to_str().unwrap(),
            output_path.to_str().unwrap(),
            "-noappend",
            "-comp",
            "zstd",
        ])
        .status()
        .await
        .context("failed to run mksquashfs — is squashfs-tools installed?")?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        bail!("mksquashfs failed");
    }

    // Cleanup temp dir
    let _ = std::fs::remove_dir_all(&tmp_dir);

    info!(
        image = %uri,
        path = %output_path.display(),
        "image imported successfully"
    );
    Ok(output_path)
}

/// List imported images.
pub fn list_images() -> Vec<(String, u64)> {
    let dir = Path::new(IMAGE_DIR);
    if !dir.exists() {
        return Vec::new();
    }

    let mut images = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "sqsh") {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                images.push((name, size));
            }
        }
    }
    images.sort_by(|a, b| a.0.cmp(&b.0));
    images
}

/// Remove an imported image.
pub fn remove_image(name: &str) -> anyhow::Result<()> {
    let path = PathBuf::from(IMAGE_DIR).join(format!("{}.sqsh", sanitize_name(name)));
    if !path.exists() {
        bail!("image '{}' not found", name);
    }
    std::fs::remove_file(&path)?;
    info!(name, "image removed");
    Ok(())
}

/// Sanitize an image name for use as a filename.
fn sanitize_name(name: &str) -> String {
    name.replace("docker://", "")
        .replace('/', "+")
        .replace(':', "+")
}

/// Check if a binary is on PATH.
fn which(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mount() {
        let m = parse_mount("/data:/data").unwrap();
        assert_eq!(m.source, "/data");
        assert_eq!(m.target, "/data");
        assert!(!m.readonly);

        let m = parse_mount("/src:/dst:ro").unwrap();
        assert_eq!(m.source, "/src");
        assert_eq!(m.target, "/dst");
        assert!(m.readonly);

        assert!(parse_mount("/only-one-part").is_err());
    }

    #[test]
    fn test_sanitize_name() {
        assert_eq!(
            sanitize_name("docker://nvcr.io/nvidia/pytorch:24.01"),
            "nvcr.io+nvidia+pytorch+24.01"
        );
        assert_eq!(sanitize_name("ubuntu:22.04"), "ubuntu+22.04");
    }

    #[test]
    fn test_build_gpu_mounts() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            gpu_devices: vec![0, 1],
            environment: HashMap::new(),
        };
        let script = build_gpu_mounts(&config, "/tmp/rootfs");
        assert!(script.contains("/dev/dri"));
        assert!(script.contains("/dev/kfd"));
        assert!(script.contains("ROCR_VISIBLE_DEVICES=0,1"));
        assert!(script.contains("CUDA_VISIBLE_DEVICES=0,1"));
    }
}
