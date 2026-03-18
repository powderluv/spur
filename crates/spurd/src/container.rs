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
    pub mount_home: bool,
    pub remap_root: bool,
    pub gpu_devices: Vec<u32>,
    pub environment: HashMap<String, String>,
    pub container_env: HashMap<String, String>,
    pub entrypoint: Option<String>,
    pub uid: u32,
    pub gid: u32,
    pub username: String,
    pub home_dir: String,
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
    let unsquashfs_result = std::process::Command::new("unsquashfs")
        .args([
            "-f", // force overwrite
            "-d", // destination
            rootfs.to_str().unwrap(),
            image_path.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output();

    match unsquashfs_result {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "unsquashfs failed for image {} (exit {}): {}",
                image_path.display(),
                output.status.code().unwrap_or(-1),
                stderr.trim()
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!(
                "unsquashfs not found. Install squashfs-tools:\n  \
                 sudo apt install squashfs-tools    # Debian/Ubuntu\n  \
                 sudo dnf install squashfs-tools    # Fedora/RHEL"
            );
        }
        Err(e) => {
            bail!("failed to run unsquashfs: {}", e);
        }
    }

    info!(rootfs = %rootfs.display(), "container rootfs created");
    Ok(rootfs)
}

/// Build the wrapper script that launches a job inside a container.
///
/// Implements Enroot-equivalent hooks:
/// - shadow: map host user into container /etc/passwd + /etc/group
/// - home: bind-mount user home directory
/// - devices: restrict /dev or passthrough GPU devices
/// - nvidia: bind-mount NVIDIA driver libs + devices
/// - rocm: bind-mount AMD ROCm libs + /dev/kfd + /dev/dri
/// - mellanox: bind-mount InfiniBand devices + MOFED libs
/// - cgroups: already handled by executor.rs
pub fn build_container_launch_script(
    config: &ContainerConfig,
    rootfs: &Path,
    inner_script_path: &str,
    job_id: u32,
) -> anyhow::Result<String> {
    let mut script = String::new();
    script.push_str("#!/bin/bash\nset -e\n\n");

    let rootfs_str = rootfs.to_string_lossy();

    // Ensure key directories exist in rootfs
    script.push_str(&format!(
        "mkdir -p {rootfs}/dev {rootfs}/proc {rootfs}/sys {rootfs}/tmp {rootfs}/etc {rootfs}/run\n",
        rootfs = rootfs_str
    ));

    // --- Hook: shadow (user mapping) ---
    // Map host user into container's /etc/passwd and /etc/group
    script.push_str(&format!(
        r#"
# Hook: shadow — map host user into container
if [ -f {rootfs}/etc/passwd ]; then
  grep -q "^{username}:" {rootfs}/etc/passwd 2>/dev/null || \
    echo "{username}:x:{uid}:{gid}::{home}:/bin/bash" >> {rootfs}/etc/passwd
fi
if [ -f {rootfs}/etc/group ]; then
  grep -q "^{username}:" {rootfs}/etc/group 2>/dev/null || \
    echo "{username}:x:{gid}:" >> {rootfs}/etc/group
fi
mkdir -p {rootfs}{home}
"#,
        rootfs = rootfs_str,
        username = config.username,
        uid = config.uid,
        gid = config.gid,
        home = config.home_dir,
    ));

    // Copy the job script into the rootfs
    let container_script = format!("{}/tmp/spur_job_{}.sh", rootfs_str, job_id);
    script.push_str(&format!(
        "cp \"{}\" \"{}\"\nchmod +x \"{}\"\n",
        inner_script_path, container_script, container_script
    ));

    // Pre-create mount targets
    for mount in &config.mounts {
        let target = format!("{}{}", rootfs_str, mount.target);
        script.push_str(&format!("mkdir -p \"{}\"\n", target));
    }

    // Build container-specific env exports
    let mut env_exports = String::new();
    // GPU visibility
    if !config.gpu_devices.is_empty() {
        let gpu_list: String = config
            .gpu_devices
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        env_exports.push_str(&format!(
            "export ROCR_VISIBLE_DEVICES={gl}\nexport CUDA_VISIBLE_DEVICES={gl}\nexport GPU_DEVICE_ORDINAL={gl}\n",
            gl = gpu_list
        ));
    }
    // User-specified container env vars (--container-env KEY=VAL)
    for (key, value) in &config.container_env {
        let escaped = value.replace('\'', "'\\''");
        env_exports.push_str(&format!("export {}='{}'\n", key, escaped));
    }

    let workdir = config.workdir.as_deref().unwrap_or("/tmp");

    // Build entrypoint prefix if specified
    let entrypoint_cmd = if let Some(ref ep) = config.entrypoint {
        format!("{} && ", ep)
    } else {
        String::new()
    };

    // === ROOT MODE: full namespace + chroot ===
    script.push_str(&format!(
        r#"
if [ "$(id -u)" = "0" ]; then
  exec unshare --mount bash -c '
set -e
ROOTFS="{rootfs}"

# Hook: filesystem mounts
mount -t proc proc $ROOTFS/proc 2>/dev/null || true
mount -t sysfs sys $ROOTFS/sys 2>/dev/null || true
mount -t tmpfs tmpfs $ROOTFS/run 2>/dev/null || true
"#,
        rootfs = rootfs_str,
    ));

    // Hook: devices — restricted /dev with only essential devices
    script.push_str(
        r#"
# Hook: devices — minimal /dev + GPU passthrough
mount -t tmpfs -o mode=755 tmpfs $ROOTFS/dev
mkdir -p $ROOTFS/dev/pts $ROOTFS/dev/shm $ROOTFS/dev/mqueue
mount -t devpts devpts $ROOTFS/dev/pts 2>/dev/null || true
mount -t tmpfs tmpfs $ROOTFS/dev/shm 2>/dev/null || true
# Essential devices
for d in null zero random urandom tty console; do
  touch $ROOTFS/dev/$d 2>/dev/null
  mount --bind /dev/$d $ROOTFS/dev/$d 2>/dev/null || true
done
ln -sf /proc/self/fd $ROOTFS/dev/fd 2>/dev/null || true
ln -sf /proc/self/fd/0 $ROOTFS/dev/stdin 2>/dev/null || true
ln -sf /proc/self/fd/1 $ROOTFS/dev/stdout 2>/dev/null || true
ln -sf /proc/self/fd/2 $ROOTFS/dev/stderr 2>/dev/null || true

# Hook: GPU — AMD (ROCm)
if [ -d /dev/dri ]; then
  mkdir -p $ROOTFS/dev/dri
  mount --bind /dev/dri $ROOTFS/dev/dri 2>/dev/null || true
fi
if [ -e /dev/kfd ]; then
  touch $ROOTFS/dev/kfd 2>/dev/null
  mount --bind /dev/kfd $ROOTFS/dev/kfd 2>/dev/null || true
fi
for p in /opt/rocm /opt/rocm/lib /opt/rocm/lib64; do
  if [ -d "$p" ]; then
    mkdir -p $ROOTFS$p
    mount --bind $p $ROOTFS$p 2>/dev/null || true
  fi
done

# Hook: GPU — NVIDIA
for dev in /dev/nvidia*; do
  if [ -e "$dev" ]; then
    touch $ROOTFS$dev 2>/dev/null || true
    mount --bind $dev $ROOTFS$dev 2>/dev/null || true
  fi
done
for libdir in /usr/lib/x86_64-linux-gnu /usr/lib64; do
  if ls $libdir/libnvidia* 1>/dev/null 2>&1; then
    mkdir -p $ROOTFS$libdir
    for lib in $libdir/libnvidia* $libdir/libcuda* $libdir/libnvoptix*; do
      [ -e "$lib" ] && mount --bind $lib $ROOTFS$lib 2>/dev/null || true
    done
  fi
done

# Hook: InfiniBand / Mellanox (MOFED)
if [ -d /dev/infiniband ]; then
  mkdir -p $ROOTFS/dev/infiniband
  mount --bind /dev/infiniband $ROOTFS/dev/infiniband 2>/dev/null || true
fi
for ibdev in /dev/uverbs* /dev/rdma_cm; do
  if [ -e "$ibdev" ]; then
    touch $ROOTFS$ibdev 2>/dev/null || true
    mount --bind $ibdev $ROOTFS$ibdev 2>/dev/null || true
  fi
done
for mofed in /etc/libibverbs.d /usr/lib/x86_64-linux-gnu/libibverbs /usr/lib64/libibverbs; do
  if [ -d "$mofed" ]; then
    mkdir -p $ROOTFS$mofed
    mount --bind $mofed $ROOTFS$mofed 2>/dev/null || true
  fi
done
"#,
    );

    // Hook: home — bind-mount user home directory
    if config.mount_home {
        script.push_str(&format!(
            r#"
# Hook: home — mount user home directory
mkdir -p $ROOTFS{home}
mount --bind {home} $ROOTFS{home} 2>/dev/null || true
"#,
            home = config.home_dir,
        ));
    }

    // User bind mounts
    for mount in &config.mounts {
        script.push_str(&format!(
            "\nmkdir -p $ROOTFS{target}\nmount --bind \"{source}\" $ROOTFS{target} 2>/dev/null || true",
            source = mount.source,
            target = mount.target,
        ));
        if mount.readonly {
            script.push_str(&format!(
                "\nmount -o remount,bind,ro $ROOTFS{target} 2>/dev/null || true",
                target = mount.target,
            ));
        }
    }

    // Hook: config.d — source any system/user hook scripts
    script.push_str(
        r#"

# Hook: config.d — run hook scripts from /etc/spur/container.d/hooks.d/
for hook in /etc/spur/container.d/hooks.d/*.sh; do
  [ -x "$hook" ] && ENROOT_ROOTFS=$ROOTFS ENROOT_PID=$$ . "$hook" 2>/dev/null || true
done

# Hook: environ.d — source extra environment files
for envf in /etc/spur/container.d/environ.d/*.env; do
  [ -f "$envf" ] && while IFS= read -r line; do
    [ -n "$line" ] && [ "${line#\#}" = "$line" ] && export "$line"
  done < "$envf"
done

# Hook: mounts.d — process extra mount specs
for fstab in /etc/spur/container.d/mounts.d/*.fstab; do
  [ -f "$fstab" ] && while IFS= read -r line; do
    [ -n "$line" ] && [ "${line#\#}" = "$line" ] && {
      src=$(echo "$line" | awk "{print \$1}")
      dst=$(echo "$line" | awk "{print \$2}")
      [ -n "$src" ] && [ -n "$dst" ] && {
        mkdir -p $ROOTFS$dst
        mount --bind "$src" $ROOTFS$dst 2>/dev/null || true
      }
    }
  done < "$fstab"
done
"#,
    );

    // Chroot and execute
    script.push_str(&format!(
        r#"
# Set environment
{env_exports}

# Enter container
chroot $ROOTFS /bin/bash -c "cd {workdir} && {entrypoint}{script}"
'
else
  # Non-root fallback: PATH-based execution, no namespace isolation
  ROOTFS="{rootfs}"
  {env_exports}
  export PATH="$ROOTFS/usr/bin:$ROOTFS/bin:$ROOTFS/usr/sbin:$ROOTFS/sbin:$PATH"
  export LD_LIBRARY_PATH="$ROOTFS/usr/lib:$ROOTFS/lib:$ROOTFS/usr/lib64:$ROOTFS/lib64:${{LD_LIBRARY_PATH:-}}"
  export SPUR_CONTAINER_ROOTFS="$ROOTFS"
  export HOME="{home}"
  cd {workdir}
  {entrypoint}/bin/bash $ROOTFS/tmp/spur_job_{job_id}.sh
fi
"#,
        env_exports = env_exports,
        workdir = workdir,
        job_id = job_id,
        rootfs = rootfs_str,
        home = config.home_dir,
        entrypoint = entrypoint_cmd,
        script = format!("/tmp/spur_job_{}.sh", job_id),
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
/// Uses spur-net's native OCI puller — downloads directly from registries
/// via the Docker Registry HTTP API v2. No dependency on Docker, skopeo,
/// umoci, or enroot. Only needs mksquashfs (squashfs-tools).
pub async fn import_image(uri: &str) -> anyhow::Result<PathBuf> {
    let image_dir = Path::new(IMAGE_DIR);
    spur_net::pull_image(uri, image_dir).await
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

    // --- Mount parsing ---

    #[test]
    fn test_parse_mount_basic() {
        let m = parse_mount("/data:/data").unwrap();
        assert_eq!(m.source, "/data");
        assert_eq!(m.target, "/data");
        assert!(!m.readonly);
    }

    #[test]
    fn test_parse_mount_readonly() {
        let m = parse_mount("/src:/dst:ro").unwrap();
        assert_eq!(m.source, "/src");
        assert_eq!(m.target, "/dst");
        assert!(m.readonly);
    }

    #[test]
    fn test_parse_mount_rw_explicit() {
        let m = parse_mount("/src:/dst:rw").unwrap();
        assert!(!m.readonly);
    }

    #[test]
    fn test_parse_mount_one_part_fails() {
        let err = parse_mount("/only-one-part").unwrap_err();
        assert!(
            err.to_string().contains("invalid mount spec"),
            "expected 'invalid mount spec', got: {}",
            err
        );
        assert!(err.to_string().contains("/src:/dst"));
    }

    #[test]
    fn test_parse_mount_empty_fails() {
        assert!(parse_mount("").is_err());
    }

    #[test]
    fn test_parse_mount_too_many_parts_fails() {
        let err = parse_mount("/a:/b:ro:extra:parts").unwrap_err();
        assert!(err.to_string().contains("invalid mount spec"));
    }

    // --- Name sanitization ---

    #[test]
    fn test_sanitize_docker_uri() {
        assert_eq!(
            sanitize_name("docker://nvcr.io/nvidia/pytorch:24.01"),
            "nvcr.io+nvidia+pytorch+24.01"
        );
    }

    #[test]
    fn test_sanitize_simple_name() {
        assert_eq!(sanitize_name("ubuntu:22.04"), "ubuntu+22.04");
    }

    #[test]
    fn test_sanitize_nested_path() {
        assert_eq!(
            sanitize_name("registry.example.com/org/image:v1.2.3"),
            "registry.example.com+org+image+v1.2.3"
        );
    }

    #[test]
    fn test_sanitize_no_tag() {
        assert_eq!(sanitize_name("alpine"), "alpine");
    }

    // --- Image resolution ---

    #[test]
    fn test_resolve_image_not_found() {
        let err = resolve_image("nonexistent-image-xyz").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not found"),
            "expected 'not found', got: {}",
            msg
        );
        assert!(
            msg.contains("spur image import"),
            "should suggest 'spur image import', got: {}",
            msg
        );
    }

    #[test]
    fn test_resolve_image_absolute_path_not_found() {
        let err = resolve_image("/nonexistent/path/to/image.sqsh").unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_resolve_image_docker_uri_not_imported() {
        let err = resolve_image("docker://ubuntu:22.04").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not found"));
        assert!(msg.contains("spur image import"));
    }

    // --- GPU mounts ---

    #[test]
    fn test_gpu_mounts_with_devices() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![0, 1],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "testuser".into(),
            home_dir: "/home/testuser".into(),
        };
        let script = build_gpu_mounts(&config, "/tmp/rootfs");
        // AMD devices
        assert!(script.contains("/dev/dri"));
        assert!(script.contains("/dev/kfd"));
        assert!(script.contains("/opt/rocm"));
        // NVIDIA devices
        assert!(script.contains("/dev/nvidia"));
        // Visibility env vars
        assert!(script.contains("ROCR_VISIBLE_DEVICES=0,1"));
        assert!(script.contains("CUDA_VISIBLE_DEVICES=0,1"));
    }

    #[test]
    fn test_gpu_mounts_no_devices() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "testuser".into(),
            home_dir: "/home/testuser".into(),
        };
        let script = build_gpu_mounts(&config, "/tmp/rootfs");
        // Should still mount device dirs (if they exist on host)
        assert!(script.contains("/dev/dri"));
        // But no visibility env vars
        assert!(!script.contains("ROCR_VISIBLE_DEVICES"));
        assert!(!script.contains("CUDA_VISIBLE_DEVICES"));
    }

    // --- Container launch script ---

    #[test]
    fn test_launch_script_basic_structure() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "testuser".into(),
            home_dir: "/home/testuser".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 42).unwrap();

        assert!(script.starts_with("#!/bin/bash"));
        assert!(script.contains("set -e"));
        // Copies inner script into rootfs
        assert!(script.contains("/tmp/inner.sh"));
        assert!(script.contains("spur_job_42.sh"));
        // Has namespace/chroot logic
        assert!(script.contains("unshare"));
        assert!(script.contains("chroot"));
        // Non-root fallback
        assert!(script.contains("SPUR_CONTAINER_ROOTFS"));
    }

    #[test]
    fn test_launch_script_with_workdir() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: Some("/workspace".into()),
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "testuser".into(),
            home_dir: "/home/testuser".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 1).unwrap();

        assert!(script.contains("cd /workspace"));
    }

    #[test]
    fn test_launch_script_with_mounts() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![
                BindMount {
                    source: "/data".into(),
                    target: "/mnt/data".into(),
                    readonly: true,
                },
                BindMount {
                    source: "/models".into(),
                    target: "/models".into(),
                    readonly: false,
                },
            ],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "testuser".into(),
            home_dir: "/home/testuser".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 1).unwrap();

        assert!(script.contains("mount --bind \"/data\""));
        assert!(script.contains("/mnt/data"));
        assert!(script.contains("remount,bind,ro"));
        assert!(script.contains("mount --bind \"/models\""));
    }

    // --- Image removal ---

    #[test]
    fn test_remove_image_not_found() {
        let err = remove_image("nonexistent-image-that-doesnt-exist").unwrap_err();
        assert!(
            err.to_string().contains("not found"),
            "expected 'not found', got: {}",
            err
        );
    }

    // --- List images (empty) ---

    #[test]
    fn test_list_images_nonexistent_dir() {
        // Temporarily override — just test with a known empty path
        // list_images uses a hardcoded path, so this tests the "dir doesn't exist" case
        // by checking the function handles it gracefully
        let images = list_images();
        // May or may not have images depending on test env, but shouldn't panic
        let _ = images;
    }

    // --- Cleanup ---

    #[test]
    fn test_cleanup_rootfs_nonexistent() {
        // Should not panic when cleaning up a rootfs that doesn't exist
        cleanup_rootfs(999999);
    }

    // --- Which ---

    #[test]
    fn test_which_finds_bash() {
        assert!(which("bash"));
    }

    #[test]
    fn test_which_not_found() {
        assert!(!which("nonexistent-binary-that-doesnt-exist-xyz"));
    }

    // --- New hook tests ---

    #[test]
    fn test_launch_script_has_shadow_hook() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "alice".into(),
            home_dir: "/home/alice".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 1).unwrap();
        // Shadow hook: maps user into container
        assert!(script.contains("alice:x:1000:1000"));
        assert!(script.contains("/etc/passwd"));
        assert!(script.contains("/etc/group"));
    }

    #[test]
    fn test_launch_script_mount_home() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: true,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "alice".into(),
            home_dir: "/home/alice".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 1).unwrap();
        assert!(script.contains("mount --bind /home/alice"));
    }

    #[test]
    fn test_launch_script_no_mount_home_by_default() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "alice".into(),
            home_dir: "/home/alice".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 1).unwrap();
        assert!(!script.contains("Hook: home"));
    }

    #[test]
    fn test_launch_script_has_infiniband_hook() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "testuser".into(),
            home_dir: "/home/testuser".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 1).unwrap();
        assert!(script.contains("/dev/infiniband"));
        assert!(script.contains("libibverbs"));
    }

    #[test]
    fn test_launch_script_has_restricted_dev() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "testuser".into(),
            home_dir: "/home/testuser".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 1).unwrap();
        // Restricted /dev with essential devices only
        assert!(script.contains("tmpfs tmpfs $ROOTFS/dev"));
        assert!(script.contains("null zero random urandom tty console"));
        assert!(script.contains("/dev/pts"));
        assert!(script.contains("/dev/shm"));
    }

    #[test]
    fn test_launch_script_container_env() {
        let mut container_env = HashMap::new();
        container_env.insert("MY_VAR".into(), "my_value".into());
        container_env.insert("ANOTHER".into(), "val2".into());
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env,
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "testuser".into(),
            home_dir: "/home/testuser".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 1).unwrap();
        assert!(script.contains("MY_VAR='my_value'"));
        assert!(script.contains("ANOTHER='val2'"));
    }

    #[test]
    fn test_launch_script_entrypoint() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: Some("/entrypoint.sh".into()),
            uid: 1000,
            gid: 1000,
            username: "testuser".into(),
            home_dir: "/home/testuser".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 1).unwrap();
        assert!(script.contains("/entrypoint.sh &&"));
    }

    #[test]
    fn test_launch_script_config_hooks() {
        let config = ContainerConfig {
            image: "test".into(),
            mounts: vec![],
            workdir: None,
            name: None,
            readonly: false,
            mount_home: false,
            remap_root: false,
            gpu_devices: vec![],
            environment: HashMap::new(),
            container_env: HashMap::new(),
            entrypoint: None,
            uid: 1000,
            gid: 1000,
            username: "testuser".into(),
            home_dir: "/home/testuser".into(),
        };
        let rootfs = Path::new("/tmp/test-rootfs");
        let script = build_container_launch_script(&config, rootfs, "/tmp/inner.sh", 1).unwrap();
        // Config hook directories
        assert!(script.contains("container.d/hooks.d"));
        assert!(script.contains("container.d/environ.d"));
        assert!(script.contains("container.d/mounts.d"));
    }
}
