//! `spur image` subcommands for container image management.

use std::process::Stdio;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

/// Container image management.
#[derive(Parser, Debug)]
#[command(name = "image", about = "Manage container images")]
pub struct ImageArgs {
    #[command(subcommand)]
    pub command: ImageCommand,
}

#[derive(Subcommand, Debug)]
pub enum ImageCommand {
    /// Import a container image as squashfs.
    ///
    /// Supports Docker/OCI registries, local Docker daemon, and Podman.
    Import {
        /// Image URI. Formats:
        ///   ubuntu:22.04                    — Docker Hub
        ///   nvcr.io/nvidia/pytorch:24.01    — custom registry
        ///   docker://image:tag              — explicit Docker registry
        ///   dockerd://image:tag             — local Docker daemon
        ///   podman://image:tag              — local Podman
        image: String,

        /// Target architecture (default: amd64)
        #[arg(short = 'a', long, default_value = "amd64")]
        arch: String,
    },
    /// List imported images.
    List,
    /// Remove an imported image.
    Remove {
        /// Image name
        name: String,
    },
    /// Export a named container back to squashfs.
    Export {
        /// Container name
        name: String,
        /// Output file (default: <name>.sqsh)
        #[arg(short = 'o', long)]
        output: Option<String>,
    },
}

pub async fn main() -> Result<()> {
    let args = ImageArgs::parse();

    match args.command {
        ImageCommand::Import { image, arch } => cmd_import(&image, &arch).await,
        ImageCommand::List => cmd_list(),
        ImageCommand::Remove { name } => cmd_remove(&name),
        ImageCommand::Export { name, output } => cmd_export(&name, output.as_deref()),
    }
}

async fn cmd_import(image: &str, arch: &str) -> Result<()> {
    // Handle local Docker daemon import
    if let Some(docker_image) = image.strip_prefix("dockerd://") {
        return cmd_import_dockerd(docker_image).await;
    }

    // Handle local Podman import
    if let Some(podman_image) = image.strip_prefix("podman://") {
        return cmd_import_podman(podman_image).await;
    }

    // Registry import (native OCI puller)
    let image_ref = spur_net::oci::parse_image_ref(image);
    eprintln!(
        "Importing {}:{} from {} (arch: {})",
        image_ref.repository, image_ref.tag, image_ref.registry, arch
    );

    let image_dir = std::path::Path::new("/var/spool/spur/images");
    let path = spur_net::pull_image(image, image_dir).await?;

    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "Imported: {} ({:.1} MB)",
        path.display(),
        size as f64 / 1_048_576.0
    );

    Ok(())
}

/// Import from local Docker daemon via `docker save`.
async fn cmd_import_dockerd(image: &str) -> Result<()> {
    eprintln!("Importing from Docker daemon: {}", image);

    let name = spur_net::oci::sanitize_name(image);
    let output_path = format!("/var/spool/spur/images/{}.sqsh", name);
    if std::path::Path::new(&output_path).exists() {
        eprintln!("Image already imported: {}", output_path);
        return Ok(());
    }

    std::fs::create_dir_all("/var/spool/spur/images")?;
    let tmp_dir = format!("/var/spool/spur/containers/.import_dockerd_{}", name);
    let rootfs = format!("{}/rootfs", tmp_dir);
    std::fs::create_dir_all(&rootfs)?;

    // docker save → tar, then extract layers
    let output = tokio::process::Command::new("docker")
        .args(["save", image])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("failed to run docker — is Docker installed and running?")?;

    if !output.status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("docker save failed: {}", stderr.trim());
    }

    // Extract the docker save tar, then extract each layer
    extract_docker_save_tar(&output.stdout, &rootfs)?;

    // Pack into squashfs
    pack_squashfs(&rootfs, &output_path)?;
    let _ = std::fs::remove_dir_all(&tmp_dir);

    let size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "Imported: {} ({:.1} MB)",
        output_path,
        size as f64 / 1_048_576.0
    );
    Ok(())
}

/// Import from local Podman via `podman save`.
async fn cmd_import_podman(image: &str) -> Result<()> {
    eprintln!("Importing from Podman: {}", image);

    let name = spur_net::oci::sanitize_name(image);
    let output_path = format!("/var/spool/spur/images/{}.sqsh", name);
    if std::path::Path::new(&output_path).exists() {
        eprintln!("Image already imported: {}", output_path);
        return Ok(());
    }

    std::fs::create_dir_all("/var/spool/spur/images")?;
    let tmp_dir = format!("/var/spool/spur/containers/.import_podman_{}", name);
    let rootfs = format!("{}/rootfs", tmp_dir);
    std::fs::create_dir_all(&rootfs)?;

    let output = tokio::process::Command::new("podman")
        .args(["save", image])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("failed to run podman — is Podman installed?")?;

    if !output.status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("podman save failed: {}", stderr.trim());
    }

    extract_docker_save_tar(&output.stdout, &rootfs)?;
    pack_squashfs(&rootfs, &output_path)?;
    let _ = std::fs::remove_dir_all(&tmp_dir);

    let size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "Imported: {} ({:.1} MB)",
        output_path,
        size as f64 / 1_048_576.0
    );
    Ok(())
}

/// Extract a `docker save` tar archive into a rootfs.
/// The tar contains a manifest.json listing layer tarballs.
fn extract_docker_save_tar(tar_data: &[u8], rootfs: &str) -> Result<()> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let dest = std::path::Path::new(rootfs);
    let mut archive = tar::Archive::new(tar_data);
    let tmp = format!("{}/.docker_save", rootfs);
    std::fs::create_dir_all(&tmp)?;

    // First pass: extract all files from the docker save tar
    archive.set_overwrite(true);
    archive
        .unpack(&tmp)
        .context("failed to extract docker save archive")?;

    // Parse manifest.json to find layer order
    let manifest_path = format!("{}/manifest.json", tmp);
    let manifest_str =
        std::fs::read_to_string(&manifest_path).context("no manifest.json in docker save")?;
    let manifest: Vec<serde_json::Value> =
        serde_json::from_str(&manifest_str).context("invalid manifest.json")?;

    let layers = manifest
        .first()
        .and_then(|m| m.get("Layers"))
        .and_then(|l| l.as_array())
        .ok_or_else(|| anyhow::anyhow!("no layers in manifest.json"))?;

    // Extract each layer in order
    for layer_path in layers {
        let layer_file = layer_path
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("invalid layer path"))?;
        let full_path = format!("{}/{}", tmp, layer_file);
        let data = std::fs::read(&full_path)
            .with_context(|| format!("failed to read layer {}", layer_file))?;

        // Try gzip decompress, fall back to plain tar
        let result = if data.starts_with(&[0x1f, 0x8b]) {
            let decoder = GzDecoder::new(data.as_slice());
            let mut archive = tar::Archive::new(decoder);
            archive.set_overwrite(true);
            archive.unpack(dest)
        } else {
            let mut archive = tar::Archive::new(data.as_slice());
            archive.set_overwrite(true);
            archive.unpack(dest)
        };

        if let Err(e) = result {
            eprintln!("Warning: layer {} extraction had errors: {}", layer_file, e);
        }
    }

    // Clean up docker save temp files
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

/// Pack a directory into a squashfs file.
fn pack_squashfs(rootfs: &str, output: &str) -> Result<()> {
    let result = std::process::Command::new("mksquashfs")
        .args([rootfs, output, "-noappend", "-comp", "zstd", "-quiet"])
        .output();

    match result {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            bail!("mksquashfs failed: {}", stderr.trim())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!(
                "mksquashfs not found. Install squashfs-tools:\n  \
                 sudo apt install squashfs-tools    # Debian/Ubuntu\n  \
                 sudo dnf install squashfs-tools    # Fedora/RHEL"
            )
        }
        Err(e) => bail!("failed to run mksquashfs: {}", e),
    }
}

fn cmd_list() -> Result<()> {
    let image_dir = std::path::Path::new("/var/spool/spur/images");
    if !image_dir.exists() {
        eprintln!("No images imported yet.");
        return Ok(());
    }

    let mut images: Vec<(String, u64)> = Vec::new();
    for entry in std::fs::read_dir(image_dir)?.flatten() {
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

    if images.is_empty() {
        eprintln!("No images imported yet.");
        return Ok(());
    }

    images.sort_by(|a, b| a.0.cmp(&b.0));

    println!("{:<50} {:>10}", "IMAGE", "SIZE");
    for (name, size) in &images {
        let display_name = name.replace('+', "/");
        let size_str = if *size > 1_073_741_824 {
            format!("{:.1} GB", *size as f64 / 1_073_741_824.0)
        } else {
            format!("{:.1} MB", *size as f64 / 1_048_576.0)
        };
        println!("{:<50} {:>10}", display_name, size_str);
    }

    Ok(())
}

fn cmd_remove(name: &str) -> Result<()> {
    let sanitized = spur_net::oci::sanitize_name(name);
    let path = format!("/var/spool/spur/images/{}.sqsh", sanitized);

    if !std::path::Path::new(&path).exists() {
        bail!("image '{}' not found", name);
    }

    std::fs::remove_file(&path)?;
    eprintln!("Removed: {}", name);
    Ok(())
}

/// Export a named container's rootfs back to a squashfs image.
fn cmd_export(name: &str, output: Option<&str>) -> Result<()> {
    let sanitized = spur_net::oci::sanitize_name(name);
    let container_dir = format!("/var/spool/spur/containers/{}", sanitized);

    if !std::path::Path::new(&container_dir).exists() {
        bail!(
            "container '{}' not found. Only named containers (--container-name) can be exported.",
            name
        );
    }

    let output_path = output
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("/var/spool/spur/images/{}.sqsh", sanitized));

    eprintln!("Exporting container '{}' to {}", name, output_path);
    pack_squashfs(&container_dir, &output_path)?;

    let size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "Exported: {} ({:.1} MB)",
        output_path,
        size as f64 / 1_048_576.0
    );
    Ok(())
}
