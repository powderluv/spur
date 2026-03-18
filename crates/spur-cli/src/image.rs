//! `spur image` subcommands for container image management.

use anyhow::{Context, Result};
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
    /// Import a Docker/OCI image as squashfs.
    ///
    /// Downloads the image and converts it to a squashfs file that can be
    /// used with --container-image in job submissions.
    Import {
        /// Image URI (e.g., "docker://nvcr.io/nvidia/pytorch:24.01", "ubuntu:22.04")
        image: String,
    },
    /// List imported images.
    List,
    /// Remove an imported image.
    Remove {
        /// Image name
        name: String,
    },
}

pub async fn main() -> Result<()> {
    let args = ImageArgs::parse();

    match args.command {
        ImageCommand::Import { image } => cmd_import(&image).await,
        ImageCommand::List => cmd_list(),
        ImageCommand::Remove { name } => cmd_remove(&name),
    }
}

async fn cmd_import(image: &str) -> Result<()> {
    eprintln!("Importing image: {}", image);

    // Use the container module from spurd (shared logic)
    // For now, call the tools directly since spur-cli doesn't depend on spurd
    let image_dir = "/var/spool/spur/images";
    std::fs::create_dir_all(image_dir)?;

    let name = image
        .replace("docker://", "")
        .replace('/', "+")
        .replace(':', "+");
    let output_path = format!("{}/{}.sqsh", image_dir, name);

    if std::path::Path::new(&output_path).exists() {
        eprintln!("Image already imported: {}", output_path);
        return Ok(());
    }

    // Try enroot first
    let enroot_status = tokio::process::Command::new("enroot")
        .args(["import", "--output", &output_path, image])
        .status()
        .await;

    if let Ok(status) = enroot_status {
        if status.success() {
            let size = std::fs::metadata(&output_path)
                .map(|m| m.len())
                .unwrap_or(0);
            eprintln!(
                "Imported: {} ({:.1} MB)",
                output_path,
                size as f64 / 1_048_576.0
            );
            return Ok(());
        }
    }

    // Fallback: skopeo + umoci + mksquashfs
    let tmp_dir = format!("/var/spool/spur/containers/import_{}", name);
    let oci_dir = format!("{}/oci", tmp_dir);
    let rootfs_dir = format!("{}/rootfs", tmp_dir);
    std::fs::create_dir_all(&oci_dir)?;
    std::fs::create_dir_all(&rootfs_dir)?;

    let skopeo_src = if image.starts_with("docker://") {
        image.to_string()
    } else {
        format!("docker://{}", image)
    };

    eprintln!("Downloading with skopeo...");
    let status = tokio::process::Command::new("skopeo")
        .args(["copy", &skopeo_src, &format!("oci:{}", oci_dir)])
        .status()
        .await
        .context("failed to run skopeo — install with: sudo apt install skopeo")?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        anyhow::bail!("skopeo copy failed");
    }

    eprintln!("Extracting layers with umoci...");
    let status = tokio::process::Command::new("umoci")
        .args([
            "unpack",
            "--image",
            &format!("{}:latest", oci_dir),
            &rootfs_dir,
        ])
        .status()
        .await
        .context("failed to run umoci — install with: sudo apt install umoci")?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        anyhow::bail!("umoci unpack failed");
    }

    eprintln!("Creating squashfs...");
    let rootfs_content = format!("{}/rootfs", rootfs_dir);
    let status = tokio::process::Command::new("mksquashfs")
        .args([&rootfs_content, &output_path, "-noappend", "-comp", "zstd"])
        .status()
        .await
        .context("failed to run mksquashfs — install with: sudo apt install squashfs-tools")?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        anyhow::bail!("mksquashfs failed");
    }

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
    let sanitized = name
        .replace("docker://", "")
        .replace('/', "+")
        .replace(':', "+");
    let path = format!("/var/spool/spur/images/{}.sqsh", sanitized);

    if !std::path::Path::new(&path).exists() {
        anyhow::bail!("image '{}' not found", name);
    }

    std::fs::remove_file(&path)?;
    eprintln!("Removed: {}", name);
    Ok(())
}
