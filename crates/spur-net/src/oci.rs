//! Native OCI/Docker image puller.
//!
//! Downloads container images directly from registries using the
//! Docker Registry HTTP API v2. No dependency on Docker, skopeo,
//! umoci, or enroot.
//!
//! Flow:
//! 1. Parse image reference (registry/repo:tag)
//! 2. Authenticate (token-based for Docker Hub, anonymous for others)
//! 3. Fetch manifest → list of layer digests
//! 4. Download each layer (tar.gz blobs)
//! 5. Extract layers in order to build rootfs
//! 6. Pack rootfs into squashfs via mksquashfs

use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use flate2::read::GzDecoder;
use reqwest::header::{ACCEPT, AUTHORIZATION};
use serde::Deserialize;
use tracing::{debug, info};

/// A parsed container image reference.
#[derive(Debug, Clone)]
pub struct ImageRef {
    pub registry: String,
    pub repository: String,
    pub tag: String,
}

/// Docker Registry auth token response.
#[derive(Deserialize)]
struct TokenResponse {
    token: String,
}

/// OCI/Docker manifest (simplified — handles both v2s2 and OCI).
#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    layers: Vec<LayerDescriptor>,
    // v1 compat: some registries return "fsLayers" instead
}

#[derive(Deserialize)]
struct LayerDescriptor {
    digest: String,
    size: u64,
    #[serde(rename = "mediaType")]
    media_type: String,
}

/// Parse an image reference into registry, repository, and tag.
///
/// Examples:
/// - `ubuntu:22.04` → `docker.io`, `library/ubuntu`, `22.04`
/// - `nvcr.io/nvidia/pytorch:24.01` → `nvcr.io`, `nvidia/pytorch`, `24.01`
/// - `docker://ubuntu` → `docker.io`, `library/ubuntu`, `latest`
/// - `ghcr.io/org/repo` → `ghcr.io`, `org/repo`, `latest`
pub fn parse_image_ref(image: &str) -> ImageRef {
    let image = image.strip_prefix("docker://").unwrap_or(image);

    let (name, tag) = if let Some((n, t)) = image.rsplit_once(':') {
        // Make sure the ':' is for the tag, not a port
        if t.contains('/') {
            (image, "latest")
        } else {
            (n, t)
        }
    } else {
        (image, "latest")
    };

    let (registry, repository) =
        if name.contains('.') || name.contains(':') || name.contains("localhost") {
            // Has a dot or colon → explicit registry
            if let Some((reg, repo)) = name.split_once('/') {
                (reg.to_string(), repo.to_string())
            } else {
                ("docker.io".to_string(), format!("library/{}", name))
            }
        } else if name.contains('/') {
            // user/repo format → Docker Hub
            ("docker.io".to_string(), name.to_string())
        } else {
            // bare name → Docker Hub official library
            ("docker.io".to_string(), format!("library/{}", name))
        };

    ImageRef {
        registry,
        repository,
        tag: tag.to_string(),
    }
}

/// Pull an image from a registry and create a squashfs file.
///
/// Returns the path to the squashfs file.
pub async fn pull_image(image: &str, output_dir: &Path) -> anyhow::Result<PathBuf> {
    let image_ref = parse_image_ref(image);
    info!(
        registry = %image_ref.registry,
        repository = %image_ref.repository,
        tag = %image_ref.tag,
        "pulling image"
    );

    let sanitized = sanitize_name(image);
    let sqsh_path = output_dir.join(format!("{}.sqsh", sanitized));

    if sqsh_path.exists() {
        info!(path = %sqsh_path.display(), "image already exists");
        return Ok(sqsh_path);
    }

    std::fs::create_dir_all(output_dir)?;

    // Create temp directory for rootfs assembly
    let tmp_dir = output_dir.join(format!(".pulling_{}", sanitized));
    let rootfs_dir = tmp_dir.join("rootfs");
    std::fs::create_dir_all(&rootfs_dir)?;

    let result = pull_and_extract(&image_ref, &rootfs_dir).await;
    if let Err(e) = &result {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return Err(anyhow::anyhow!("{}", e));
    }

    // Pack into squashfs
    info!("creating squashfs image");
    let mksquashfs_result = std::process::Command::new("mksquashfs")
        .args([
            rootfs_dir.to_str().unwrap(),
            sqsh_path.to_str().unwrap(),
            "-noappend",
            "-comp",
            "zstd",
            "-quiet",
        ])
        .output();

    match mksquashfs_result {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let _ = std::fs::remove_dir_all(&tmp_dir);
            bail!("mksquashfs failed: {}", stderr.trim());
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            bail!(
                "mksquashfs not found. Install squashfs-tools:\n  \
                 sudo apt install squashfs-tools    # Debian/Ubuntu\n  \
                 sudo dnf install squashfs-tools    # Fedora/RHEL"
            );
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            bail!("failed to run mksquashfs: {}", e);
        }
    }

    // Clean up temp dir
    let _ = std::fs::remove_dir_all(&tmp_dir);

    let size = std::fs::metadata(&sqsh_path).map(|m| m.len()).unwrap_or(0);
    info!(
        path = %sqsh_path.display(),
        size_mb = size / 1_048_576,
        "image pulled successfully"
    );

    Ok(sqsh_path)
}

/// Download manifest and layers, extract to rootfs directory.
async fn pull_and_extract(image_ref: &ImageRef, rootfs_dir: &Path) -> anyhow::Result<()> {
    let client = reqwest::Client::builder().user_agent("spur/0.1").build()?;

    // Get auth token
    let token = get_auth_token(&client, image_ref).await?;

    // Fetch manifest
    let registry_url = registry_base_url(&image_ref.registry);
    let manifest_url = format!(
        "{}/v2/{}/manifests/{}",
        registry_url, image_ref.repository, image_ref.tag
    );

    debug!(url = %manifest_url, "fetching manifest");
    let mut req = client.get(&manifest_url).header(
        ACCEPT,
        "application/vnd.oci.image.manifest.v1+json, \
         application/vnd.docker.distribution.manifest.v2+json, \
         application/vnd.oci.image.index.v1+json, \
         application/vnd.docker.distribution.manifest.list.v2+json",
    );
    if let Some(ref token) = token {
        req = req.header(AUTHORIZATION, format!("Bearer {}", token));
    }

    let resp = req.send().await.context("failed to fetch manifest")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!(
            "registry returned {} for manifest of {}:{}\n{}",
            status,
            image_ref.repository,
            image_ref.tag,
            body.chars().take(500).collect::<String>()
        );
    }

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let manifest_body = resp.text().await?;

    // Handle manifest list / image index (multi-arch)
    let manifest: Manifest =
        if content_type.contains("manifest.list") || content_type.contains("image.index") {
            let index = resolve_manifest_list(
                &client,
                &manifest_body,
                &registry_url,
                image_ref,
                token.as_deref(),
            )
            .await?;
            index
        } else {
            serde_json::from_str(&manifest_body).context("failed to parse manifest JSON")?
        };

    if manifest.layers.is_empty() {
        bail!("manifest has no layers — image may be empty or unsupported format");
    }

    info!(layers = manifest.layers.len(), "downloading layers");

    // Download and extract each layer
    for (i, layer) in manifest.layers.iter().enumerate() {
        let total_mb = layer.size / 1_048_576;
        info!(
            layer = i + 1,
            total = manifest.layers.len(),
            digest = %layer.digest,
            size_mb = total_mb,
            "downloading layer"
        );

        let blob_url = format!(
            "{}/v2/{}/blobs/{}",
            registry_url, image_ref.repository, layer.digest
        );
        let mut req = client.get(&blob_url);
        if let Some(ref token) = token {
            req = req.header(AUTHORIZATION, format!("Bearer {}", token));
        }

        let resp = req.send().await.context("failed to download layer")?;
        if !resp.status().is_success() {
            bail!(
                "registry returned {} for layer {}",
                resp.status(),
                layer.digest
            );
        }

        let bytes = resp.bytes().await.context("failed to read layer body")?;

        // Layers are gzipped tarballs
        if layer.media_type.contains("gzip") || layer.digest.starts_with("sha256:") {
            extract_tar_gz(&bytes, rootfs_dir)
                .with_context(|| format!("failed to extract layer {}", i + 1))?;
        }
    }

    Ok(())
}

/// Registry credentials loaded from file or environment.
#[derive(Debug, Clone)]
pub struct RegistryCredentials {
    pub username: String,
    pub password: String,
}

/// Load credentials for a registry from:
/// 1. Environment: SPUR_REGISTRY_USER + SPUR_REGISTRY_PASSWORD
/// 2. Credentials file: ~/.config/spur/credentials (netrc format)
/// 3. Docker config: ~/.docker/config.json (for compat)
pub fn load_credentials(registry: &str) -> Option<RegistryCredentials> {
    // 1. Environment variables
    if let (Ok(user), Ok(pass)) = (
        std::env::var("SPUR_REGISTRY_USER"),
        std::env::var("SPUR_REGISTRY_PASSWORD"),
    ) {
        if !user.is_empty() {
            return Some(RegistryCredentials {
                username: user,
                password: pass,
            });
        }
    }

    // 2. Spur credentials file (netrc format: machine <registry> login <user> password <pass>)
    let cred_path = dirs_credentials_path();
    if let Ok(content) = std::fs::read_to_string(&cred_path) {
        if let Some(cred) = parse_netrc(&content, registry) {
            return Some(cred);
        }
    }

    // 3. Docker config.json (base64 encoded "user:pass" in auths)
    if let Some(cred) = load_docker_config_auth(registry) {
        return Some(cred);
    }

    None
}

fn dirs_credentials_path() -> PathBuf {
    if let Ok(config) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(config).join("spur/credentials")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".config/spur/credentials")
    } else {
        PathBuf::from("/etc/spur/credentials")
    }
}

fn parse_netrc(content: &str, registry: &str) -> Option<RegistryCredentials> {
    let mut machine_match = false;
    let mut username = None;
    let mut password = None;
    let tokens: Vec<&str> = content.split_whitespace().collect();
    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "machine" if i + 1 < tokens.len() => {
                machine_match = tokens[i + 1] == registry
                    || (registry == "docker.io" && tokens[i + 1] == "registry-1.docker.io");
                username = None;
                password = None;
                i += 2;
            }
            "login" if machine_match && i + 1 < tokens.len() => {
                username = Some(tokens[i + 1].to_string());
                i += 2;
            }
            "password" if machine_match && i + 1 < tokens.len() => {
                password = Some(tokens[i + 1].to_string());
                i += 2;
            }
            _ => i += 1,
        }
        if machine_match && username.is_some() && password.is_some() {
            return Some(RegistryCredentials {
                username: username.unwrap(),
                password: password.unwrap(),
            });
        }
    }
    None
}

fn load_docker_config_auth(registry: &str) -> Option<RegistryCredentials> {
    let docker_config = if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".docker/config.json")
    } else {
        return None;
    };

    let content = std::fs::read_to_string(&docker_config).ok()?;
    let config: serde_json::Value = serde_json::from_str(&content).ok()?;
    let auths = config.get("auths")?;

    // Try exact match and common aliases
    let keys_to_try = if registry == "docker.io" {
        vec![
            "docker.io",
            "https://index.docker.io/v1/",
            "registry-1.docker.io",
        ]
    } else {
        vec![registry]
    };

    for key in keys_to_try {
        if let Some(entry) = auths.get(key) {
            if let Some(auth_b64) = entry.get("auth").and_then(|a| a.as_str()) {
                use std::io::Read;
                let decoded = base64_decode(auth_b64)?;
                let (user, pass) = decoded.split_once(':')?;
                return Some(RegistryCredentials {
                    username: user.to_string(),
                    password: pass.to_string(),
                });
            }
        }
    }

    None
}

fn base64_decode(input: &str) -> Option<String> {
    // Simple base64 decode without pulling in a crate
    let input = input.trim();
    let bytes: Vec<u8> = input
        .bytes()
        .filter_map(|b| match b {
            b'A'..=b'Z' => Some(b - b'A'),
            b'a'..=b'z' => Some(b - b'a' + 26),
            b'0'..=b'9' => Some(b - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            b'=' => None,
            _ => None,
        })
        .collect();

    let mut result = Vec::new();
    for chunk in bytes.chunks(4) {
        if chunk.len() >= 2 {
            result.push((chunk[0] << 2) | (chunk[1] >> 4));
        }
        if chunk.len() >= 3 {
            result.push((chunk[1] << 4) | (chunk[2] >> 2));
        }
        if chunk.len() >= 4 {
            result.push((chunk[2] << 6) | chunk[3]);
        }
    }

    String::from_utf8(result).ok()
}

/// Get an auth token from the registry.
///
/// Supports:
/// - Docker Hub token auth
/// - Basic auth with credentials from file/env
/// - Anonymous access for public images
async fn get_auth_token(
    client: &reqwest::Client,
    image_ref: &ImageRef,
) -> anyhow::Result<Option<String>> {
    let creds = load_credentials(&image_ref.registry);

    if image_ref.registry == "docker.io" {
        let url = format!(
            "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{}:pull",
            image_ref.repository
        );
        let mut req = client.get(&url);
        if let Some(ref creds) = creds {
            req = req.basic_auth(&creds.username, Some(&creds.password));
        }
        let resp = req
            .send()
            .await
            .context("failed to get Docker Hub auth token")?;
        if resp.status().is_success() {
            let token_resp: TokenResponse = resp.json().await?;
            return Ok(Some(token_resp.token));
        }
    }

    // For non-Docker Hub registries with credentials, use basic auth
    // The token will be passed as-is (basic auth encoded)
    if let Some(creds) = creds {
        use std::fmt::Write;
        let mut basic = String::new();
        write!(
            basic,
            "Basic {}",
            encode_basic_auth(&creds.username, &creds.password)
        )
        .ok();
        return Ok(Some(basic));
    }

    // Try anonymous access
    Ok(None)
}

fn encode_basic_auth(user: &str, pass: &str) -> String {
    let input = format!("{}:{}", user, pass);
    let mut result = String::new();
    let b64_chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(b64_chars[((triple >> 18) & 0x3F) as usize] as char);
        result.push(b64_chars[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(b64_chars[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(b64_chars[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Resolve a manifest list (multi-arch) to a single amd64/linux manifest.
async fn resolve_manifest_list(
    client: &reqwest::Client,
    body: &str,
    registry_url: &str,
    image_ref: &ImageRef,
    token: Option<&str>,
) -> anyhow::Result<Manifest> {
    #[derive(Deserialize)]
    struct ManifestList {
        manifests: Vec<ManifestEntry>,
    }
    #[derive(Deserialize)]
    struct ManifestEntry {
        digest: String,
        #[serde(default)]
        platform: Option<Platform>,
    }
    #[derive(Deserialize)]
    struct Platform {
        architecture: String,
        os: String,
    }

    let list: ManifestList = serde_json::from_str(body).context("failed to parse manifest list")?;

    // Find linux/amd64
    let entry = list
        .manifests
        .iter()
        .find(|m| {
            m.platform
                .as_ref()
                .map(|p| p.architecture == "amd64" && p.os == "linux")
                .unwrap_or(false)
        })
        .or_else(|| list.manifests.first())
        .ok_or_else(|| anyhow::anyhow!("no linux/amd64 manifest found in manifest list"))?;

    debug!(digest = %entry.digest, "resolved manifest list to platform manifest");

    let url = format!(
        "{}/v2/{}/manifests/{}",
        registry_url, image_ref.repository, entry.digest
    );
    let mut req = client.get(&url).header(
        ACCEPT,
        "application/vnd.oci.image.manifest.v1+json, \
         application/vnd.docker.distribution.manifest.v2+json",
    );
    if let Some(token) = token {
        req = req.header(AUTHORIZATION, format!("Bearer {}", token));
    }

    let resp = req.send().await?;
    if !resp.status().is_success() {
        bail!("failed to fetch platform manifest: {}", resp.status());
    }

    let manifest: Manifest = resp
        .json()
        .await
        .context("failed to parse platform manifest")?;
    Ok(manifest)
}

/// Extract a gzipped tarball into a directory.
fn extract_tar_gz(data: &[u8], dest: &Path) -> anyhow::Result<()> {
    let decoder = GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);
    archive.set_overwrite(true);
    // Unpack, ignoring permission errors (common in rootless)
    for entry in archive.entries()? {
        match entry {
            Ok(mut entry) => {
                // Skip whiteout files (.wh.*) — used for layer deletion
                let path = entry.path()?.to_path_buf();
                let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
                if filename.starts_with(".wh.") {
                    // Whiteout: delete the corresponding file
                    let target = if filename == ".wh..wh..opq" {
                        // Opaque whiteout: directory should be empty
                        // (skip for now — complex to handle)
                        continue;
                    } else {
                        let real_name = filename.strip_prefix(".wh.").unwrap();
                        dest.join(path.parent().unwrap_or(Path::new("")))
                            .join(real_name)
                    };
                    let _ = std::fs::remove_file(&target);
                    let _ = std::fs::remove_dir_all(&target);
                    continue;
                }

                if let Err(e) = entry.unpack_in(dest) {
                    // Ignore permission errors on special files
                    debug!(path = %path.display(), error = %e, "skipping entry");
                }
            }
            Err(e) => {
                debug!(error = %e, "skipping tar entry");
            }
        }
    }
    Ok(())
}

/// Get the base URL for a registry.
fn registry_base_url(registry: &str) -> String {
    if registry == "docker.io" {
        "https://registry-1.docker.io".to_string()
    } else if registry.starts_with("localhost") {
        format!("http://{}", registry)
    } else {
        format!("https://{}", registry)
    }
}

/// Sanitize an image name for use as a filename.
pub fn sanitize_name(name: &str) -> String {
    name.replace("docker://", "")
        .replace('/', "+")
        .replace(':', "+")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dockerhub_official() {
        let r = parse_image_ref("ubuntu:22.04");
        assert_eq!(r.registry, "docker.io");
        assert_eq!(r.repository, "library/ubuntu");
        assert_eq!(r.tag, "22.04");
    }

    #[test]
    fn test_parse_dockerhub_user() {
        let r = parse_image_ref("nvidia/cuda:12.0-base");
        assert_eq!(r.registry, "docker.io");
        assert_eq!(r.repository, "nvidia/cuda");
        assert_eq!(r.tag, "12.0-base");
    }

    #[test]
    fn test_parse_custom_registry() {
        let r = parse_image_ref("nvcr.io/nvidia/pytorch:24.01");
        assert_eq!(r.registry, "nvcr.io");
        assert_eq!(r.repository, "nvidia/pytorch");
        assert_eq!(r.tag, "24.01");
    }

    #[test]
    fn test_parse_ghcr() {
        let r = parse_image_ref("ghcr.io/org/repo:v1.2.3");
        assert_eq!(r.registry, "ghcr.io");
        assert_eq!(r.repository, "org/repo");
        assert_eq!(r.tag, "v1.2.3");
    }

    #[test]
    fn test_parse_no_tag() {
        let r = parse_image_ref("alpine");
        assert_eq!(r.registry, "docker.io");
        assert_eq!(r.repository, "library/alpine");
        assert_eq!(r.tag, "latest");
    }

    #[test]
    fn test_parse_docker_prefix() {
        let r = parse_image_ref("docker://ubuntu:22.04");
        assert_eq!(r.registry, "docker.io");
        assert_eq!(r.repository, "library/ubuntu");
        assert_eq!(r.tag, "22.04");
    }

    #[test]
    fn test_parse_localhost_registry() {
        let r = parse_image_ref("localhost:5000/myimage:dev");
        assert_eq!(r.registry, "localhost:5000");
        assert_eq!(r.repository, "myimage");
        assert_eq!(r.tag, "dev");
    }

    #[test]
    fn test_registry_base_url() {
        assert_eq!(
            registry_base_url("docker.io"),
            "https://registry-1.docker.io"
        );
        assert_eq!(registry_base_url("ghcr.io"), "https://ghcr.io");
        assert_eq!(registry_base_url("localhost:5000"), "http://localhost:5000");
    }

    #[test]
    fn test_sanitize() {
        assert_eq!(sanitize_name("ubuntu:22.04"), "ubuntu+22.04");
        assert_eq!(
            sanitize_name("docker://nvcr.io/nvidia/pytorch:24.01"),
            "nvcr.io+nvidia+pytorch+24.01"
        );
    }
}
