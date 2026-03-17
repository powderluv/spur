//! Detect the node's network address for registration with the controller.
//!
//! Priority: WireGuard interface IP > hostname resolution > fallback.

use std::process::Command;

use tracing::{debug, info, warn};

use crate::address::{AddressSource, NodeAddress};

/// Detect the best address for this node to advertise to the controller.
///
/// Checks for a WireGuard interface first (default: `spur0`), then falls
/// back to hostname resolution.
pub fn detect_node_address(hostname: &str, port: u16, wg_interface: &str) -> NodeAddress {
    // Try WireGuard interface first
    if let Some(ip) = get_wg_address(wg_interface) {
        info!(ip = %ip, interface = wg_interface, "detected WireGuard address");
        return NodeAddress {
            ip,
            hostname: hostname.to_string(),
            port,
            source: AddressSource::WireGuardDetected {
                interface: wg_interface.to_string(),
            },
        };
    }

    // Fall back to hostname resolution
    let ip = resolve_hostname(hostname);
    debug!(ip = %ip, hostname, "using hostname-resolved address");
    NodeAddress {
        ip,
        hostname: hostname.to_string(),
        port,
        source: AddressSource::Static,
    }
}

/// Query a WireGuard interface for its assigned address.
/// Parses output of `ip addr show <interface>`.
fn get_wg_address(interface: &str) -> Option<String> {
    let output = Command::new("ip")
        .args(["addr", "show", interface])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Parse "inet 10.44.0.2/16" from ip addr output
    for line in stdout.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("inet ") {
            // Take address part before /prefix
            if let Some(addr) = rest.split('/').next() {
                let addr = addr.trim();
                if !addr.is_empty() {
                    return Some(addr.to_string());
                }
            }
        }
    }

    None
}

/// Resolve hostname to IP address.
fn resolve_hostname(hostname: &str) -> String {
    use std::net::ToSocketAddrs;
    let addr_str = format!("{}:0", hostname);
    match addr_str.to_socket_addrs() {
        Ok(mut addrs) => {
            if let Some(addr) = addrs.next() {
                addr.ip().to_string()
            } else {
                warn!(hostname, "hostname resolved to no addresses, using 127.0.0.1");
                "127.0.0.1".to_string()
            }
        }
        Err(e) => {
            warn!(hostname, error = %e, "failed to resolve hostname, using 127.0.0.1");
            "127.0.0.1".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_falls_back_to_hostname() {
        // On a machine without spur0 interface, should fall back to hostname
        let addr = detect_node_address("localhost", 6818, "spur0");
        assert_eq!(addr.port, 6818);
        assert_eq!(addr.source, AddressSource::Static);
        // localhost should resolve to 127.0.0.1
        assert!(addr.ip == "127.0.0.1" || addr.ip == "::1");
    }
}
