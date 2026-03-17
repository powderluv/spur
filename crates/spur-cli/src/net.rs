//! `spur net` subcommands for WireGuard mesh management.

use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use spur_net::wireguard::{self, WgConfig, WgPeer};
use spur_net::address::AddressPool;

/// Network mesh management.
#[derive(Parser, Debug)]
#[command(name = "net", about = "Manage WireGuard mesh network")]
pub struct NetArgs {
    #[command(subcommand)]
    pub command: NetCommand,
}

#[derive(Subcommand, Debug)]
pub enum NetCommand {
    /// Initialize the WireGuard mesh on the controller node.
    ///
    /// Generates a keypair, assigns the .1 address, writes the config,
    /// and brings up the interface.
    Init {
        /// CIDR block for the mesh (default: 10.44.0.0/16)
        #[arg(long, default_value = "10.44.0.0/16")]
        cidr: String,
        /// WireGuard interface name
        #[arg(long, default_value = "spur0")]
        interface: String,
        /// WireGuard listen port
        #[arg(long, default_value_t = 51820)]
        port: u16,
        /// Config directory
        #[arg(long, default_value = "/etc/wireguard")]
        config_dir: PathBuf,
    },
    /// Join the WireGuard mesh from a compute node.
    ///
    /// Generates a local keypair, connects to the controller to exchange
    /// keys, and brings up the interface.
    Join {
        /// Controller's real IP and WireGuard port (e.g. 192.168.1.100:51820)
        #[arg(long)]
        endpoint: String,
        /// Controller's WireGuard public key
        #[arg(long)]
        server_key: String,
        /// IP address to assign to this node (e.g. 10.44.0.2)
        #[arg(long)]
        address: String,
        /// CIDR prefix length
        #[arg(long, default_value_t = 16)]
        prefix_len: u8,
        /// WireGuard interface name
        #[arg(long, default_value = "spur0")]
        interface: String,
        /// Config directory
        #[arg(long, default_value = "/etc/wireguard")]
        config_dir: PathBuf,
    },
    /// Show WireGuard mesh status.
    Status {
        /// WireGuard interface name
        #[arg(long, default_value = "spur0")]
        interface: String,
    },
    /// Add a peer to the running WireGuard interface.
    AddPeer {
        /// Peer's WireGuard public key
        #[arg(long)]
        key: String,
        /// Peer's allowed IP (e.g. 10.44.0.3/32)
        #[arg(long)]
        allowed_ip: String,
        /// Peer's endpoint (e.g. 192.168.1.11:51820)
        #[arg(long)]
        endpoint: Option<String>,
        /// WireGuard interface name
        #[arg(long, default_value = "spur0")]
        interface: String,
    },
}

pub async fn main() -> Result<()> {
    let args = NetArgs::parse();

    match args.command {
        NetCommand::Init {
            cidr,
            interface,
            port,
            config_dir,
        } => cmd_init(&cidr, &interface, port, &config_dir),
        NetCommand::Join {
            endpoint,
            server_key,
            address,
            prefix_len,
            interface,
            config_dir,
        } => cmd_join(&endpoint, &server_key, &address, prefix_len, &interface, &config_dir),
        NetCommand::Status { interface } => cmd_status(&interface),
        NetCommand::AddPeer {
            key,
            allowed_ip,
            endpoint,
            interface,
        } => cmd_add_peer(&key, &allowed_ip, endpoint.as_deref(), &interface),
    }
}

fn cmd_init(cidr: &str, interface: &str, port: u16, config_dir: &Path) -> Result<()> {
    eprintln!("Initializing WireGuard mesh...");

    // Generate keypair
    let keypair = wireguard::generate_keypair()
        .context("failed to generate WireGuard keypair")?;

    // Allocate .1 for the controller
    let mut pool = AddressPool::new(cidr)?;
    let controller_ip = pool.allocate()?;

    let config = WgConfig {
        private_key: keypair.private_key,
        address: format!("{}/{}", controller_ip, pool.prefix_len()),
        listen_port: Some(port),
        peers: vec![],
    };

    // Write config
    std::fs::create_dir_all(config_dir)?;
    let config_path = config_dir.join(format!("{}.conf", interface));
    config.write_to(&config_path)?;
    eprintln!("Config written to {}", config_path.display());

    // Bring up interface
    wireguard::interface_up(interface)?;

    eprintln!();
    eprintln!("WireGuard mesh initialized:");
    eprintln!("  Interface:  {}", interface);
    eprintln!("  Address:    {}/{}", controller_ip, pool.prefix_len());
    eprintln!("  Port:       {}", port);
    eprintln!("  Public key: {}", keypair.public_key);
    eprintln!();
    eprintln!("To add compute nodes, run on each node:");
    eprintln!("  spur net join \\");
    eprintln!("    --endpoint <this-host-ip>:{} \\", port);
    eprintln!("    --server-key {} \\", keypair.public_key);
    eprintln!("    --address <assigned-ip>");
    eprintln!();
    eprintln!("Then add the node as a peer on this controller:");
    eprintln!("  spur net add-peer --key <node-pubkey> --allowed-ip <node-ip>/32 --endpoint <node-real-ip>:{}", port);

    Ok(())
}

fn cmd_join(
    endpoint: &str,
    server_key: &str,
    address: &str,
    prefix_len: u8,
    interface: &str,
    config_dir: &Path,
) -> Result<()> {
    eprintln!("Joining WireGuard mesh...");

    // Generate local keypair
    let keypair = wireguard::generate_keypair()
        .context("failed to generate WireGuard keypair")?;

    let config = WgConfig {
        private_key: keypair.private_key,
        address: format!("{}/{}", address, prefix_len),
        listen_port: Some(51820),
        peers: vec![WgPeer {
            public_key: server_key.to_string(),
            // Route all mesh traffic through the controller initially
            // (in a full mesh, each peer would have its own entry)
            allowed_ips: format!("{}/{}", address_network(address, prefix_len)?, prefix_len),
            endpoint: Some(endpoint.to_string()),
            persistent_keepalive: Some(25),
        }],
    };

    // Write config
    std::fs::create_dir_all(config_dir)?;
    let config_path = config_dir.join(format!("{}.conf", interface));
    config.write_to(&config_path)?;

    // Bring up interface
    wireguard::interface_up(interface)?;

    eprintln!();
    eprintln!("Joined WireGuard mesh:");
    eprintln!("  Interface:  {}", interface);
    eprintln!("  Address:    {}/{}", address, prefix_len);
    eprintln!("  Server:     {}", endpoint);
    eprintln!("  Public key: {}", keypair.public_key);
    eprintln!();
    eprintln!("Add this node as a peer on the controller:");
    eprintln!("  spur net add-peer --key {} --allowed-ip {}/32", keypair.public_key, address);

    Ok(())
}

fn cmd_status(interface: &str) -> Result<()> {
    let output = std::process::Command::new("wg")
        .args(["show", interface])
        .output()
        .context("failed to run `wg show`")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No such device") {
            eprintln!("Interface {} is not up. Run `spur net init` or `spur net join` first.", interface);
            return Ok(());
        }
        bail!("wg show failed: {}", stderr);
    }

    println!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}

fn cmd_add_peer(key: &str, allowed_ip: &str, endpoint: Option<&str>, interface: &str) -> Result<()> {
    let peer = WgPeer {
        public_key: key.to_string(),
        allowed_ips: allowed_ip.to_string(),
        endpoint: endpoint.map(|s| s.to_string()),
        persistent_keepalive: Some(25),
    };

    wireguard::add_peer(interface, &peer)?;
    eprintln!("Peer added to {}:", interface);
    eprintln!("  Public key:  {}", key);
    eprintln!("  Allowed IPs: {}", allowed_ip);
    if let Some(ep) = endpoint {
        eprintln!("  Endpoint:    {}", ep);
    }

    Ok(())
}

/// Given an IP and prefix length, compute the network address.
fn address_network(ip: &str, prefix_len: u8) -> Result<String> {
    let addr: Ipv4Addr = ip.parse().context("invalid IP address")?;
    let mask = !((1u32 << (32 - prefix_len)) - 1);
    let network = u32::from(addr) & mask;
    Ok(Ipv4Addr::from(network).to_string())
}
