mod agent_server;
pub mod container;
mod executor;
mod gpu;
mod reporter;

use std::sync::Arc;

use clap::Parser;
use tracing::info;

use reporter::NodeReporter;

#[derive(Parser)]
#[command(name = "spurd", about = "Spur node agent daemon")]
struct Args {
    /// Configuration file path
    #[arg(short = 'f', long, default_value = "/etc/spur/spur.conf")]
    config: std::path::PathBuf,

    /// Controller address
    #[arg(
        long,
        env = "SPUR_CONTROLLER_ADDR",
        default_value = "http://localhost:6817"
    )]
    controller: String,

    /// Agent gRPC listen address
    #[arg(long, default_value = "[::]:6818")]
    listen: String,

    /// Node name (defaults to hostname)
    #[arg(short = 'N', long)]
    hostname: Option<String>,

    /// Foreground mode
    #[arg(short = 'D', long)]
    foreground: bool,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.parse().unwrap()),
        )
        .init();

    let hostname = args.hostname.unwrap_or_else(|| {
        hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".into())
    });

    // Parse listen port from the listen address for registration
    let listen_port: u16 = args
        .listen
        .rsplit(':')
        .next()
        .and_then(|p| p.parse().ok())
        .unwrap_or(6818);

    info!(
        version = env!("CARGO_PKG_VERSION"),
        hostname = %hostname,
        controller = %args.controller,
        listen = %args.listen,
        "spurd starting"
    );

    // Detect node address (WireGuard > hostname)
    let wg_interface = std::env::var("SPUR_WG_INTERFACE").unwrap_or_else(|_| "spur0".into());
    let node_address = spur_net::detect_node_address(&hostname, listen_port, &wg_interface);
    info!(
        ip = %node_address.ip,
        port = node_address.port,
        source = ?node_address.source,
        "node address detected"
    );

    // Discover local resources
    let resources = reporter::discover_resources();
    info!(
        cpus = resources.cpus,
        memory_mb = resources.memory_mb,
        gpus = resources.gpus.len(),
        "resources discovered"
    );

    // Create the node reporter
    let reporter = Arc::new(NodeReporter::new(
        hostname.clone(),
        args.controller.clone(),
        resources,
        node_address,
    ));

    // Register with controller
    reporter.register().await?;

    // Start heartbeat loop
    let hb_reporter = reporter.clone();
    tokio::spawn(async move {
        hb_reporter.heartbeat_loop().await;
    });

    // Start agent gRPC server (receives job launches from spurctld)
    let agent_service = agent_server::AgentService::new(reporter.clone());
    agent_service.start_monitor(args.controller.clone());

    let addr = args.listen.parse()?;
    info!(%addr, "agent gRPC server listening");

    tonic::transport::Server::builder()
        .add_service(spur_proto::proto::slurm_agent_server::SlurmAgentServer::new(agent_service))
        .serve(addr)
        .await?;

    Ok(())
}
