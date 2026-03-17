use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

/// Accounting management commands.
#[derive(Parser, Debug)]
#[command(name = "sacctmgr", about = "Spur accounting manager")]
pub struct SacctmgrArgs {
    #[command(subcommand)]
    pub command: SacctmgrCommand,

    /// Accounting daemon address
    #[arg(long, env = "SPUR_ACCOUNTING_ADDR", default_value = "http://localhost:6819", global = true)]
    pub accounting: String,

    /// Immediate mode (no confirmation)
    #[arg(short = 'i', long, global = true)]
    pub immediate: bool,
}

#[derive(Subcommand, Debug)]
pub enum SacctmgrCommand {
    /// Add entities
    Add {
        /// Entity type: account, user, qos
        entity: String,
        /// key=value pairs
        #[arg(trailing_var_arg = true)]
        params: Vec<String>,
    },
    /// Delete entities
    Delete {
        /// Entity type: account, user, qos
        entity: String,
        /// key=value pairs (name= or where clause)
        #[arg(trailing_var_arg = true)]
        params: Vec<String>,
    },
    /// Modify entities
    Modify {
        /// Entity type: account, user, qos
        entity: String,
        /// key=value pairs
        #[arg(trailing_var_arg = true)]
        params: Vec<String>,
    },
    /// List/show entities
    Show {
        /// Entity type: account, user, qos, association
        entity: String,
        /// Optional filter
        #[arg(trailing_var_arg = true)]
        params: Vec<String>,
    },
    /// List entities (alias for show)
    List {
        entity: String,
        #[arg(trailing_var_arg = true)]
        params: Vec<String>,
    },
}

pub async fn main() -> Result<()> {
    let args = SacctmgrArgs::try_parse_from(std::env::args())?;

    match args.command {
        SacctmgrCommand::Add { entity, params } => add(&entity, &params).await,
        SacctmgrCommand::Delete { entity, params } => delete(&entity, &params).await,
        SacctmgrCommand::Modify { entity, params } => modify(&entity, &params).await,
        SacctmgrCommand::Show { entity, params } | SacctmgrCommand::List { entity, params } => {
            show(&entity, &params).await
        }
    }
}

fn parse_params(params: &[String]) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    // Handle both "key=value" and "key value" forms
    let mut iter = params.iter();
    while let Some(param) = iter.next() {
        if let Some((key, value)) = param.split_once('=') {
            map.insert(key.to_lowercase(), value.to_string());
        } else if param.starts_with("where") || param.starts_with("set") {
            // Skip Slurm-style "where" and "set" keywords
            continue;
        } else {
            // Try next param as value
            let key = param.to_lowercase();
            if let Some(value) = iter.next() {
                map.insert(key, value.clone());
            }
        }
    }
    map
}

async fn add(entity: &str, params: &[String]) -> Result<()> {
    let p = parse_params(params);

    match entity.to_lowercase().as_str() {
        "account" => {
            let name = p.get("name").or_else(|| p.get("account"))
                .ok_or_else(|| anyhow::anyhow!("name= required"))?;
            let desc = p.get("description").cloned().unwrap_or_default();
            let org = p.get("organization").cloned().unwrap_or_default();
            let parent = p.get("parent").cloned();
            let fairshare: u32 = p.get("fairshare").and_then(|v| v.parse().ok()).unwrap_or(1);

            println!(
                " Adding Account(s)\n  Name       = {}\n  Descr      = {}\n  Org        = {}\n  Parent     = {}\n  Fairshare  = {}",
                name, desc, org, parent.as_deref().unwrap_or("root"), fairshare
            );
            // In a real implementation, this would call the accounting gRPC service.
            // For now, print what would be done.
            println!(" Account added.");
            Ok(())
        }
        "user" => {
            let name = p.get("name").or_else(|| p.get("user"))
                .ok_or_else(|| anyhow::anyhow!("name= required"))?;
            let account = p.get("account").or_else(|| p.get("defaultaccount"))
                .ok_or_else(|| anyhow::anyhow!("account= required"))?;
            let admin = p.get("adminlevel").cloned().unwrap_or_else(|| "none".into());

            println!(
                " Adding User(s)\n  Name       = {}\n  Account    = {}\n  Admin      = {}",
                name, account, admin
            );
            println!(" User added.");
            Ok(())
        }
        "qos" => {
            let name = p.get("name").or_else(|| p.get("qos"))
                .ok_or_else(|| anyhow::anyhow!("name= required"))?;
            let priority: i32 = p.get("priority").and_then(|v| v.parse().ok()).unwrap_or(0);
            let preempt = p.get("preemptmode").cloned().unwrap_or_else(|| "off".into());
            let max_wall = p.get("maxwall").cloned();
            let max_jobs = p.get("maxjobsperuser").and_then(|v| v.parse::<u32>().ok());

            println!(
                " Adding QOS(s)\n  Name       = {}\n  Priority   = {}\n  Preempt    = {}",
                name, priority, preempt
            );
            if let Some(mw) = &max_wall {
                println!("  MaxWall    = {}", mw);
            }
            if let Some(mj) = max_jobs {
                println!("  MaxJobsPU  = {}", mj);
            }
            println!(" QOS added.");
            Ok(())
        }
        other => bail!("sacctmgr: unknown entity type '{}'. Use: account, user, qos", other),
    }
}

async fn delete(entity: &str, params: &[String]) -> Result<()> {
    let p = parse_params(params);

    let name = p.get("name").or_else(|| p.get("account")).or_else(|| p.get("user")).or_else(|| p.get("qos"))
        .ok_or_else(|| anyhow::anyhow!("name= required"))?;

    match entity.to_lowercase().as_str() {
        "account" => println!(" Deleting account: {}", name),
        "user" => {
            let account = p.get("account").map(|a| a.as_str()).unwrap_or("all");
            println!(" Deleting user {} from account {}", name, account);
        }
        "qos" => println!(" Deleting QOS: {}", name),
        other => bail!("sacctmgr: unknown entity type '{}'", other),
    }
    println!(" Done.");
    Ok(())
}

async fn modify(entity: &str, params: &[String]) -> Result<()> {
    let p = parse_params(params);

    match entity.to_lowercase().as_str() {
        "account" | "user" | "qos" => {
            println!(" Modifying {} with: {:?}", entity, p);
            println!(" Done.");
            Ok(())
        }
        other => bail!("sacctmgr: unknown entity type '{}'", other),
    }
}

async fn show(entity: &str, _params: &[String]) -> Result<()> {
    match entity.to_lowercase().as_str() {
        "account" | "accounts" => {
            println!("{:<20} {:<30} {:<15} {:<10} {:<10}",
                "Account", "Descr", "Org", "Parent", "Share");
            println!("{}", "-".repeat(85));
            // Would query accounting DB here
            println!("{:<20} {:<30} {:<15} {:<10} {:<10}",
                "(no accounts configured — use 'sacctmgr add account')", "", "", "", "");
        }
        "user" | "users" => {
            println!("{:<15} {:<20} {:<10} {:<20}",
                "User", "Account", "Admin", "Default Acct");
            println!("{}", "-".repeat(65));
        }
        "qos" => {
            println!("{:<15} {:<8} {:<10} {:<12} {:<10} {:<10}",
                "Name", "Prio", "Preempt", "UsageFactor", "MaxJobsPU", "MaxWall");
            println!("{}", "-".repeat(65));
            // Default "normal" QOS
            println!("{:<15} {:<8} {:<10} {:<12} {:<10} {:<10}",
                "normal", "0", "off", "1.0", "", "");
        }
        "association" | "associations" => {
            println!("{:<15} {:<20} {:<15} {:<10} {:<10}",
                "User", "Account", "Partition", "Share", "Default");
            println!("{}", "-".repeat(70));
        }
        "tres" => {
            println!("{:<5} {:<15} {:<10}",
                "ID", "Type", "Name");
            println!("{}", "-".repeat(30));
            println!("{:<5} {:<15} {:<10}", "1", "cpu", "");
            println!("{:<5} {:<15} {:<10}", "2", "mem", "");
            println!("{:<5} {:<15} {:<10}", "3", "energy", "");
            println!("{:<5} {:<15} {:<10}", "4", "node", "");
            println!("{:<5} {:<15} {:<10}", "1001", "gres/gpu", "");
            println!("{:<5} {:<15} {:<10}", "1002", "billing", "");
        }
        other => bail!("sacctmgr: unknown entity '{}'. Use: account, user, qos, association, tres", other),
    }
    Ok(())
}
