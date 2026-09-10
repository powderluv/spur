// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `spurstepd`: the per-job supervisor spurd hands a job off to, running for
//! the job's whole lifetime independent of spurd's own restarts.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // stderr is inherited across the double-fork, so these land in the agent log
    // instead of being discarded for the job's whole lifetime.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let exit_code = spurd::stepd::run_process(&args).await.map_err(|error| {
        eprintln!("spurstepd failed: {error:#}");
        error
    })?;
    std::process::exit(exit_code);
}
