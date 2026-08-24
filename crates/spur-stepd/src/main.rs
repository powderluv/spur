// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `spurstepd`: the per-job supervisor spurd hands a job off to. Runs for the
//! job's whole lifetime, independent of spurd's own restarts/upgrades — see
//! `spurd::stepd` for the supervision loop this binary drives.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exit_code = spurd::stepd::run_process(&args).await.map_err(|error| {
        eprintln!("spurstepd failed: {error:#}");
        error
    })?;
    std::process::exit(exit_code);
}
