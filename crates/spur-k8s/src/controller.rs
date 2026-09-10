// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one way the operator opens a channel to spurctld.

use std::future::Future;
use std::time::Duration;

use tonic::transport::{Channel, Endpoint};
use tonic::Status;

use spur_proto::proto::slurm_controller_client::SlurmControllerClient;

// A Service with no ready endpoint drops the SYN and the kernel retries it for
// over two minutes. Give up long before that, so the task's retry loop opens a
// fresh connection soon after the controller becomes ready.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

// A killed controller Pod leaves the channel open until the kernel gives up on
// it, and every RPC waits with it. Pings find the dead peer in seconds; the
// request bound is the backstop. No RPC on this channel streams.
const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(10);
const KEEP_ALIVE_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

pub fn controller_url(addr: &str) -> String {
    if addr.starts_with("http") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    }
}

pub async fn connect(addr: &str) -> anyhow::Result<SlurmControllerClient<Channel>> {
    let channel = Endpoint::from_shared(controller_url(addr))?
        .connect_timeout(CONNECT_TIMEOUT)
        .http2_keep_alive_interval(KEEP_ALIVE_INTERVAL)
        .keep_alive_timeout(KEEP_ALIVE_TIMEOUT)
        .keep_alive_while_idle(true)
        .timeout(REQUEST_TIMEOUT)
        .connect()
        .await?;
    Ok(SlurmControllerClient::new(channel)
        .max_decoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(spur_proto::MAX_GRPC_MESSAGE_SIZE))
}

/// A channel whose peer is gone answers nothing until the kernel gives up on
/// it, and the request bound alone does not close it, so the next call would
/// wait on the same dead connection. Drop it after a transport error instead.
pub fn is_transport_error(status: &Status) -> bool {
    use tonic::Code;
    matches!(
        status.code(),
        Code::Unavailable | Code::Unknown | Code::Cancelled | Code::DeadlineExceeded
    )
}

/// The one long-lived client of a task: reconnects after a transport error.
pub struct ControllerClient {
    addr: String,
    client: Option<SlurmControllerClient<Channel>>,
}

impl ControllerClient {
    pub fn connected(addr: &str, client: SlurmControllerClient<Channel>) -> Self {
        Self {
            addr: addr.to_string(),
            client: Some(client),
        }
    }

    pub async fn call<T, F, Fut>(&mut self, rpc: F) -> Result<T, Status>
    where
        F: FnOnce(SlurmControllerClient<Channel>) -> Fut,
        Fut: Future<Output = Result<T, Status>>,
    {
        let client = match &self.client {
            Some(client) => client.clone(),
            None => {
                let client = connect(&self.addr).await.map_err(|e| {
                    Status::unavailable(format!("cannot reach the controller: {e}"))
                })?;
                self.client.insert(client).clone()
            }
        };
        let result = rpc(client).await;
        if let Err(status) = &result {
            if is_transport_error(status) {
                self.client = None;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unconnected() -> ControllerClient {
        let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        ControllerClient::connected("127.0.0.1:1", SlurmControllerClient::new(channel))
    }

    #[tokio::test]
    async fn a_transport_error_drops_the_channel_and_a_refusal_keeps_it() {
        let mut ctrl = unconnected();
        let refused: Result<(), Status> = ctrl
            .call(|_| async { Err(Status::not_found("no such job")) })
            .await;
        assert!(refused.is_err());
        assert!(
            ctrl.client.is_some(),
            "a refusal is an answer, the channel is fine"
        );

        let gone: Result<(), Status> = ctrl
            .call(|_| async { Err(Status::cancelled("Timeout expired")) })
            .await;
        assert!(gone.is_err());
        assert!(
            ctrl.client.is_none(),
            "a timeout means the peer may be gone"
        );
    }

    /// A peer that accepts the connection and never answers is what a killed
    /// Pod looks like until the kernel gives up. The connect runs on the real
    /// clock, because loopback I/O must be polled; the paused clock afterwards
    /// jumps straight to the timers, so the real bounds apply and the test
    /// still ends at once.
    #[tokio::test]
    async fn a_silent_peer_fails_the_call_in_bounded_time_and_drops_the_channel() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let client = connect(&addr)
            .await
            .expect("the kernel completes the handshake");
        let mut ctrl = ControllerClient::connected(&addr, client);

        tokio::time::pause();
        let err = ctrl
            .call(|mut c| async move { c.ping(()).await })
            .await
            .expect_err("nothing answers on that socket");
        assert!(is_transport_error(&err), "{err}");
        assert!(ctrl.client.is_none(), "the dead channel must not be reused");
        drop(listener);
    }

    #[test]
    fn transport_errors_are_the_codes_a_dead_peer_produces() {
        assert!(is_transport_error(&Status::unavailable(
            "tcp connect error"
        )));
        assert!(is_transport_error(&Status::cancelled("Timeout expired")));
        assert!(!is_transport_error(&Status::not_found("no such job")));
        assert!(!is_transport_error(&Status::permission_denied("no")));
    }

    #[test]
    fn controller_url_adds_a_scheme_only_when_one_is_missing() {
        assert_eq!(
            controller_url("spurctld-client:6817"),
            "http://spurctld-client:6817"
        );
        assert_eq!(
            controller_url("http://spurctld:6817"),
            "http://spurctld:6817"
        );
        assert_eq!(
            controller_url("https://spurctld:6817"),
            "https://spurctld:6817"
        );
    }
}
