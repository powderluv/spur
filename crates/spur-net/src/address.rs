//! IP address pool allocation from a CIDR block.
//!
//! The controller maintains a pool of IPs and hands them out to agents
//! during WireGuard mesh setup. Address .1 is reserved for the controller.

use std::collections::HashSet;
use std::net::Ipv4Addr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AddressError {
    #[error("invalid CIDR: {0}")]
    InvalidCidr(String),
    #[error("address pool exhausted")]
    PoolExhausted,
}

/// How a node's address was determined.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AddressSource {
    /// Address assigned from the WireGuard mesh pool.
    WireGuard { interface: String },
    /// Detected from an existing WireGuard interface.
    WireGuardDetected { interface: String },
    /// Static address from config or hostname resolution.
    Static,
}

/// A resolved node address with metadata about its source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAddress {
    pub ip: String,
    pub hostname: String,
    pub port: u16,
    pub source: AddressSource,
}

/// Manages IP allocation from a CIDR block (e.g. 10.44.0.0/16).
///
/// Address .0 is network, .1 is reserved for the controller,
/// .2+ are assigned to agents in order.
pub struct AddressPool {
    base: u32,
    prefix_len: u8,
    allocated: HashSet<u32>,
}

impl AddressPool {
    /// Create a new pool from a CIDR string like "10.44.0.0/16".
    pub fn new(cidr: &str) -> Result<Self, AddressError> {
        let (addr_str, prefix_str) = cidr
            .split_once('/')
            .ok_or_else(|| AddressError::InvalidCidr(cidr.into()))?;
        let addr: Ipv4Addr = addr_str
            .parse()
            .map_err(|_| AddressError::InvalidCidr(cidr.into()))?;
        let prefix_len: u8 = prefix_str
            .parse()
            .map_err(|_| AddressError::InvalidCidr(cidr.into()))?;
        if prefix_len > 30 {
            return Err(AddressError::InvalidCidr(
                "prefix length must be <= 30".into(),
            ));
        }
        let base = u32::from(addr);
        // Mask to network address
        let mask = !((1u32 << (32 - prefix_len)) - 1);
        let base = base & mask;

        let mut allocated = HashSet::new();
        // Reserve network address (.0)
        allocated.insert(base);

        Ok(Self {
            base,
            prefix_len,
            allocated,
        })
    }

    /// Mark an address as already allocated (e.g. loading from state).
    pub fn mark_allocated(&mut self, ip: Ipv4Addr) {
        self.allocated.insert(u32::from(ip));
    }

    /// Allocate the next available IP from the pool.
    pub fn allocate(&mut self) -> Result<Ipv4Addr, AddressError> {
        let host_bits = 32 - self.prefix_len;
        let max_host = (1u32 << host_bits) - 1; // broadcast
        // Start from .1 (skip .0 network)
        for offset in 1..max_host {
            let candidate = self.base + offset;
            if !self.allocated.contains(&candidate) {
                self.allocated.insert(candidate);
                return Ok(Ipv4Addr::from(candidate));
            }
        }
        Err(AddressError::PoolExhausted)
    }

    /// Allocate a specific IP (e.g. .1 for the controller).
    pub fn allocate_specific(&mut self, ip: Ipv4Addr) -> Result<(), AddressError> {
        let addr = u32::from(ip);
        if self.allocated.contains(&addr) {
            return Err(AddressError::PoolExhausted);
        }
        self.allocated.insert(addr);
        Ok(())
    }

    /// The CIDR string for this pool (e.g. "10.44.0.0/16").
    pub fn cidr(&self) -> String {
        format!("{}/{}", Ipv4Addr::from(self.base), self.prefix_len)
    }

    /// The prefix length.
    pub fn prefix_len(&self) -> u8 {
        self.prefix_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_allocation() {
        let mut pool = AddressPool::new("10.44.0.0/24").unwrap();
        let first = pool.allocate().unwrap();
        assert_eq!(first, Ipv4Addr::new(10, 44, 0, 1));
        let second = pool.allocate().unwrap();
        assert_eq!(second, Ipv4Addr::new(10, 44, 0, 2));
    }

    #[test]
    fn test_pool_specific() {
        let mut pool = AddressPool::new("10.44.0.0/24").unwrap();
        pool.allocate_specific(Ipv4Addr::new(10, 44, 0, 1)).unwrap();
        let next = pool.allocate().unwrap();
        // .1 is taken, so next available is .2
        assert_eq!(next, Ipv4Addr::new(10, 44, 0, 2));
    }

    #[test]
    fn test_pool_exhaustion() {
        // /30 gives us 4 addresses: .0 (net), .1, .2, .3 (broadcast excluded)
        let mut pool = AddressPool::new("10.0.0.0/30").unwrap();
        assert!(pool.allocate().is_ok()); // .1
        assert!(pool.allocate().is_ok()); // .2
        assert!(pool.allocate().is_err()); // .3 is broadcast, pool exhausted
    }

    #[test]
    fn test_pool_mark_allocated() {
        let mut pool = AddressPool::new("10.44.0.0/24").unwrap();
        pool.mark_allocated(Ipv4Addr::new(10, 44, 0, 1));
        pool.mark_allocated(Ipv4Addr::new(10, 44, 0, 2));
        let next = pool.allocate().unwrap();
        assert_eq!(next, Ipv4Addr::new(10, 44, 0, 3));
    }

    #[test]
    fn test_invalid_cidr() {
        assert!(AddressPool::new("not-a-cidr").is_err());
        assert!(AddressPool::new("10.0.0.0/33").is_err());
        assert!(AddressPool::new("10.0.0.0").is_err());
    }
}
