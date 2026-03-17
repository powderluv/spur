pub mod address;
pub mod detect;
pub mod wireguard;

pub use address::{AddressPool, AddressSource, NodeAddress};
pub use detect::detect_node_address;
pub use wireguard::{WgConfig, WgPeer};
