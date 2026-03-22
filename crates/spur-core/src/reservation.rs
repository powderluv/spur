use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A resource reservation that blocks nodes for specific users/accounts
/// during a time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reservation {
    pub name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub nodes: Vec<String>,
    pub accounts: Vec<String>,
    pub users: Vec<String>,
}

impl Reservation {
    /// Check if the reservation is active at the given time.
    pub fn is_active(&self, now: DateTime<Utc>) -> bool {
        now >= self.start_time && now < self.end_time
    }

    /// Check if the reservation covers a specific node.
    pub fn covers_node(&self, node: &str) -> bool {
        self.nodes.iter().any(|n| n == node)
    }

    /// Check if a user (and optionally account) is allowed to use this reservation.
    pub fn allows_user(&self, user: &str, account: Option<&str>) -> bool {
        if self.users.is_empty() && self.accounts.is_empty() {
            return true; // No restrictions
        }
        if self.users.iter().any(|u| u == user) {
            return true;
        }
        if let Some(acct) = account {
            if self.accounts.iter().any(|a| a == acct) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_reservation() -> Reservation {
        let now = Utc::now();
        Reservation {
            name: "maint".into(),
            start_time: now - Duration::hours(1),
            end_time: now + Duration::hours(1),
            nodes: vec!["node001".into(), "node002".into()],
            accounts: vec!["research".into()],
            users: vec!["alice".into()],
        }
    }

    #[test]
    fn test_is_active() {
        let res = make_reservation();
        assert!(res.is_active(Utc::now()));
        assert!(!res.is_active(res.end_time + Duration::seconds(1)));
        assert!(!res.is_active(res.start_time - Duration::seconds(1)));
    }

    #[test]
    fn test_covers_node() {
        let res = make_reservation();
        assert!(res.covers_node("node001"));
        assert!(res.covers_node("node002"));
        assert!(!res.covers_node("node003"));
    }

    #[test]
    fn test_allows_user() {
        let res = make_reservation();
        assert!(res.allows_user("alice", None));
        assert!(res.allows_user("bob", Some("research")));
        assert!(!res.allows_user("bob", Some("other")));
        assert!(!res.allows_user("bob", None));
    }

    #[test]
    fn test_unrestricted_reservation() {
        let now = Utc::now();
        let res = Reservation {
            name: "open".into(),
            start_time: now - Duration::hours(1),
            end_time: now + Duration::hours(1),
            nodes: vec!["node001".into()],
            accounts: Vec::new(),
            users: Vec::new(),
        };
        assert!(res.allows_user("anyone", None));
    }
}
