//! Data structures for MBTI dataset records.

use serde::Deserialize;

/// A single MBTI dataset record containing personality type and text posts.
#[derive(Debug, Deserialize, Clone)]
pub struct MbtiRecord {
    /// MBTI personality type (e.g., "INTJ", "ENFP")
    #[serde(rename = "type")]
    pub mbti_type: String,
    /// Concatenated social media posts by this user
    pub posts: String,
}
