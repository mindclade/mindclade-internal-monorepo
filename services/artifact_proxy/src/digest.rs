//! Strict SHA-256 artifact identifiers.

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{fmt, str::FromStr};

/// A canonical lower-case `sha256:` digest.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Digest([u8; 32]);

impl Digest {
    /// Compute a digest from bytes.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Return the lower-case hexadecimal body without the algorithm prefix.
    #[must_use]
    pub fn hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "sha256:{}", self.hex())
    }
}

impl FromStr for Digest {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let body = value
            .strip_prefix("sha256:")
            .ok_or_else(|| "digest must start with sha256:".to_owned())?;
        if body.len() != 64
            || body
                .bytes()
                .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
        {
            return Err("digest must contain 64 lower-case hexadecimal characters".to_owned());
        }
        let decoded = hex::decode(body).map_err(|error| error.to_string())?;
        let bytes: [u8; 32] = decoded
            .try_into()
            .map_err(|_| "invalid SHA-256 length".to_owned())?;
        Ok(Self(bytes))
    }
}

impl TryFrom<String> for Digest {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<Digest> for String {
    fn from(value: Digest) -> Self {
        value.to_string()
    }
}
