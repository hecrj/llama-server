use crate::Error;
use crate::http;

use std::fmt;
use std::io;
use std::str::FromStr;

const REPOSITORY: &str = "hecrj/llama-server";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Build(u32);

impl Build {
    pub async fn latest() -> Result<Self, Error> {
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct Release {
            tag_name: String,
        }

        let client = http::client();

        let latest_release_url =
            format!("https://api.github.com/repos/{REPOSITORY}/releases/latest");

        let Release { tag_name } = client
            .get(latest_release_url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        Ok(tag_name.parse()?)
    }

    pub fn locked(number: u32) -> Self {
        Self(number)
    }

    pub fn number(self) -> u32 {
        self.0
    }

    pub fn url(self) -> String {
        format!("https://github.com/{REPOSITORY}/releases/download/{self}")
    }
}

impl FromStr for Build {
    type Err = io::Error;

    fn from_str(build: &str) -> Result<Self, Self::Err> {
        if !build.starts_with('b') {
            return Err(io::Error::other(format!("invalid build: {build}")));
        }

        build
            .trim_start_matches('b')
            .parse()
            .map(Self)
            .map_err(io::Error::other)
    }
}

impl fmt::Display for Build {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b{}", self.0)
    }
}
