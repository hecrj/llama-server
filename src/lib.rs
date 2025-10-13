mod artifact;
mod cache;
mod error;
mod http;

pub use artifact::Artifact;
pub use error::Error;
pub use http::Progress;

use cache::Cache;

use bitflags::bitflags;
use sipper::{Sipper, Straw, sipper};

use std::fmt;
use std::io;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Server {
    pub build: Build,
    pub backends: Backends,
    executable: PathBuf,
}

impl Server {
    pub async fn list() -> Result<Vec<Self>, Error> {
        todo!()
    }

    pub async fn find(_build: Build, _backends: Backends) -> Option<Self> {
        todo!()
    }

    pub fn download(build: Build, backends: Backends) -> impl Straw<Self, Stage, Error> {
        sipper(async move |sender| {
            let cache = Cache::new(build);

            let artifacts = [Artifact::Server]
                .into_iter()
                .chain(backends.available().map(Artifact::Backend));

            let mut components = Vec::new();

            for artifact in artifacts {
                let component = cache
                    .download(artifact)
                    .with(|progress| Stage::Downloading(artifact, progress))
                    .run(sender.clone())
                    .await?;

                components.push(component);
            }

            let executable = cache.link(components).await?;

            Ok(Self {
                build,
                backends,
                executable,
            })
        })
    }

    pub async fn delete(_build: Build, _backends: Backends) -> Result<Self, Error> {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Downloading(Artifact, Progress),
}

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

        let Release { tag_name } = client
            .get(artifact::latest_release_url())
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let number = tag_name
            .trim_start_matches('b')
            .parse()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

        Ok(Self(number))
    }

    pub fn locked(number: u32) -> Self {
        Self(number)
    }

    pub fn number(self) -> u32 {
        self.0
    }
}

impl fmt::Display for Build {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b{}", self.0)
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Backends: u32 {
        const CUDA = 1;
        const HIP = 1 << 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Backend {
    Cuda,
    Hip,
}

impl Backends {
    pub fn available(self) -> impl Iterator<Item = Backend> {
        let mut backends = Vec::new();

        if cfg!(target_os = "macos") {
            return backends.into_iter();
        }

        if self.contains(Self::CUDA) {
            backends.push(Backend::Cuda);
        }

        if self.contains(Self::HIP) {
            backends.push(Backend::Hip);
        }

        backends.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn it_works() -> Result<(), Error> {
        let build = Build::latest().await?;
        assert!(build.number() > 0);

        let server = Server::download(build, Backends::all()).await?;
        assert_eq!(server.build, build);
        assert_eq!(server.backends, Backends::all());

        Ok(())
    }
}
