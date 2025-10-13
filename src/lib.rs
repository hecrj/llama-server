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
use tokio::process;
use tokio::time::{self, Duration};

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

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

    pub async fn boot(self, model: impl AsRef<Path>, settings: Settings) -> Result<Process, Error> {
        let child = process::Command::new(self.executable)
            .args(
                format!(
                    "--model {model} --host {host} --port {port} --gpu-layers {gpu_layers} --jinja",
                    model = model.as_ref().display(),
                    host = settings.host,
                    port = settings.port,
                    gpu_layers = settings.gpu_layers,
                )
                .split_whitespace(),
            )
            .kill_on_drop(true)
            .spawn()?;

        let url = format!("http://{}:{}", settings.host, settings.port);

        loop {
            if let Ok(response) = http::client().get(format!("{url}/health")).send().await {
                if response.error_for_status().is_ok() {
                    break;
                }
            }

            time::sleep(Duration::from_secs(1)).await;
        }

        Ok(Process { url, _raw: child })
    }

    pub async fn delete(self) -> Result<Self, Error> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub port: u32,
    pub host: String,
    pub gpu_layers: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_owned(),
            gpu_layers: 80,
        }
    }
}

#[derive(Debug)]
pub struct Process {
    pub url: String,
    _raw: process::Child,
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

    use tokio::fs;
    use tokio::io;

    #[tokio::test]
    #[ignore]
    async fn it_works() -> Result<(), Error> {
        const MODEL_URL: &str = "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-UD-Q4_K_XL.gguf?download=true";
        const MODEL_FILE: &str = "Qwen3.gguf";

        let build = Build::latest().await.unwrap_or(Build::locked(6730));
        let server = Server::download(build, Backends::all()).await?;

        assert_eq!(server.build, build);
        assert_eq!(server.backends, Backends::all());

        if !fs::try_exists(MODEL_FILE).await? {
            let model = fs::File::create(MODEL_FILE).await?;
            http::download(MODEL_URL, &mut io::BufWriter::new(model)).await?;
        }

        let process = server.boot(MODEL_FILE, Settings::default()).await?;
        assert_eq!(process.url, "http://127.0.0.1:8080");

        Ok(())
    }
}
