pub mod backend;

mod artifact;
mod build;
mod cache;
mod error;
mod http;

pub use artifact::Artifact;
pub use backend::Backend;
pub use build::Build;
pub use error::Error;
pub use http::Progress;

use crate::cache::Cache;

use sipper::{Sipper, Straw, sipper};
use tokio::process;
use tokio::time::{self, Duration};

use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Server {
    pub build: Build,
    pub backends: backend::Set,
    pub executable: PathBuf,
}

impl Server {
    pub async fn list() -> Result<Vec<Build>, Error> {
        let mut builds: Vec<_> = Cache::list().await?.iter().map(Cache::build).collect();

        builds.sort();

        Ok(builds)
    }

    pub fn download(build: Build, backends: backend::Set) -> impl Straw<Self, Stage, Error> {
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
                backends: backends.normalize(),
                executable,
            })
        })
    }

    pub async fn boot(
        &self,
        model: impl AsRef<Path>,
        settings: Settings,
    ) -> Result<Instance, Error> {
        let process = process::Command::new(&self.executable)
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
            .stdin(settings.stdin)
            .stdout(settings.stdout)
            .stderr(settings.stderr)
            .kill_on_drop(true)
            .spawn()?;

        Ok(Instance {
            host: settings.host,
            port: settings.port,
            process,
        })
    }

    pub async fn delete(build: Build) -> Result<(), Error> {
        Cache::new(build).delete().await
    }
}

#[derive(Debug)]
pub struct Settings {
    pub host: String,
    pub port: u32,
    pub gpu_layers: u32,
    pub stdin: Stdio,
    pub stdout: Stdio,
    pub stderr: Stdio,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 8080,
            gpu_layers: 80,
            stdin: Stdio::null(),
            stdout: Stdio::null(),
            stderr: Stdio::null(),
        }
    }
}

#[derive(Debug)]
pub struct Instance {
    pub host: String,
    pub port: u32,
    pub process: process::Child,
}

impl Instance {
    pub fn url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    pub async fn wait_until_ready(&mut self) -> Result<(), Error> {
        loop {
            if let Some(status) = self.process.try_wait()? {
                return Err(io::Error::other(format!(
                    "llama-server exited unexpectedly: {status}"
                )))?;
            }

            if let Ok(response) = http::client()
                .get(format!("{}/health", self.url()))
                .send()
                .await
                && response.error_for_status().is_ok()
            {
                break;
            }

            time::sleep(Duration::from_secs(1)).await;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Downloading(Artifact, Progress),
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokio::fs;
    use tokio::io;

    #[tokio::test]
    #[ignore]
    async fn it_works() -> Result<(), Error> {
        const MODEL_URL: &str = "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-UD-Q2_K_XL.gguf?download=true";
        const MODEL_FILE: &str = "Qwen3.gguf";

        let is_ci = std::env::var("CI").as_deref() == Ok("true");

        if is_ci {
            let installed = Server::list().await?;
            assert!(installed.is_empty());
        }

        let build = Build::latest().await.unwrap_or(Build::locked(6730));
        let server = Server::download(build, backend::Set::all()).await?;

        assert_eq!(server.build, build);
        assert_eq!(
            server.backends,
            if cfg!(target_os = "macos") {
                backend::Set::empty()
            } else {
                backend::Set::all()
            }
        );

        if !fs::try_exists(MODEL_FILE).await? {
            let model = fs::File::create(MODEL_FILE).await?;
            http::download(MODEL_URL, &mut io::BufWriter::new(model)).await?;
        }

        let mut instance = server
            .boot(
                MODEL_FILE,
                Settings {
                    stdout: Stdio::inherit(),
                    stderr: Stdio::inherit(),
                    ..Settings::default()
                },
            )
            .await?;
        instance.wait_until_ready().await?;
        assert_eq!(instance.url(), "http://127.0.0.1:8080");

        if is_ci {
            drop(instance);

            let installed = Server::list().await?;
            assert!(installed.len() == 1);
            assert_eq!(installed.first(), Some(&server.build));

            Server::delete(server.build).await?;

            let installed = Server::list().await?;
            assert!(installed.is_empty());
        }

        Ok(())
    }
}
