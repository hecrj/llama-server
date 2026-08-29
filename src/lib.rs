//! Download, embed, and run llama.cpp in your Rust projects.
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
use url::Url;

use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;

/// A server instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Server {
    /// The specific [`Build`] of the [`Server`].
    pub build: Build,
    /// The available backends of the [`Server`].
    pub backends: backend::Set,
    /// The path to the executable binary of the [`Server`].
    pub executable: PathBuf,
}

impl Server {
    /// Lists all the [`Server`] builds installed in the system.
    pub async fn list() -> Result<Vec<Build>, Error> {
        let mut builds: Vec<_> = Cache::list().await?.iter().map(Cache::build).collect();

        builds.sort();

        Ok(builds)
    }

    /// Download and installs the given [`Build`] of a [`Server`] with the given backends.
    pub fn download(build: Build, backends: backend::Set) -> impl Straw<Self, Download, Error> {
        sipper(async move |sender| {
            let cache = Cache::new(build);

            let artifacts = [Artifact::Server]
                .into_iter()
                .chain(backends.available().map(Artifact::Backend));

            let mut components = Vec::new();

            for artifact in artifacts {
                let component = cache
                    .download(artifact)
                    .with(|progress| Download { artifact, progress })
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

    /// Boots an [`Instance`] of the [`Server`] using the given model.
    pub async fn boot(
        &self,
        models_dir: impl AsRef<Path>,
        settings: Settings,
    ) -> Result<Instance, Error> {
        let models_dir = models_dir.as_ref();

        let process = process::Command::new(&self.executable)
            .args(
                format!(
                    "--host {host} --port {port} --models_dir {models_dir}",
                    host = settings.host,
                    port = settings.port,
                    models_dir = models_dir.display(),
                )
                .split_whitespace(),
            )
            .stdin(settings.stdin)
            .stdout(settings.stdout)
            .stderr(settings.stderr)
            .kill_on_drop(true)
            .spawn()?;

        let url = Url::parse(&format!(
            "http://{host}:{port}",
            host = settings.host,
            port = settings.port
        ))?;

        Ok(Instance {
            url,
            models_dir: models_dir.to_path_buf(),
            process,
        })
    }

    /// Deletes the [`Server`] installation with the given [`Build`].
    pub async fn delete(build: Build) -> Result<(), Error> {
        Cache::new(build).delete().await
    }
}

/// The configurable options of a new [`Instance`].
#[derive(Debug)]
pub struct Settings {
    /// The host URI that should be listened to by the [`Instance`].
    pub host: String,
    /// The host port to the [`Instance`] should be binded to.
    pub port: u32,
    /// The standard input stream.
    pub stdin: Stdio,
    /// The standard output stream.
    pub stdout: Stdio,
    /// The standard error stream.
    pub stderr: Stdio,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 9931,
            stdin: Stdio::null(),
            stdout: Stdio::null(),
            stderr: Stdio::null(),
        }
    }
}

/// An active [`Server`].
#[derive(Debug)]
pub struct Instance {
    /// The URL of the [`Instance`].
    pub url: Url,
    /// The directory containing model files of the [`Instance`].
    pub models_dir: PathBuf,
    /// The process of the [`Instance`].
    pub process: process::Child,
}

impl Instance {
    /// The URL of the [`Instance`].
    pub fn url(&self) -> &Url {
        &self.url
    }

    /// Waits until the [`Instance`] is warmed up and ready to receive requests.
    pub async fn wait_until_ready(&mut self) -> Result<(), Error> {
        loop {
            if let Some(status) = self.process.try_wait()? {
                return Err(io::Error::other(format!(
                    "llama-server exited unexpectedly: {status}"
                ))
                .into());
            }

            if let Ok(response) = http::client()
                .get(format!("{}health", self.url))
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

/// The download state of a [`Server`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Download {
    /// The [`Artifact`] being downloaded.
    pub artifact: Artifact,
    /// The download [`Progress`].
    pub progress: Progress,
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokio::fs;
    use tokio::io;

    use std::env;

    #[tokio::test]
    #[ignore]
    async fn it_works() -> Result<(), Error> {
        const MODEL_URL: &str = "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-UD-Q2_K_XL.gguf?download=true";
        const MODEL_FILE: &str = "Qwen3.gguf";

        let is_ci = env::var("CI").as_deref() == Ok("true");

        let models_dir = env::current_dir()?.join("models");
        let model_file = models_dir.join(MODEL_FILE);

        if !fs::try_exists(&models_dir).await? {
            fs::create_dir(&models_dir).await?;
        }

        if !fs::try_exists(&model_file).await? {
            let model = fs::File::create(&model_file).await?;
            http::download(MODEL_URL, &mut io::BufWriter::new(model)).await?;
        }

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

        let mut instance = server
            .boot(
                models_dir,
                Settings {
                    stdout: Stdio::inherit(),
                    stderr: Stdio::inherit(),
                    ..Settings::default()
                },
            )
            .await?;
        instance.wait_until_ready().await?;
        assert_eq!(instance.url().as_str(), "http://127.0.0.1:9931/");

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
