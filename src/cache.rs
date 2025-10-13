use crate::{Artifact, Backend, Build, Error, Progress};

use bytes::Bytes;
use sipper::{Sipper, Straw, sipper};
use tokio::fs;
use tokio::task;

use std::collections::BTreeSet;
use std::io;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Cache {
    path: PathBuf,
    build: Build,
}

impl Cache {
    pub fn new(build: Build) -> Self {
        Self {
            path: directories::ProjectDirs::from("", "hecrj", "llama-server")
                .expect("valid project directory")
                .cache_dir()
                .join(build.to_string())
                .to_path_buf(),
            build,
        }
    }

    pub fn download(&self, artifact: Artifact) -> impl Straw<Component, Progress, Error> {
        sipper(async move |sender| {
            let component = match artifact {
                Artifact::Server => Component::Server,
                Artifact::Backend(backend) => Component::Backend(backend),
            };

            if !fs::try_exists(self.path.join(component.directory())).await? {
                let bytes = artifact.download(self.build).run(sender).await?;

                task::spawn_blocking({
                    let cache = self.clone();

                    move || cache.extract(component, bytes)
                })
                .await??;
            }

            Ok(component)
        })
    }

    pub async fn link(
        &self,
        components: impl IntoIterator<Item = Component>,
    ) -> Result<PathBuf, Error> {
        let instance = Instance::new(components);
        let path = self.path.join(instance.directory());

        if !fs::try_exists(&path).await? {
            fs::create_dir(&path).await?;

            for component in instance.components {
                let mut read_component =
                    fs::read_dir(self.path.join(component.directory())).await?;

                while let Some(entry) = read_component.next_entry().await? {
                    let entry_path = entry.path();

                    let Some(file_name) = entry_path.file_name() else {
                        continue;
                    };

                    let dest_path = path.join(file_name);

                    if fs::try_exists(&dest_path).await? {
                        continue;
                    }

                    fs::hard_link(entry_path, dest_path).await?;
                }
            }
        }

        Ok(path.join(if cfg!(target_os = "windows") {
            "llama-server.exe"
        } else {
            "llama-server"
        }))
    }

    fn extract(&self, component: Component, bytes: Bytes) -> Result<(), Error> {
        let directory = self.path.join(component.directory());

        let mut archive = zip::ZipArchive::new(io::Cursor::new(bytes))?;
        archive.extract(directory)?;

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Component {
    Server,
    Backend(Backend),
}

impl Component {
    fn directory(self) -> &'static str {
        match self {
            Self::Server => "server",
            Self::Backend(backend) => match backend {
                Backend::Cuda => "backend-cuda",
                Backend::Hip => "backend-hip",
            },
        }
    }
}

struct Instance {
    components: BTreeSet<Component>,
}

impl Instance {
    fn new(components: impl IntoIterator<Item = Component>) -> Self {
        let mut components = BTreeSet::from_iter(components.into_iter());
        components.insert(Component::Server);

        Self { components }
    }

    fn directory(&self) -> String {
        self.components
            .iter()
            .map(|component| component.directory().trim_start_matches("backend-"))
            .collect::<Vec<_>>()
            .join("-")
    }
}
