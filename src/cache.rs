use crate::{Artifact, Backend, Build, Error, Progress};

use sipper::{Sipper, Straw, sipper};
use tokio::fs;
use tokio::io;
use tokio::task;

use std::collections::BTreeSet;
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Cache {
    path: PathBuf,
    build: Build,
}

impl Cache {
    pub fn new(build: Build) -> Self {
        Self {
            path: root().join(build.to_string()),
            build,
        }
    }

    pub async fn list() -> Result<Vec<Self>, Error> {
        fs::create_dir_all(root()).await?;

        let mut caches = Vec::new();
        let mut read_dir = fs::read_dir(root()).await?;

        while let Some(entry) = read_dir.next_entry().await? {
            if !entry.file_type().await?.is_dir() {
                continue;
            }

            let path = entry.path();

            let Some(name) = path.file_name() else {
                continue;
            };

            let Ok(build) = name.to_string_lossy().parse() else {
                continue;
            };

            caches.push(Cache::new(build));
        }

        Ok(caches)
    }

    pub fn build(&self) -> Build {
        self.build
    }

    pub fn download(&self, artifact: Artifact) -> impl Straw<Component, Progress, Error> {
        sipper(async move |sender| {
            fs::create_dir_all(&self.path).await?;

            let component = match artifact {
                Artifact::Server => Component::Server,
                Artifact::Backend(backend) => Component::Backend(backend),
            };

            if !fs::try_exists(self.path.join(component.directory())).await? {
                let file = fs::File::create(self.path.join(component.archive())).await?;

                artifact
                    .download(self.build, &mut io::BufWriter::new(file))
                    .run(sender)
                    .await?;

                task::spawn_blocking({
                    let cache = self.clone();

                    move || cache.extract(component)
                })
                .await??;

                fs::remove_file(self.path.join(component.archive())).await?;
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
                    if !entry.file_type().await?.is_file() {
                        continue;
                    }

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

    pub async fn delete(self) -> Result<(), Error> {
        fs::remove_dir_all(self.path).await?;
        Ok(())
    }

    fn extract(&self, component: Component) -> Result<(), Error> {
        let directory = self.path.join(component.directory());
        let file = std::fs::File::open(self.path.join(component.archive()))?;

        let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))?;
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

    fn archive(self) -> String {
        format!("{}.zip", self.directory())
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

fn root() -> PathBuf {
    env::var("LLAMA_SERVER_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            directories::ProjectDirs::from("", "hecrj", "llama-server")
                .expect("valid project directory")
                .cache_dir()
                .to_path_buf()
        })
}
