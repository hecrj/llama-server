use crate::http;
use crate::{Backend, Build, Error};

use sipper::Straw;
use tokio::io::AsyncWrite;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Artifact {
    Server,
    Backend(Backend),
}

impl Artifact {
    pub(crate) fn download<W: AsyncWrite + Unpin>(
        self,
        build: Build,
        writer: &mut W,
    ) -> impl Straw<(), http::Progress, Error> {
        let release_url = build.url();

        http::download(
            match self {
                Artifact::Server => format!("{release_url}/llama-server-{build}-{PLATFORM}.zip"),
                Artifact::Backend(backend) => {
                    let name = match backend {
                        Backend::Cuda => "cuda",
                        Backend::Hip => "hip",
                    };

                    format!("{release_url}/backend-{name}-{build}-{PLATFORM}.zip")
                }
            },
            writer,
        )
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const PLATFORM: &str = "linux-x64";

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const PLATFORM: &str = "macos-x64";

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const PLATFORM: &str = "macos-arm64";

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
const PLATFORM: &str = "windows-x64";
