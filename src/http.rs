use crate::Error;

use sipper::{Straw, sipper};
use tokio::io::AsyncWrite;

use std::sync::LazyLock;
use std::time::Instant;

pub fn client() -> reqwest::Client {
    static CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
        reqwest::Client::builder()
            .user_agent(format!(
                "{} {}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            ))
            .build()
            .expect("should be a valid client")
    });

    CLIENT.clone()
}

pub fn download<'a, W: AsyncWrite + Unpin>(
    url: impl reqwest::IntoUrl + Send + 'a,
    writer: &'a mut W,
) -> impl Straw<(), Progress, Error> + 'a {
    use tokio::io::AsyncWriteExt;

    sipper(move |mut progress| async move {
        let mut download = client().get(url).send().await?;
        let start = Instant::now();
        let total = download.content_length().unwrap_or_default();

        let mut downloaded = 0;

        progress
            .send(Progress {
                total,
                downloaded,
                speed: 0,
            })
            .await;

        while let Some(chunk) = download.chunk().await? {
            downloaded += chunk.len() as u64;
            let speed = (downloaded as f32 / start.elapsed().as_secs_f32()) as u64;

            progress
                .send(Progress {
                    total,
                    downloaded,
                    speed,
                })
                .await;

            writer.write_all(&chunk).await?;
        }

        writer.flush().await?;

        Ok(())
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Progress {
    pub downloaded: u64,
    pub total: u64,
    pub speed: u64,
}
