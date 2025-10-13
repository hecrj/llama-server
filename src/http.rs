use crate::Error;

use bytes::Bytes;
use sipper::{Straw, sipper};

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

pub fn download<'a>(
    url: impl reqwest::IntoUrl + Send + 'a,
) -> impl Straw<Bytes, Progress, Error> + 'a {
    sipper(move |mut progress| async move {
        let mut download = client().get(url).send().await?;
        let start = Instant::now();
        let total = download.content_length().unwrap_or_default();

        let mut buffer = Vec::with_capacity(total as usize);
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

            buffer.extend_from_slice(&chunk);
        }

        Ok(Bytes::from(buffer))
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Progress {
    pub downloaded: u64,
    pub total: u64,
    pub speed: u64,
}
