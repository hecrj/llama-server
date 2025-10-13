use std::io;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Error {
    IOFailed(Arc<io::Error>),
    RequestFailed(Arc<reqwest::Error>),
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::IOFailed(Arc::new(error))
    }
}

impl From<Error> for io::Error {
    fn from(error: Error) -> Self {
        match error {
            Error::IOFailed(error) => io::Error::new(error.kind(), error),
            Error::RequestFailed(error) => io::Error::other(error),
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self::RequestFailed(Arc::new(error))
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(error: tokio::task::JoinError) -> Self {
        Error::IOFailed(Arc::new(io::Error::other(error)))
    }
}

impl From<zip::result::ZipError> for Error {
    fn from(error: zip::result::ZipError) -> Self {
        Error::IOFailed(Arc::new(io::Error::other(error)))
    }
}
