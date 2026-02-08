//! Pick your preferred compute backends.
use bitflags::bitflags;

/// A compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Backend {
    /// The NVIDIA CUDA backend.
    Cuda,
    /// The AMD HIP backend.
    Hip,
}

bitflags! {
    /// A set of compute backends.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Set: u32 {
        /// The NVIDIA CUDA backend.
        const CUDA = 1;
        /// The AMD HIP backend.
        const HIP = 1 << 1;
    }
}

impl Set {
    /// Returns the backends in the [`Set`] that are also available in the current
    /// platform.
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

    /// Returns a new [`Set`] with any unavailable backends filtered out.
    pub fn normalize(self) -> Self {
        self.available().fold(Self::empty(), |backends, backend| {
            backends
                | match backend {
                    Backend::Cuda => Self::CUDA,
                    Backend::Hip => Self::HIP,
                }
        })
    }
}
