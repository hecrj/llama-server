use bitflags::bitflags;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Backend {
    Cuda,
    Hip,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Set: u32 {
        const CUDA = 1;
        const HIP = 1 << 1;
    }
}

impl Set {
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
