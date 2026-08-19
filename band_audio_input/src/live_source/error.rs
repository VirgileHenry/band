#[derive(Debug)]
pub enum LiveSourceError {
    SourceDied,
}

impl std::fmt::Display for LiveSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceDied => write!(f, "Source died, no more samples available"),
        }
    }
}

impl std::error::Error for LiveSourceError {}

#[derive(Debug)]
pub enum LiveSourceConstructionError {
    Cpal(cpal::Error),
    UnsupportedFormat(cpal::SampleFormat),
    InvalidChannelCount { channel_count: u16 },
}

impl std::fmt::Display for LiveSourceConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpal(error) => write!(f, "Cpal error: {error}"),
            Self::UnsupportedFormat(format) => write!(f, "Unsupported source format: {format}"),
            Self::InvalidChannelCount { channel_count } => write!(f, "Invalid channel count: {channel_count}"),
        }
    }
}

impl std::error::Error for LiveSourceConstructionError {}

impl From<cpal::Error> for LiveSourceConstructionError {
    fn from(error: cpal::Error) -> Self {
        Self::Cpal(error)
    }
}
