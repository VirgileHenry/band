#[derive(Debug)]
pub enum ResamplerConstructionError {
    Rubato(rubato::ResamplerConstructionError),
}

impl std::fmt::Display for ResamplerConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rubato(error) => write!(f, "Failed to construct rubato FFT resampler: {error}"),
        }
    }
}

impl std::error::Error for ResamplerConstructionError {}

impl From<rubato::ResamplerConstructionError> for ResamplerConstructionError {
    fn from(error: rubato::ResamplerConstructionError) -> Self {
        Self::Rubato(error)
    }
}

#[derive(Debug)]
pub enum ResamplerError<I>
where
    I: crate::AudioInputStream,
{
    InputStreamError(I::InputError),
    ResampleError(rubato::ResampleError),
    BufferSizeError(rubato::audioadapter_buffers::SizeError),
}

impl<I> std::fmt::Display for ResamplerError<I>
where
    I: crate::AudioInputStream,
    I::InputError: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputStreamError(error) => write!(f, "Inner input stream error: {error}"),
            Self::ResampleError(error) => write!(f, "Failed to resample: {error}"),
            Self::BufferSizeError(error) => write!(f, "Invalid buffer sizes: {error}"),
        }
    }
}

impl<I> std::error::Error for ResamplerError<I>
where
    I: crate::AudioInputStream + std::fmt::Debug,
    I::InputError: std::error::Error,
{
}

impl<I> From<rubato::ResampleError> for ResamplerError<I>
where
    I: crate::AudioInputStream,
{
    fn from(error: rubato::ResampleError) -> Self {
        Self::ResampleError(error)
    }
}

impl<I> From<rubato::audioadapter_buffers::SizeError> for ResamplerError<I>
where
    I: crate::AudioInputStream,
{
    fn from(error: rubato::audioadapter_buffers::SizeError) -> Self {
        Self::BufferSizeError(error)
    }
}
