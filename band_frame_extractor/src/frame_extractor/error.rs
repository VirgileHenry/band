#[derive(Debug)]
pub enum FrameExtractorConstructionError {
    SampleRateMismatch { expected: usize, found: usize },
}

impl std::fmt::Display for FrameExtractorConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SampleRateMismatch { expected, found } => write!(
                f,
                "Audio source has an invalid sample rate: expected {expected}, found {found}"
            ),
        }
    }
}

impl std::error::Error for FrameExtractorConstructionError {}

#[derive(Debug)]
pub enum FrameExtractorError<I>
where
    I: band_audio_input::AudioInputStream,
{
    AudioInput(I::InputError),
    Fft(realfft::FftError),
}

impl<I> std::fmt::Display for FrameExtractorError<I>
where
    I: band_audio_input::AudioInputStream,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AudioInput(error) => write!(f, "Error from the audio input stream: {error}"),
            Self::Fft(error) => write!(f, "Fft error: {error}"),
        }
    }
}

impl<I> std::error::Error for FrameExtractorError<I> where I: band_audio_input::AudioInputStream + std::fmt::Debug {}

impl<I> From<realfft::FftError> for FrameExtractorError<I>
where
    I: band_audio_input::AudioInputStream,
{
    fn from(error: realfft::FftError) -> Self {
        Self::Fft(error)
    }
}
