pub struct FrameExtractor<I, const SAMPLE_RATE: usize>
where
    I: band_audio_input::AudioInputStream,
{
    /// Input stream that produces frames at the given sample rate.
    input_stream: I,
}

impl<I, const SAMPLE_RATE: usize> FrameExtractor<I, SAMPLE_RATE>
where
    I: band_audio_input::AudioInputStream,
{
    /// Create a new frame extractor reading from the given input stream.
    pub fn new(input_stream: I) -> Result<Self, FrameExtractorContructionError> {
        if input_stream.sample_rate() != SAMPLE_RATE {
            return Err(FrameExtractorContructionError::SampleRateMismatch {
                expected: SAMPLE_RATE,
                found: input_stream.sample_rate(),
            });
        }

        Ok(Self { input_stream })
    }
}

pub enum FrameExtractorContructionError {
    SampleRateMismatch { expected: usize, found: usize },
}
