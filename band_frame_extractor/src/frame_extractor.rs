mod error;
#[cfg(test)]
mod test;

pub use error::FrameExtractorError;

/// Structure to read a sound input and convert it to frames of frequencies.
///
/// Fourier Transform is used to convert a sliding window of sound samples
/// to the frequencies in that sliding window.
///
/// All params are const generics, so that a hard type alias can be made
/// and every other part uses the same config. This avoid bugs where the model is
/// not trained on the same sample rate as the live audio, for instance
pub struct FrameExtractor<I, const SAMPLE_RATE: usize, const WINDOW_SIZE: usize, const HOP: usize, const N_BINS: usize>
where
    I: band_audio_input::AudioInputStream,
{
    /// Input stream that produces frames at the given sample rate.
    input_stream: I,

    /// Real to complex FFT, cached for reusability
    planned_fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>,

    /// Sliding window over the audio samples.
    sample_window: [f32; WINDOW_SIZE],
    /// Number of samples in the sample window, from 0 to sample_window_end.
    sample_window_end: usize,

    /// Cached hann window for filtering the FFT
    hann_window: [f32; WINDOW_SIZE],

    /// Scratch and input buffer for the FFT
    fft_scratch_input: [f32; WINDOW_SIZE],
    /// Output buffer for the FFT
    fft_output: [realfft::num_complex::Complex<f32>; N_BINS],

    /// Tracker over the number of frames sent
    next_frame_index: usize,
}

impl<I, const SAMPLE_RATE: usize, const WINDOW_SIZE: usize, const HOP: usize, const N_BINS: usize>
    FrameExtractor<I, SAMPLE_RATE, WINDOW_SIZE, HOP, N_BINS>
where
    I: band_audio_input::AudioInputStream,
{
    /// Create a new frame extractor reading from the given input stream.
    pub fn new(input_stream: I) -> Result<Self, error::FrameExtractorConstructionError> {
        assert_eq!(WINDOW_SIZE / 2 + 1, N_BINS, "N_BINS shall match WINDOW_SIZE / 2 + 1");

        if input_stream.sample_rate() != SAMPLE_RATE {
            return Err(error::FrameExtractorConstructionError::SampleRateMismatch {
                expected: SAMPLE_RATE,
                found: input_stream.sample_rate(),
            });
        }

        let mut fft_planner = realfft::RealFftPlanner::new();
        let planned_fft = fft_planner.plan_fft_forward(WINDOW_SIZE);

        tracing::info!("Created FrameExtractor<SR {}, HOP {}, BINS {}>", SAMPLE_RATE, HOP, N_BINS);

        Ok(Self {
            input_stream,

            planned_fft,

            sample_window: [0.0; WINDOW_SIZE],
            sample_window_end: 0,
            hann_window: Self::build_hann_window::<WINDOW_SIZE>(),
            fft_scratch_input: [0.0; WINDOW_SIZE],
            fft_output: [realfft::num_complex::c32(0.0, 0.0); N_BINS],

            next_frame_index: 0,
        })
    }

    /// Get the next spectral frame.
    ///
    /// This will attempt to read the underlying input stream to fill up the sample window,
    /// then will perform the FFT on said window.
    ///
    /// If the underlying source is unavailable or at the end of the input, the same result will be propagated.
    pub fn next_frame(&mut self) -> Result<NextFrameResult<N_BINS>, FrameExtractorError<I>> {
        loop {
            let buffer: &mut [f32] = &mut self.sample_window[self.sample_window_end..];

            match self.input_stream.next_samples(buffer) {
                Ok(band_audio_input::NextSamplesResult::Some(written_sample_count)) => {
                    /* Some samples are received: check if we have enough for FFT, otherwise, the loop will ask for more */
                    self.sample_window_end += written_sample_count;
                    if self.sample_window_end == self.sample_window.len() {
                        /* We have enough samples to perform the fft, go for it */
                        let spectral_frame = self.fft()?;
                        /* Move the samples, since we used them */
                        self.hop();
                        return Ok(NextFrameResult::Frame(spectral_frame));
                    }
                }
                Ok(band_audio_input::NextSamplesResult::Unavailable) => return Ok(NextFrameResult::Unavailable),
                Ok(band_audio_input::NextSamplesResult::EndOfInput) => return Ok(NextFrameResult::EndOfInput),
                Err(error) => return Err(FrameExtractorError::AudioInput(error)),
            }
        }
    }

    /// Performs the FFT on the input sample_window and return the result as a spectral frame.
    fn fft(&mut self) -> Result<crate::SpectralFrame<N_BINS>, realfft::FftError> {
        /* Fill the scratch input with the sample window and the hann window */
        for i in 0..WINDOW_SIZE {
            self.fft_scratch_input[i] = self.sample_window[i] * self.hann_window[i];
        }

        /* Compute the FFT */
        self.planned_fft.process(&mut self.fft_scratch_input, &mut self.fft_output)?;

        /* Create the spectral frame, the feature we want is the magnitude of the complex number */
        let result = crate::SpectralFrame::from_complex(self.next_frame_index, &self.fft_output);
        self.next_frame_index += 1;
        Ok(result)
    }

    /// Performs a hop, moves the samples of the sample window to the left.
    fn hop(&mut self) {
        /* memmove the range HOP..end to 0..end-HOP */
        self.sample_window.copy_within(HOP.., 0);
        /* Reduce the size of the window */
        self.sample_window_end = self.sample_window_end.saturating_sub(HOP);
    }

    fn build_hann_window<const SIZE: usize>() -> [f32; SIZE] {
        std::array::from_fn(|i| {
            let x = std::f32::consts::TAU * i as f32 / SIZE as f32;
            0.5 * (1.0 - x.cos())
        })
    }
}

pub enum NextFrameResult<const N_BINS: usize> {
    Frame(crate::SpectralFrame<N_BINS>),
    Unavailable,
    EndOfInput,
}
