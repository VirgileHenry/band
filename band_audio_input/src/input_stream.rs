/// An audio input stream is an abstraction over any audio sources.
/// The sources can be live audio, files, or even custom generated on the fly inputs.
///
/// The `AudioInputStream` trait allows the rest of the system to be abstract over
/// the input kind.
pub trait AudioInputStream {
    type InputError: std::error::Error;

    /// Get the sample rate of the audio input.
    fn sample_rate(&self) -> usize;

    /// Read the available samples of the audio input into the provided buffer.
    fn next_samples(&mut self, buffer: &mut [f32]) -> Result<NextSamplesResult, Self::InputError>;
}

/// Possible results when samples are requested.
///
/// We can either receive some samples, or need to wait, or we have reached the end of the input.
pub enum NextSamplesResult {
    /// Some samples have been returned into the provided buffer.
    ///
    /// The number of samples fed are given by the provided usize.
    Some(usize),

    /// No samples are available yet, but some might be later.
    Unavailable,

    /// We have reached the end of the input.
    ///
    /// This variant only make sense for non-streaming inputs like files.
    EndOfInput,
}
