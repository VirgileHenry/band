/// Sampling rate at which we perform all the conversion and prediction.
///
/// This is the number of audio samples per seconds we handle.
/// For input streams that have different rates, we use a resampler.
pub const SAMPLING_RATE: usize = 44100;

/// Number of times we want to process input audio per second.
///
/// The bigger, the faster we can read instruments onset, but the most expensive it is.
pub const FFT_PROCESSING_RATE: usize = 100;

/// Number of samples we move when processing the next frame.
pub const HOP_SIZE: usize = SAMPLING_RATE / FFT_PROCESSING_RATE;

/// Number of audio samples used for the FFT.
///
/// This shall always remain a fixed power of 2, so the fast FFT can be fast.
pub const FFT_SIZE: usize = 2048;

/// Number of frequencies output by the FFT.
///
/// This is equal to the fft_size / 2 + 1.
pub const FFT_FREQ_COUNT: usize = FFT_SIZE / 2 + 1;

/// Number of Mel frequency bins used for the prediction.
///
/// This directly impact the size of the input we have to process.
/// The more Mel bins, the better the quality, but the more processing is required.
pub const MEL_FREQ_COUNT: usize = 128;

/// Number of instruments we attempt to guess from the incoming audio.
///
/// Instruments are defined during training, they can be anything.
pub const INSTRUMENT_COUNT: usize = 8;

/// Minimum frequency we take into account in the filter (Hz)
pub const FREQ_MIN: usize = 30;

/// Maximum frequency we take into account in the filter (Hz)
pub const FREQ_MAX: usize = 11025;

/// Size of each chunk we process at once.
///
/// This is what gives context to the model.
pub const TIME_CHUNK_LENGTH: usize = 256;

/// Size of each chunk we process at once.
///
/// This is what gives context to the model.
pub const TIME_CHUNK_STRIDE: usize = 20;
