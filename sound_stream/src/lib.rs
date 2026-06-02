//! Streaming mel spectrogram extraction.
//!
//! Push audio samples in, drain mel frames out. Same extractor is used by
//! the training preprocessor (push all samples of a WAV at once) and by live
//! inference (push samples from each cpal callback as they arrive).

use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use rubato::Resampler;
use std::sync::Arc;

pub struct MelExtractor {
    /// Number of samples we advance each frame.
    /// This is computed from the given sample rate and the desired FFT rate.
    resampler: Option<rubato::Fft<f32>>,

    /// FFT engine (planned once for the given fft_size).
    fft: Arc<dyn RealToComplex<f32>>,
    /// Buffer for the FFT input
    fft_input: [f32; config::FFT_SIZE],
    /// Buffer for the FFT output
    fft_output: [Complex32; config::FFT_FREQ_COUNT],

    /// Precomputed Hann window of length fft_size.
    hann: [f32; config::FFT_SIZE],

    /// Precomputed filterbank: filterbank[band] = (bin_start, weights_for_band).
    /// Storing only the nonzero range per band keeps the per-frame loop tight.
    filterbank: [MelBand; config::MEL_FREQ_COUNT],

    /// Raw input, at the sample rate of the raw input.
    raw_input: Vec<f32>,
    /// Sound samples to process.
    ///
    /// These samples have been resampled to the target sample rate.
    /// Once there is enough samples, we can perform the FFT and discard the first hop.
    resampled_input: Vec<f32>,

    /// Scratch buffer for the FFT
    fft_scratch: [Complex32; config::FFT_FREQ_COUNT],
    /// Scratch buffer for the magnitude results of the FFT
    mags_scratch: [f32; config::FFT_FREQ_COUNT],
    /// Scratch buffer for the computed mel bands
    mels_scratch: [f32; config::MEL_FREQ_COUNT],
}

struct MelBand {
    bin_start: usize,
    weight_count: usize,
    /// Most entries are zero, only 0..weight_count is used
    weights: [f32; config::FFT_FREQ_COUNT],
}

impl MelExtractor {
    pub fn new(input_sample_rate: usize) -> Result<Self, rubato::ResamplerConstructionError> {
        let resampler = if input_sample_rate == config::SAMPLING_RATE {
            None
        } else {
            Some(rubato::Fft::new(
                input_sample_rate,
                config::SAMPLING_RATE,
                128,
                2,
                1,
                rubato::FixedSync::Output,
            )?)
        };

        let mut planner = RealFftPlanner::<f32>::new();

        let fft = planner.plan_fft_forward(config::FFT_SIZE);
        let fft_input = [0.0; config::FFT_SIZE];
        let fft_output = [Complex32::ZERO; config::FFT_FREQ_COUNT];

        /* Periodic Hann window, matches librosa/scipy STFT convention. */
        let hann: [f32; config::FFT_SIZE] = std::array::from_fn(|n| {
            use std::f32::consts::PI;
            0.5 - 0.5 * (2.0 * PI * n as f32 / config::FFT_SIZE as f32).cos()
        });

        let filterbank = build_filterbank();

        let fft_scratch = [Complex32::ZERO; config::FFT_FREQ_COUNT];
        let mags_scratch = [0.0; config::FFT_FREQ_COUNT];
        let mels_scratch = [0.0; config::MEL_FREQ_COUNT];

        Ok(Self {
            resampler,
            fft,
            fft_input,
            fft_output,
            hann,
            filterbank,
            raw_input: Vec::new(),
            resampled_input: Vec::new(),
            fft_scratch,
            mags_scratch,
            mels_scratch,
        })
    }

    /// Append samples to the sliding window. Cheap — just memcpy + counter updates.
    pub fn push(&mut self, samples: &[f32]) {
        match self.resampler.as_mut() {
            None => self.resampled_input.extend_from_slice(samples),
            Some(resampler) => {
                /* store raw samples to be processed */
                self.raw_input.extend_from_slice(samples);

                /* Loop to resample the raw input to our target sample rate */
                loop {
                    use audioadapter_buffers::direct::InterleavedSlice;

                    let needed_in = resampler.input_frames_next();
                    if self.raw_input.len() < needed_in {
                        break;
                    }
                    let needed_out = resampler.output_frames_next();

                    let in_slice = &self.raw_input[..needed_in];
                    let in_adapter = InterleavedSlice::new(in_slice, 1, needed_in).unwrap();

                    let start = self.resampled_input.len();
                    self.resampled_input.resize(start + needed_out, 0.0);
                    let mut out_adapter = InterleavedSlice::new_mut(&mut self.resampled_input[start..], 1, needed_out).unwrap();

                    let (_, written) = resampler
                        .process_into_buffer(&in_adapter, &mut out_adapter, None)
                        .expect("resample");

                    self.resampled_input.truncate(start + written);
                    self.raw_input.drain(..needed_in);
                }
            }
        }
    }

    /// Compute one frame from the current fft_buffer contents.
    /// Does not advance the buffer; that's the caller's job.
    fn next_frame(&mut self) -> Option<[f32; config::MEL_FREQ_COUNT]> {
        /* Get the window, return if it's not available */
        let fft_window = self.resampled_input.get(0..config::FFT_SIZE)?;

        /* Store the FFT input: raw input samples scaled with Hann window */
        for i in 0..config::FFT_SIZE {
            self.fft_input[i] = fft_window[i] * self.hann[i];
        }

        /* If we extracted a window, we can discard the first 0..hop_size samples. */
        /* This will advance the sliding window. */
        self.resampled_input.drain(0..config::HOP_SIZE);

        /* Run the FFT */
        let input = &mut self.fft_input;
        let output = &mut self.fft_output;
        let scratch = &mut self.fft_scratch;
        let fft_res = self.fft.process_with_scratch(input, output, scratch);
        if let Err(e) = fft_res {
            log::error!("Failed to run the FFT: {e}");
            return None;
        }

        /* Get the magnitude of the results */
        for i in 0..config::FFT_FREQ_COUNT {
            let c = output[i];
            self.mags_scratch[i] = (c.re * c.re + c.im * c.im).sqrt();
        }

        /* Reset mels scratch */
        for i in 0..config::MEL_FREQ_COUNT {
            self.mels_scratch[i] = 0.0;
        }

        /* Apply filterbank → magnitude per mel band, then convert to dB. */
        for (band_idx, band) in self.filterbank.iter().enumerate() {
            let mut sum = 0.0_f32;
            for k in 0..band.weight_count {
                sum += self.mags_scratch[band.bin_start + k] * band.weights[k];
            }
            self.mels_scratch[band_idx] = 20.0 * sum.max(1e-10).log10();
        }

        Some(self.mels_scratch.clone())
    }
}

impl Iterator for MelExtractor {
    type Item = [f32; config::MEL_FREQ_COUNT];
    fn next(&mut self) -> Option<Self::Item> {
        self.next_frame()
    }
}

/// Fixme: review ?
fn build_filterbank() -> [MelBand; config::MEL_FREQ_COUNT] {
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10f32.powf(mel / 2595.0) - 1.0)
    }

    // n_mels+2 evenly-spaced points on the mel scale; each band uses three consecutive
    // points as (left, center, right) corners of a triangular filter.
    let mel_min = hz_to_mel(config::FREQ_MIN as f32);
    let mel_max = hz_to_mel(config::FREQ_MAX as f32);

    let mel_pts: [f32; config::MEL_FREQ_COUNT + 2] = std::array::from_fn(|i| {
        let t = i as f32 / (config::MEL_FREQ_COUNT + 1) as f32;
        mel_min + (mel_max - mel_min) * t
    });
    let hz_pts: [f32; config::MEL_FREQ_COUNT + 2] = std::array::from_fn(|i| mel_to_hz(mel_pts[i]));

    let bin_hz = |bin: usize| bin as f32 * config::SAMPLING_RATE as f32 / config::FFT_SIZE as f32;

    std::array::from_fn(|m| {
        let left = hz_pts[m];
        let center = hz_pts[m + 1];
        let right = hz_pts[m + 2];

        let mut weights = [0.0_f32; config::FFT_FREQ_COUNT];
        let mut bin_start = 0;
        let mut weight_count = 0;
        let mut found_start = false;

        for bin in 0..config::FFT_FREQ_COUNT {
            let f = bin_hz(bin);
            let w = if f <= left || f >= right {
                0.0
            } else if f <= center {
                (f - left) / (center - left)
            } else {
                (right - f) / (right - center)
            };

            if w > 0.0 {
                if !found_start {
                    bin_start = bin;
                    found_start = true;
                }
                weights[weight_count] = w;
                weight_count += 1;
            } else if found_start {
                break;
            }
        }

        /* Guard: ensure weight_count is at least 1 for the unreachable case */
        if weight_count == 0 {
            weight_count = 1;
        }

        MelBand {
            bin_start,
            weight_count,
            weights,
        }
    })
}
