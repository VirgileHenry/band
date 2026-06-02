//! Sanity test: feed pure sine waves through MelExtractor at several source
//! sample rates and verify the right mel band lights up regardless. The
//! resampler inside MelExtractor should make the output rate-invariant.
//!
//! Run with: cargo run --release --bin test_mel

use sound_stream::MelExtractor;
use std::f32::consts::PI;

fn main() {
    // Source sample rates we want to verify the pipeline against. The first
    // matches the model's native rate (no resampling). The others exercise the
    // upsample/downsample paths.
    let source_rates = [22050usize, 44100, 48000, 16000];
    let tones = [100.0_f32, 440.0, 1000.0, 4000.0, 8000.0];

    for &src_rate in &source_rates {
        println!("\n=== source rate: {src_rate} Hz ===");

        for &tone_hz in &tones {
            // 1 second of pure sine at the source rate.
            let samples: Vec<f32> = (0..src_rate)
                .map(|n| 0.5 * (2.0 * PI * tone_hz * n as f32 / src_rate as f32).sin())
                .collect();

            let mut ex = MelExtractor::new(src_rate).expect("construct extractor");
            ex.push(&samples);
            let frames: Vec<_> = ex.collect();

            if frames.is_empty() {
                println!("{tone_hz:>6.0} Hz: no frames emitted");
                continue;
            }

            // Average the mel values across the middle of the run. Skip the
            // first few frames — the FFT window is partly zero until the buffer
            // fills, and the resampler also has startup transients.
            let skip = (frames.len() / 4).max(5).min(frames.len() - 1);
            let mut avg = [0.0_f32; config::MEL_FREQ_COUNT];
            let used = &frames[skip..];
            for f in used {
                for (i, v) in f.iter().enumerate() {
                    avg[i] += v;
                }
            }
            for v in avg.iter_mut() {
                *v /= used.len() as f32;
            }

            let (peak_band, peak_db) =
                avg.iter().enumerate().fold(
                    (0usize, f32::NEG_INFINITY),
                    |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
                );
            let expected_band = hz_to_mel_band(tone_hz);

            // The tone has been resampled to TARGET_RATE before mel extraction,
            // so the expected band is determined by TARGET_RATE's filterbank.
            // Anything above Nyquist of the source rate (src/2) is unrecoverable
            // and the resampler's anti-aliasing filter will have killed it —
            // expect noise in the predicted band, not a clean peak.
            let above_src_nyquist = tone_hz >= src_rate as f32 / 2.0;
            let tag = if above_src_nyquist { " [above src Nyquist]" } else { "" };

            println!(
                "{tone_hz:>6.0} Hz: peak band = {peak_band:>3} (expected ≈ {expected_band:>3})  \
                 peak = {peak_db:>6.1} dB  next-loudest = {:>6.1} dB{tag}",
                second_loudest(&avg, peak_band),
            );
        }
    }
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn hz_to_mel_band(hz: f32) -> usize {
    let mel_min = hz_to_mel(config::FREQ_MIN as f32);
    let mel_max = hz_to_mel(config::FREQ_MAX as f32);
    let mel = hz_to_mel(hz);
    let t = (mel - mel_min) / (mel_max - mel_min);
    let band = (t * (config::MEL_FREQ_COUNT as f32 + 1.0) - 1.0).round() as i32;
    band.clamp(0, config::MEL_FREQ_COUNT as i32 - 1) as usize
}

fn second_loudest(avg: &[f32], peak: usize) -> f32 {
    avg.iter()
        .enumerate()
        .filter(|(i, _)| *i != peak)
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max)
}
