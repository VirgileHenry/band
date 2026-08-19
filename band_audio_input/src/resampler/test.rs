use super::Resampler;
use crate::AudioInputStream;
use crate::NextSamplesResult;
use crate::sin::SinInputStream;

/// Drain a stream to completion, pulling with a fixed caller buffer size.
/// Panics on Unavailable: finite test sources must never starve.
fn drain_all<I: AudioInputStream>(stream: &mut I, pull_size: usize) -> Vec<f32>
where
    I::InputError: std::fmt::Debug,
{
    let mut out = Vec::new();
    let mut buf = vec![0.0f32; pull_size];
    loop {
        match stream.next_samples(&mut buf).expect("stream error") {
            NextSamplesResult::Some(n) => {
                assert!(n > 0, "Some(0) for non-empty buffer violates trait contract");
                out.extend_from_slice(&buf[..n]);
            }
            NextSamplesResult::Unavailable => {
                panic!("finite source reported Unavailable")
            }
            NextSamplesResult::EndOfInput => return out,
        }
    }
}

/// Frequency estimate by zero-crossing count. Good to ~1% on clean sines.
fn estimate_frequency(samples: &[f32], sample_rate: usize) -> f32 {
    let crossings = samples.windows(2).filter(|w| (w[0] <= 0.0) != (w[1] <= 0.0)).count();
    let duration = samples.len() as f32 / sample_rate as f32;
    crossings as f32 / 2.0 / duration
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

#[test]
fn output_length_matches_ratio() {
    // 1 second at 48 kHz -> expect ~1 second at 22050.
    let input_len = 48_000;
    let sine = SinInputStream::new(440.0, 48_000, Some(input_len));
    let mut rs = Resampler::<_, 22050>::new(sine).expect("construction");
    let out = drain_all(&mut rs, 1000);

    let expected = input_len * 22050 / 48_000; // 22050
    // Tolerance: filter delay + one output chunk of slack.
    // Exact equality is NOT expected (resampler delay, partial flush).
    let tolerance = 32;
    assert!(
        (out.len() as i64 - expected as i64).unsigned_abs() as usize <= tolerance,
        "got {} samples, expected ~{expected} (±{tolerance})",
        out.len()
    );
}

#[test]
fn frequency_is_preserved() {
    let sine = SinInputStream::new(440.0, 48_000, Some(96_000)); // 2 s
    let mut rs = Resampler::<_, 22050>::new(sine).expect("construction");
    let out = drain_all(&mut rs, 1000);

    // Skip the resampler's startup transient before measuring.
    let settled = &out[2048..];
    let f = estimate_frequency(settled, 22050);
    assert!((f - 440.0).abs() < 5.0, "estimated {f} Hz, expected ~440 Hz");
    // And the signal actually has energy (didn't get filtered away):
    // full-scale sine RMS = 1/sqrt(2) ~= 0.707.
    assert!(rms(settled) > 0.6, "rms {} too low", rms(settled));
}

#[test]
fn above_22050_nyquist_is_rejected() {
    // 15 kHz is legal at 48 kHz input but above 22050/2 = 11025 Hz.
    // A correct resampler filters it out; an interpolator aliases it
    // to ~7 kHz at full amplitude.
    let sine = SinInputStream::new(15_000.0, 48_000, Some(96_000));
    let mut rs = Resampler::<_, 22050>::new(sine).expect("construction");
    let out = drain_all(&mut rs, 1000);

    let settled = &out[2048..];
    assert!(
        rms(settled) < 0.05,
        "rms {} — above-Nyquist content leaked through (aliasing)",
        rms(settled)
    );
}

#[test]
fn eof_flush_preserves_tail_on_non_chunk_multiple() {
    // Length deliberately NOT a multiple of the 512 input chunk:
    // exercises the partial_len flush path.
    let input_len = 48_000 + 137;
    let sine = SinInputStream::new(440.0, 48_000, Some(input_len));
    let mut rs = Resampler::<_, 22050>::new(sine).expect("construction");
    let out = drain_all(&mut rs, 1000);

    let expected = input_len * 22050 / 48_000;
    let tolerance = 1024;
    assert!(
        (out.len() as i64 - expected as i64).unsigned_abs() as usize <= tolerance,
        "tail dropped? got {}, expected ~{expected}",
        out.len()
    );
}

#[test]
fn end_of_input_is_idempotent() {
    let sine = SinInputStream::new(440.0, 48_000, Some(4800));
    let mut rs = Resampler::<_, 22050>::new(sine).expect("construction");
    let _ = drain_all(&mut rs, 1000);

    let mut buf = vec![0.0f32; 64];
    for _ in 0..3 {
        assert!(matches!(
            rs.next_samples(&mut buf).expect("stream error"),
            NextSamplesResult::EndOfInput
        ));
    }
}

#[test]
fn tiny_caller_buffers_lose_nothing() {
    // Same input drained with pull_size 7 vs 1000 must yield identical output.
    let make = || {
        let sine = SinInputStream::new(440.0, 48_000, Some(20_000));
        Resampler::<_, 22050>::new(sine).expect("construction")
    };
    let a = drain_all(&mut make(), 7);
    let b = drain_all(&mut make(), 1000);
    assert_eq!(a.len(), b.len(), "length differs by pull size");
    // Deterministic pipeline: same input, same resampler => bit-identical.
    assert!(a.iter().zip(&b).all(|(x, y)| x == y), "output depends on caller buffer size");
}
