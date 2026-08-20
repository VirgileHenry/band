use crate::FrameExtractor;
use crate::NextFrameResult;
use band_audio_input::sin::SinInputStream;

/* One canonical test config, mirroring the production alias */
const SR: usize = 22050;
const WIN: usize = 2048;
const HOP: usize = 256;
const BINS: usize = WIN / 2 + 1; // 1025

type TestExtractor = FrameExtractor<SinInputStream, SR, WIN, HOP, BINS>;

fn extractor(freq: f32, samples: usize) -> TestExtractor {
    let sine = SinInputStream::new(freq, SR, Some(samples));
    FrameExtractor::new(sine).expect("construction")
}

/// Drain a finite source to EndOfInput. Panics on Unavailable:
/// a bounded sine must never starve.
fn drain_all<I, const A: usize, const B: usize, const C: usize, const D: usize>(
    ex: &mut FrameExtractor<I, A, B, C, D>,
) -> Vec<crate::SpectralFrame<D>>
where
    I: band_audio_input::AudioInputStream,
{
    let mut frames = Vec::new();
    loop {
        match ex.next_frame() {
            Ok(NextFrameResult::Frame(f)) => frames.push(f),
            Ok(NextFrameResult::Unavailable) => panic!("finite source reported Unavailable"),
            Ok(NextFrameResult::EndOfInput) => return frames,
            Err(_) => panic!("extractor error"),
        }
    }
}

/// The frequency a bin index represents: bin * SR / WIN.
fn bin_to_hz(bin: usize) -> f32 {
    bin as f32 * SR as f32 / WIN as f32
}

#[test]
fn sine_peaks_in_the_right_bin() {
    /* Expected bin: round(440 * WIN / SR) = round(40.87) = 41 */
    let expected_bin = (440.0 * WIN as f32 / SR as f32).round() as usize;

    let mut ex = extractor(440.0, SR); /* 1 second */
    let frames = drain_all(&mut ex);
    assert!(!frames.is_empty());

    for frame in &frames {
        let (max_bin, max_val) = frame
            .bins()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, v)| (i, *v))
            .unwrap();

        /* ±1 bin: 440 Hz falls between bin centers; leakage may tip
        the argmax to the neighbor. Wider than that = a real bug. */
        assert!(
            (max_bin as i64 - expected_bin as i64).abs() <= 1,
            "frame {}: peak at bin {max_bin} ({} Hz), expected ~{expected_bin} ({} Hz)",
            frame.index(),
            bin_to_hz(max_bin),
            bin_to_hz(expected_bin),
        );

        /* Pins the normalization chain: FFT scaling x Hann coherent
        gain. Loose bounds: energy splits between adjacent bins
        when the tone is off bin-center. Breaks hard (~512x) if
        normalization is missing — which is its main job. */
        assert!(
            (0.5..=1.2).contains(&max_val),
            "frame {}: peak magnitude {max_val}, expected ~1.0",
            frame.index(),
        );
    }
}

#[test]
fn frame_count_matches_hop_arithmetic() {
    let n = SR; /* 1 second */
    let mut ex = extractor(440.0, n);
    let frames = drain_all(&mut ex);

    /* Complete windows in n samples: floor((n - WIN) / HOP) + 1 */
    let expected = (n - WIN) / HOP + 1;
    assert_eq!(frames.len(), expected);
}

#[test]
fn frame_indices_are_contiguous_from_zero() {
    let mut ex = extractor(440.0, SR / 2);
    for (i, frame) in drain_all(&mut ex).iter().enumerate() {
        assert_eq!(frame.index(), i);
    }
}

#[test]
fn input_shorter_than_window_yields_no_frames() {
    let mut ex = extractor(440.0, WIN - 1);
    assert!(drain_all(&mut ex).is_empty());
}

#[test]
fn input_of_exactly_one_window_yields_one_frame() {
    let mut ex = extractor(440.0, WIN);
    assert_eq!(drain_all(&mut ex).len(), 1);
}

#[test]
fn tail_shorter_than_hop_is_dropped() {
    /* WIN + HOP - 1 samples: frame 0 fills, then HOP-1 remain — not
    enough to slide. Exactly one frame; tail dropped, not padded. */
    let mut ex = extractor(440.0, WIN + HOP - 1);
    assert_eq!(drain_all(&mut ex).len(), 1);
}

#[test]
fn low_frequency_lands_low() {
    /* 60 Hz kick territory -> bin round(60*2048/22050) = 6. The test
    that motivated WIN=2048: at WIN=512 this would be bin ~1.4,
    indistinguishable from DC. */
    let expected_bin = (60.0 * WIN as f32 / SR as f32).round() as usize;
    let mut ex = extractor(60.0, SR);
    for frame in drain_all(&mut ex) {
        let max_bin = frame
            .bins()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            (max_bin as i64 - expected_bin as i64).abs() <= 1,
            "frame {}: 60 Hz peaked at bin {max_bin}, expected ~{expected_bin}",
            frame.index(),
        );
    }
}

#[test]
fn energy_is_concentrated_not_smeared() {
    /* Windowing sanity: with Hann, a pure tone's energy sits in a few
    bins around the peak. If someone deletes the window multiply,
    leakage spreads energy everywhere and this ratio collapses. */
    let expected_bin = (440.0 * WIN as f32 / SR as f32).round() as usize;
    let mut ex = extractor(440.0, SR / 2);
    for frame in drain_all(&mut ex) {
        let f = frame.bins();
        let near: f32 = (expected_bin - 3..=expected_bin + 3).map(|i| f[i]).sum();
        let total: f32 = f.iter().sum();
        assert!(
            near / total > 0.8,
            "frame {}: only {:.1}% of energy near the peak — window missing?",
            frame.index(),
            100.0 * near / total,
        );
    }
}

#[test]
fn sample_rate_mismatch_is_rejected() {
    let wrong_rate = SinInputStream::new(440.0, 48_000, Some(1000));
    assert!(FrameExtractor::<_, SR, WIN, HOP, BINS>::new(wrong_rate).is_err());
}

/// The triangle test: generator (48 kHz) -> resampler (-> 22050) ->
/// extractor. Three independently-tested components must agree on
/// where 440 Hz lives.
#[test]
fn resampled_sine_lands_in_the_same_bin() {
    use band_audio_input::Resampler;

    let sine = SinInputStream::new(440.0, 48_000, Some(96_000)); /* 2 s */
    let resampled = Resampler::<_, SR>::new(sine).expect("resampler");
    let mut ex = FrameExtractor::<_, SR, WIN, HOP, BINS>::new(resampled).expect("extractor");

    let frames = drain_all(&mut ex);
    let expected_bin = (440.0 * WIN as f32 / SR as f32).round() as usize;

    /* Skip early frames: resampler startup transient. */
    for frame in frames.iter().skip(10) {
        let max_bin = frame
            .bins()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            (max_bin as i64 - expected_bin as i64).abs() <= 1,
            "frame {}: resampled 440 Hz peaked at bin {max_bin}, expected ~{expected_bin}",
            frame.index(),
        );
    }
    assert!(frames.len() > 20, "resampled stream produced too few frames");
}
