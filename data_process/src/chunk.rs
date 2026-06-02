//! On-disk format: per-frame mel + per-class onset + per-class active.
//!
//! Layout per frame:
//!   mels:    [f32; N_MELS]
//!   onsets:  [u8; config::INSTRUMENT_COUNT]    (0 or 1)
//!   actives: [u8; config::INSTRUMENT_COUNT]    (0 or 1)
//! All little-endian, no padding between frames.

use crate::data::DrumClass;
use crate::midi::DrumHit;

pub struct Frame {
    pub mels: [f32; config::MEL_FREQ_COUNT],
    pub onsets: [u8; config::INSTRUMENT_COUNT],
    pub actives: [u8; config::INSTRUMENT_COUNT],
}

/// Build onset + active label grids aligned to a given number of mel frames.
pub fn build_labels(
    hits: &[DrumHit],
    n_frames: usize,
) -> (Vec<[u8; config::INSTRUMENT_COUNT]>, Vec<[u8; config::INSTRUMENT_COUNT]>) {
    let mut onsets = vec![[0u8; config::INSTRUMENT_COUNT]; n_frames];
    let mut actives = vec![[0u8; config::INSTRUMENT_COUNT]; n_frames];

    for hit in hits {
        let frame = (hit.time_seconds * config::FFT_PROCESSING_RATE as f32).round() as usize;
        if frame >= n_frames {
            continue;
        }
        let cls = hit.class as usize;
        onsets[frame][cls] = 1;
        let end = (frame + config::TIME_CHUNK_LENGTH).min(n_frames);
        for f in frame..end {
            actives[f][cls] = 1;
        }
    }
    (onsets, actives)
}

/// Serialize one full track (interleaved mels + onsets + actives, frame by frame).
pub fn write(
    out_path: &std::path::Path,
    mels: &[[f32; config::MEL_FREQ_COUNT]],
    onsets: &[[u8; config::INSTRUMENT_COUNT]],
    actives: &[[u8; config::INSTRUMENT_COUNT]],
) -> std::io::Result<usize> {
    assert_eq!(mels.len(), onsets.len());
    assert_eq!(mels.len(), actives.len());

    let frame_bytes = config::MEL_FREQ_COUNT * 4 + config::INSTRUMENT_COUNT * 2;
    let mut bytes = Vec::with_capacity(mels.len() * frame_bytes);
    for ((mel, on), act) in mels.iter().zip(onsets).zip(actives) {
        for v in mel {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(on);
        bytes.extend_from_slice(act);
    }
    std::fs::write(out_path, &bytes)?;
    Ok(mels.len())
}
