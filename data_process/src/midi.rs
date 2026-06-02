//! Parse an E-GMD MIDI file into a list of (time_seconds, drum_class) hits.

use crate::data::DrumClass;
use midly::{MetaMessage, Smf, TrackEventKind};

#[derive(Debug, Clone, Copy)]
pub struct DrumHit {
    pub time_seconds: f32,
    pub class: DrumClass,
}

pub fn parse(midi_path: &std::path::Path) -> std::io::Result<Vec<DrumHit>> {
    let bytes = std::fs::read(midi_path)?;
    let smf = Smf::parse(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    let ticks_per_beat = match smf.header.timing {
        midly::Timing::Metrical(t) => u16::from(t) as u32,
        midly::Timing::Timecode(_, _) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "SMPTE timing not supported (E-GMD uses metrical timing)",
            ));
        }
    };

    let mut hits = Vec::new();
    let mut us_per_beat: u32 = 500_000; // MIDI default = 120 BPM

    // E-GMD MIDI files have a single track; if there were multiple, we'd merge them
    // sorted by absolute tick. For single-track files, walking events is enough.
    for track in &smf.tracks {
        let mut abs_seconds: f64 = 0.0;
        for event in track {
            let delta_ticks = u32::from(event.delta);
            if delta_ticks > 0 {
                abs_seconds += delta_ticks as f64 * us_per_beat as f64 / 1_000_000.0 / ticks_per_beat as f64;
            }

            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(t)) => {
                    us_per_beat = u32::from(t);
                }
                TrackEventKind::Midi { channel, message } => {
                    // General MIDI drums are on channel 10 (zero-indexed: 9).
                    if u8::from(channel) != 9 {
                        continue;
                    }
                    if let midly::MidiMessage::NoteOn { key, vel } = message {
                        if u8::from(vel) > 0 {
                            if let Some(class) = DrumClass::from_midi(u8::from(key)) {
                                hits.push(DrumHit {
                                    time_seconds: abs_seconds as f32,
                                    class,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(hits)
}
