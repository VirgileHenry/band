#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
pub struct EgmdRow {
    pub drummer: String,
    pub session: String,
    pub id: String,
    pub style: String,
    pub bpm: u32,
    pub beat_type: String,
    pub time_signature: String,
    pub duration: f32,
    pub split: String,
    pub midi_filename: String,
    pub audio_filename: String,
    pub kit_name: String,
}

impl std::fmt::Display for EgmdRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} in {} ({}) at {}bpm ({} secs), wav: {}, midi: {}",
            self.drummer,
            self.session,
            self.style,
            self.kit_name,
            self.bpm,
            self.duration,
            self.audio_filename,
            self.midi_filename
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DrumClass {
    Kick = 0,
    Snare = 1,
    Rim = 2,
    HatClosed = 3,
    HatOpen = 4,
    Crash = 5,
    Ride = 6,
    Tom = 7,
}

impl DrumClass {
    pub const COUNT: usize = 8;

    pub fn from_midi(pitch: u8) -> Option<Self> {
        Some(match pitch {
            35 | 36 => Self::Kick,
            38 | 40 => Self::Snare,
            37 | 39 => Self::Rim,
            42 | 44 => Self::HatClosed,
            46 => Self::HatOpen,
            49 | 52 | 55 | 57 => Self::Crash,
            51 | 53 | 59 => Self::Ride,
            41 | 43 | 45 | 47 | 48 | 50 => Self::Tom,
            _ => return None,
        })
    }
}
