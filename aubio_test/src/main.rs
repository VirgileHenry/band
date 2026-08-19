//! WARNING: clanker bullshit ahead

// Multi-band "instrument" monitor: splits system audio into frequency bands
// (kick / bass / snare-mids / hats), runs a separate aubio onset detector per
// band, and tracks a smoothed energy envelope per band to distinguish:
//
//   HIT   = onset just fired in this band (flash)
//   HELD  = sustained energy above the gate, no fresh onset (e.g. bass note ringing)
//   ----  = silent
//
// Honest caveat: this is band-splitting + heuristics, NOT source separation.
// Kick and bass overlap in frequency; the thing that mostly disambiguates them
// here is temporal behavior (kick = onset then decay, bass = held energy).
//
// Cargo.toml:
// [dependencies]
// aubio-rs = "0.2"        # (or with "bundled" + CFLAGS fix)
// cpal = "0.18"
// anyhow = "1"

use aubio_rs::{Onset, OnsetMode};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc;

const WIN_SIZE: usize = 1024;
const HOP_SIZE: usize = 512;
const DISPLAY_EVERY_N_HOPS: u32 = 3; // ~30 fps at 48kHz/512

// ---------------------------------------------------------------------------
// RBJ biquad filters (audio-eq-cookbook)
// ---------------------------------------------------------------------------
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn bandpass(fs: f32, lo: f32, hi: f32) -> Self {
        let f0 = (lo * hi).sqrt(); // geometric center
        let q = f0 / (hi - lo); // Q from bandwidth
        let w0 = 2.0 * std::f32::consts::PI * f0 / fs;
        let alpha = w0.sin() / (2.0 * q);
        let (b0, b1, b2) = (alpha, 0.0, -alpha); // constant 0 dB peak
        let (a0, a1, a2) = (1.0 + alpha, -2.0 * w0.cos(), 1.0 - alpha);
        Self::normalized(b0, b1, b2, a0, a1, a2)
    }

    fn highpass(fs: f32, f0: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * f0 / fs;
        let alpha = w0.sin() / (2.0 * q);
        let c = w0.cos();
        let (b0, b1, b2) = ((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0);
        let (a0, a1, a2) = (1.0 + alpha, -2.0 * c, 1.0 - alpha);
        Self::normalized(b0, b1, b2, a0, a1, a2)
    }

    fn normalized(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

// ---------------------------------------------------------------------------
// One band = filters + onset detector + envelope follower + display state
// ---------------------------------------------------------------------------
struct Band {
    name: &'static str,
    range: &'static str,
    filters: Vec<Biquad>, // cascaded for steeper slopes
    onset: Onset,
    env: f32,     // smoothed RMS envelope (linear)
    gate_db: f32, // "held" threshold
    flash: u32,   // hops remaining to display HIT
    scratch: Vec<f32>,
}

impl Band {
    fn new(
        name: &'static str,
        range: &'static str,
        filters: Vec<Biquad>,
        mode: OnsetMode,
        threshold: f32,
        minioi_ms: f32,
        gate_db: f32,
        sample_rate: u32,
    ) -> anyhow::Result<Self> {
        let mut onset = Onset::new(mode, WIN_SIZE, HOP_SIZE, sample_rate)?;
        onset.set_threshold(threshold);
        onset.set_silence(-55.0);
        onset.set_minioi_ms(minioi_ms);
        Ok(Self {
            name,
            range,
            filters,
            onset,
            env: 0.0,
            gate_db,
            flash: 0,
            scratch: vec![0.0; HOP_SIZE],
        })
    }

    fn process_hop(&mut self, hop: &[f32]) -> anyhow::Result<()> {
        // Filter into this band
        for (i, &x) in hop.iter().enumerate() {
            let mut s = x;
            for f in &mut self.filters {
                s = f.process(s);
            }
            self.scratch[i] = s;
        }

        // Onset detection on the band signal
        if self.onset.do_result(&self.scratch)? > 0.0 {
            self.flash = 12; // ~130ms at 48kHz/512
        } else if self.flash > 0 {
            self.flash -= 1;
        }

        // Envelope: instant attack, slow release
        let rms = (self.scratch.iter().map(|s| s * s).sum::<f32>() / HOP_SIZE as f32).sqrt();
        const RELEASE: f32 = 0.92; // per hop
        self.env = if rms > self.env { rms } else { self.env * RELEASE };
        Ok(())
    }

    fn env_db(&self) -> f32 {
        20.0 * self.env.max(1e-6).log10()
    }

    fn render(&self) -> String {
        let db = self.env_db();
        let (icon, state, color) = if self.flash > 0 {
            ("\u{25CF}", "HIT ", "\x1b[1;31m") // red
        } else if db > self.gate_db {
            ("\u{25AC}", "HELD", "\x1b[1;32m") // green
        } else {
            ("\u{00B7}", "----", "\x1b[2m") // dim
        };
        // 24-char bar over -60..0 dB
        let filled = (((db + 60.0) / 60.0).clamp(0.0, 1.0) * 24.0) as usize;
        let bar: String = "\u{2588}".repeat(filled) + &"\u{2591}".repeat(24 - filled);
        format!(
            "{color}{:<6}\x1b[0m {:<10} {color}{icon} {state}\x1b[0m  {bar} {:>6.1} dB",
            self.name, self.range, db
        )
    }
}

// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .input_devices()?
        .find(|d| {
            d.description()
                .map(|desc| desc.name().to_lowercase().contains("pipewire"))
                .unwrap_or(false)
        })
        .ok_or_else(|| anyhow::anyhow!("no pipewire device (set PIPEWIRE_NODE to a sink to capture its monitor)"))?;

    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate();
    let channels = config.channels() as usize;
    let fs = sample_rate as f32;

    println!("Capturing '{device}' @ {sample_rate} Hz, {channels} ch\n");

    let mut bands = vec![
        // Kick: low thump. Energy-based onset works well down here.
        Band::new(
            "KICK",
            "30-100Hz",
            vec![Biquad::bandpass(fs, 30.0, 100.0), Biquad::bandpass(fs, 30.0, 100.0)],
            OnsetMode::Energy,
            0.4,
            90.0,
            -35.0,
            sample_rate,
        )?,
        // Bass: overlaps kick; the HELD state is what makes it readable.
        Band::new(
            "BASS",
            "70-250Hz",
            vec![Biquad::bandpass(fs, 70.0, 250.0), Biquad::bandpass(fs, 70.0, 250.0)],
            OnsetMode::SpecFlux,
            0.5,
            120.0,
            -38.0,
            sample_rate,
        )?,
        // Snare / mids: broadband crack + vocals/guitars live here.
        Band::new(
            "SNARE",
            "250-2.5k",
            vec![Biquad::bandpass(fs, 250.0, 2500.0), Biquad::bandpass(fs, 250.0, 2500.0)],
            OnsetMode::SpecFlux,
            0.35,
            70.0,
            -40.0,
            sample_rate,
        )?,
        // Hats / cymbals: HFC loves bright transients.
        Band::new(
            "HATS",
            "5k+",
            vec![Biquad::highpass(fs, 5000.0, 0.707), Biquad::highpass(fs, 5000.0, 0.707)],
            OnsetMode::Hfc,
            0.3,
            45.0,
            -45.0,
            sample_rate,
        )?,
    ];

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let stream = device.build_input_stream(
        config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mono: Vec<f32> = data
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect();
            let _ = tx.send(mono);
        },
        |err| eprintln!("stream error: {err}"),
        None,
    )?;
    stream.play()?;

    print!("\x1b[?25l"); // hide cursor
    // Reserve display lines once
    for _ in 0..bands.len() {
        println!();
    }

    let mut buf: Vec<f32> = Vec::with_capacity(HOP_SIZE * 8);
    let mut hop_count: u32 = 0;

    for chunk in rx {
        buf.extend_from_slice(&chunk);
        while buf.len() >= HOP_SIZE {
            let hop: Vec<f32> = buf.drain(..HOP_SIZE).collect();
            for band in &mut bands {
                band.process_hop(&hop)?;
            }
            hop_count += 1;

            if hop_count % DISPLAY_EVERY_N_HOPS == 0 {
                // move cursor up and redraw in place
                print!("\x1b[{}A", bands.len());
                for band in &bands {
                    println!("\x1b[2K{}", band.render());
                }
                use std::io::Write;
                std::io::stdout().flush()?;
            }
        }
    }

    print!("\x1b[?25h"); // show cursor
    Ok(())
}
