//! Live onset listener.
//!
//! Captures audio, feeds it through the streaming `MelExtractor`, keeps a
//! rolling window of the last `TIME_CHUNK_LENGTH` mel frames, runs the trained
//! model on that window every new frame, and prints per-instrument **onsets**
//! (not actives) in the terminal as they fire.
//!
//! Put this at `src/bin/listen.rs` and run:
//!     cargo run --release --bin listen            # capture default input (mic)
//!     cargo run --release --bin listen -- --list  # list devices
//!     cargo run --release --bin listen -- --device "Monitor"   # pick by name substring
//!     cargo run --release --bin listen -- --loopback           # capture default OUTPUT (Windows/WASAPI)
//!
//! ── ADJUST THESE TWO THINGS ──────────────────────────────────────────────
//!  1. `band` below = your library crate name (package name, `-` → `_`).
//!  2. In your lib.rs, make these reachable:
//!         pub use model::Model;
//!         pub use features::ChunkFeatures;
//!         pub use <mel_module>::MelExtractor;   // wherever MelExtractor lives
//!     (`ChunkPrediction` is only used as Model::forward's return type — no import needed.)
//! ─────────────────────────────────────────────────────────────────────────

mod input;

use std::collections::VecDeque;
use std::io::Write;
use std::sync::mpsc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::activation::sigmoid;

use sound_stream::MelExtractor;

/// Inference backend — plain Wgpu, no autodiff needed for inference.
type Backend = burn::backend::NdArray<f32, i32>;

const MODEL_PATH: &str = "models/decoder"; // matches save_encoder(); NamedMpk appends .mpk
const ONSET_THRESHOLD: f32 = 0.5;
const FLASH_FRAMES: u32 = 6; // how many frames a hit marker stays lit
const BAR_WIDTH: usize = 24;

/// Optional human names. If the length doesn't match INSTRUMENT_COUNT we fall back to indices.
const INSTRUMENT_NAMES: &[&str] = &["Kick", "Snare", "Rim", "HatClosed", "HatOpen", "Crash", "Ride", "Tom"];

fn main() {
    let (device, supported) = input::select_input();

    let sample_rate = supported.sample_rate().0 as usize;
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.into();

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let worker = std::thread::spawn(move || run_worker(rx, sample_rate));

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build::<f32>(&device, &stream_config, channels, tx),
        cpal::SampleFormat::I16 => build::<i16>(&device, &stream_config, channels, tx),
        cpal::SampleFormat::U16 => build::<u16>(&device, &stream_config, channels, tx),
        other => panic!("unsupported sample format: {other:?}"),
    };

    stream.play().expect("failed to start stream");
    eprintln!("Listening — Ctrl-C to stop.\n");
    let _ = worker.join();
}

/// Build an input stream for sample type `T`, downmix to mono f32, forward to worker.
fn build<T>(device: &cpal::Device, config: &cpal::StreamConfig, channels: usize, tx: mpsc::Sender<Vec<f32>>) -> cpal::Stream
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let ch = channels.max(1);
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                // Downmix interleaved frames → mono. (Allocates per callback;
                // fine for a test tool. For zero-alloc, use an SPSC ring buffer.)
                let mut mono = Vec::with_capacity(data.len() / ch);
                for frame in data.chunks(ch) {
                    use cpal::Sample;
                    let sum: f32 = frame.iter().map(|s| f32::from_sample(*s)).sum();
                    mono.push(sum / ch as f32);
                }
                let _ = tx.send(mono); // drop if worker is behind — fine for a monitor
            },
            |e| eprintln!("audio stream error: {e}"),
            None,
        )
        .expect("failed to build input stream")
}

/// Worker thread: owns the GPU device, model, extractor, and the rolling window.
fn run_worker(rx: mpsc::Receiver<Vec<f32>>, input_sample_rate: usize) {
    let device = <Backend as burn::prelude::Backend>::Device::default();

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model::Model::<Backend>::init(&device)
        .load_file(MODEL_PATH, &recorder, &device)
        .expect("failed to load model — check MODEL_PATH");

    let mut mel = MelExtractor::new(input_sample_rate).expect("MelExtractor::new");

    let t = config::TIME_CHUNK_LENGTH;
    let m = config::MEL_FREQ_COUNT;
    let n = config::INSTRUMENT_COUNT;

    let mut ring: VecDeque<[f32; config::MEL_FREQ_COUNT]> = VecDeque::with_capacity(t);
    let mut display = Display::new(n);
    let mut announced_ready = false;

    while let Ok(samples) = rx.recv() {
        mel.push(&samples);

        // Drain every mel frame currently available. Run inference per new frame
        // once the window is full, reading the *newest* frame's onset logits.
        while let Some(frame) = mel.next() {
            if ring.len() == t {
                ring.pop_front();
            }
            ring.push_back(frame);

            if ring.len() < t {
                continue;
            }
            if !announced_ready {
                eprintln!("Buffer full ({t} frames) — running.\n");
                announced_ready = true;
            }

            // Flatten window → [t * m], frame-major (matches ChunkFeatures reshape
            // to [batch, chunk_length, mel_count]).
            let mut flat = Vec::with_capacity(t * m);
            for f in &ring {
                flat.extend_from_slice(f);
            }

            let features = model::ChunkFeatures::<Backend>::from_data(&device, flat, 1);
            let pred = model.forward(&features);

            // Newest time step's onset logits: [1, t, n] -> [n]
            let last = pred.onsets.slice([0..1, (t - 1)..t, 0..n]);
            let probs: Vec<f32> = sigmoid(last).into_data().to_vec::<f32>().unwrap();

            display.render(&probs);
        }
    }
}

/// Terminal display state. Redraws an in-place block of `n` lines, one per
/// instrument: a hit marker, name, a probability bar, the value, and a running
/// hit count (rising-edge over the threshold).
struct Display {
    n: usize,
    flash: Vec<u32>,
    prev_on: Vec<bool>,
    counts: Vec<u64>,
    first: bool,
}

impl Display {
    fn new(n: usize) -> Self {
        Self {
            n,
            flash: vec![0; n],
            prev_on: vec![false; n],
            counts: vec![0; n],
            first: true,
        }
    }

    fn render(&mut self, probs: &[f32]) {
        let mut out = std::io::stdout().lock();

        if self.first {
            self.first = false;
        } else {
            let _ = write!(out, "\x1b[{}A", self.n); // cursor up n lines
        }

        for i in 0..self.n {
            let p = probs.get(i).copied().unwrap_or(0.0);
            let on = p >= ONSET_THRESHOLD;

            if on && !self.prev_on[i] {
                self.counts[i] += 1; // rising edge = one onset
                self.flash[i] = FLASH_FRAMES;
            }
            self.prev_on[i] = on;

            let lit = self.flash[i] > 0;
            if self.flash[i] > 0 {
                self.flash[i] -= 1;
            }

            let filled = ((p * BAR_WIDTH as f32).round() as usize).min(BAR_WIDTH);
            let bar = "█".repeat(filled) + &"·".repeat(BAR_WIDTH - filled);
            let marker = if lit { "\x1b[92m●\x1b[0m" } else { " " };
            let name = label(i);

            // \r col 0, \x1b[2K clear line
            let _ = write!(
                out,
                "\r\x1b[2K {marker} {name:<12} |{bar}| {p:4.2}  hits:{:<6}\n",
                self.counts[i]
            );
        }
        let _ = out.flush();
    }
}

fn label(i: usize) -> String {
    if INSTRUMENT_NAMES.len() == config::INSTRUMENT_COUNT {
        INSTRUMENT_NAMES[i].to_string()
    } else {
        format!("inst {i:02}")
    }
}

fn list_devices(host: &cpal::Host) {
    println!("Input devices:");
    if let Ok(devs) = host.input_devices() {
        for d in devs {
            println!("  {}", d.name().unwrap_or_default());
        }
    }
    println!("\nOutput devices (loopback targets / --loopback):");
    if let Ok(devs) = host.output_devices() {
        for d in devs {
            println!("  {}", d.name().unwrap_or_default());
        }
    }
}
