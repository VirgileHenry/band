use band_audio_input::Resampler;
use band_frame_extractor::FrameExtractor;
use band_frame_extractor::NextFrameResult;
use band_frame_extractor::SpectralFrame;

const SAMPLE_RATE: usize = 22050;
const WINDOW_SIZE: usize = 2048;
const HOP: usize = 256;
const N_BINS: usize = 1025;
const HISTORY: usize = 256;

fn main() -> eframe::Result {
    /* Initialize a global tracing subscriber based on the RUST_LOG env var */
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info"))
        .init();
    tracing::info!("Running binary band_tools/spectrogram");

    let (sender, receiver) = std::sync::mpsc::channel();
    spawn_audio_thread(sender);

    let app = SpectroApp {
        frames: std::collections::VecDeque::with_capacity(HISTORY + 4),
        rx: receiver,
        texture: None,
        source_name: String::from("live source"),
        row_ranges: build_row_ranges(),
        display_max: 1.0,
    };

    eframe::run_native(
        "band - live spectrogram",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 500.0]),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(app))),
    )
}

struct SpectroApp {
    frames: std::collections::VecDeque<SpectralFrame<N_BINS>>,
    rx: std::sync::mpsc::Receiver<SpectralFrame<N_BINS>>,
    texture: Option<egui::TextureHandle>,
    source_name: String,
    row_ranges: Vec<(usize, usize)>,
    display_max: f32,
}

impl eframe::App for SpectroApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        let mut dirty = false;
        while let Ok(frame) = self.rx.try_recv() {
            self.frames.push_back(frame);
            while self.frames.len() > HISTORY {
                self.frames.pop_front();
            }
            dirty = true;
        }

        if dirty || self.texture.is_none() {
            let image = self.render_spectrogram();
            match &mut self.texture {
                Some(tex) => tex.set(image, egui::TextureOptions::NEAREST),
                None => self.texture = Some(ctx.load_texture("spectrogram", image, egui::TextureOptions::NEAREST)),
            }
        }

        let image = self.render_spectrogram(); /* the real work */
        let tex = self
            .texture
            .get_or_insert_with(|| ctx.load_texture("spectrogram", image.clone(), Default::default()));
        tex.set(image, egui::TextureOptions::NEAREST);

        egui::TopBottomPanel::top("stats").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.source_name);
                ui.separator();
                ui.label(format!(
                    "{} Hz · {} bins · hop {} · {:.1} fps analysis",
                    SAMPLE_RATE,
                    N_BINS,
                    HOP,
                    SAMPLE_RATE as f32 / HOP as f32
                ));
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            let size = ui.available_size();
            ui.image((tex.id(), size)); /* stretch to fill */
        });

        /* Repaint at the analysis cadence rather than flat-out: */
        ctx.request_repaint_after(std::time::Duration::from_millis(12));
    }
}

/* Display geometry for the spectrogram image */
const DISPLAY_ROWS: usize = 256;
const FREQ_MIN: f32 = 30.0;
const FREQ_MAX: f32 = 11_000.0;
const DB_FLOOR: f32 = -60.0;

impl SpectroApp {
    fn render_spectrogram(&mut self) -> egui::ColorImage {
        let width = HISTORY;
        let height = DISPLAY_ROWS;
        let mut pixels = vec![egui::Color32::BLACK; width * height];
        let x_offset = width - self.frames.len().min(width);

        /* --- Adaptive normalization --------------------------------- */
        /* Loudest magnitude currently in view... */
        let mut view_max = 0.0f32;
        for frame in &self.frames {
            for &v in frame.bins().iter() {
                view_max = view_max.max(v);
            }
        }
        /* ...tracked with fast attack / slow release so the picture
        brightens instantly on loud input but dims gradually,
        instead of pumping frame to frame. */
        if view_max > self.display_max {
            self.display_max = view_max;
        } else {
            self.display_max *= 0.995; /* ~1s half-life at 60 fps */
        }
        let db_top = 10.0 * self.display_max.max(1e-6).log10(); /* dB of "full brightness" */

        /* --- Paint -------------------------------------------------- */
        for (i, frame) in self.frames.iter().enumerate() {
            let x = x_offset + i;
            let features = frame.bins();
            for (row, &(lo, hi)) in self.row_ranges.iter().enumerate() {
                let mag = features[lo..hi].iter().cloned().fold(0.0f32, f32::max);
                let db = 20.0 * mag.max(1e-10).log10();
                /* Window: (db_top + DB_FLOOR) .. db_top  ->  0..1 */
                let norm = ((db - db_top - DB_FLOOR) / -DB_FLOOR).clamp(0.0, 1.0);
                let y = height - 1 - row;
                pixels[y * width + x] = heat_color(norm);
            }
        }

        egui::ColorImage {
            size: [width, height],
            source_size: egui::Vec2::new(width as f32, height as f32),
            pixels,
        }
    }
}

/* Poor-man's magma: black -> deep purple -> red-orange -> yellow-white */
fn heat_color(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.33 {
        let u = t / 0.33;
        (u * 0.4 + 0.05, 0.02, u * 0.5 + 0.1) /* black -> purple */
    } else if t < 0.66 {
        let u = (t - 0.33) / 0.33;
        (0.45 + u * 0.5, 0.02 + u * 0.35, 0.6 - u * 0.5) /* purple -> orange */
    } else {
        let u = (t - 0.66) / 0.34;
        (0.95, 0.37 + u * 0.55, 0.1 + u * 0.8) /* orange -> white */
    };
    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

fn build_row_ranges() -> Vec<(usize, usize)> {
    let bin_hz = SAMPLE_RATE as f32 / WINDOW_SIZE as f32;
    let log_min = FREQ_MIN.ln();
    let log_max = FREQ_MAX.ln();
    (0..DISPLAY_ROWS)
        .map(|row| {
            let f_lo = (log_min + (log_max - log_min) * row as f32 / DISPLAY_ROWS as f32).exp();
            let f_hi = (log_min + (log_max - log_min) * (row + 1) as f32 / DISPLAY_ROWS as f32).exp();
            let lo = ((f_lo / bin_hz).floor() as usize).min(N_BINS - 1);
            let hi = ((f_hi / bin_hz).ceil() as usize).clamp(lo + 1, N_BINS);
            (lo, hi)
        })
        .collect()
}

/// For this quick visualizer, we keep the audio pulling outisde of the eframe loop.
/// This avoids stalling the audio pulling and discarding frames.
///
/// However, it prevents us from using the frame buffer.
fn spawn_audio_thread(sender: std::sync::mpsc::Sender<SpectralFrame<N_BINS>>) {
    std::thread::spawn(move || {
        let mut sources = band_audio_input::AvailableSource::list();
        let hard_picked_source = sources.swap_remove(1);
        let source = hard_picked_source.to_live_source().unwrap();

        let resampled = Resampler::<_, SAMPLE_RATE>::new(source).unwrap();
        let mut extractor = FrameExtractor::<_, SAMPLE_RATE, WINDOW_SIZE, HOP, N_BINS>::new(resampled).unwrap();
        loop {
            match extractor.next_frame() {
                Ok(NextFrameResult::Frame(f)) => {
                    if sender.send(f).is_err() {
                        break;
                    }
                }
                Ok(NextFrameResult::Unavailable) => std::thread::sleep(std::time::Duration::from_millis(3)),
                Ok(NextFrameResult::EndOfInput) => break,
                Err(e) => {
                    tracing::error!("audio thread: {e}");
                    break;
                }
            }
        }
    });
}
