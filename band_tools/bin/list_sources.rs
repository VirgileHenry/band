//! Binary to list available audio sources in the terminal.

fn main() {
    /* Initialize a global tracing subscriber based on the RUST_LOG env var */
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info"))
        .init();
    tracing::info!("Running binary band_tools/list_sources");

    let mut available_sources = band_audio_input::AvailableSource::list();
    display_sources(&available_sources);

    /* Start all sources, re displaying each time */
    for source_index in 0..available_sources.len() {
        if let Some(source) = available_sources.get_mut(source_index) {
            source.start_capture();
        }
        clear_screen(&available_sources);
        display_sources(&available_sources);
    }

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_clone = stop.clone();
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = std::io::stdin().read_line(&mut s); /* blocks until Enter */
        stop_clone.store(true, std::sync::atomic::Ordering::Relaxed);
    });

    /* Infinite loop to see sources activity */
    loop {
        /* wait, 60 fps is plenty */
        std::thread::sleep(std::time::Duration::from_millis(15));

        /* exit */
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }

        for source in available_sources.iter_mut() {
            source.refresh_live_capture();
        }

        clear_screen(&available_sources);
        display_sources(&available_sources);
    }
}

fn display_sources(sources: &[band_audio_input::AvailableSource]) {
    let name_pad = sources.iter().map(|s| s.description().name().len()).max().unwrap_or(0) + 1;

    println!();
    for source in sources.iter() {
        print!(" - {:<name_pad$} -  ", source.description().name(), name_pad = name_pad);
        match source.live_capture() {
            band_audio_input::AvailableSourceLiveCapture::NotStarted => print!("Not Started"),
            band_audio_input::AvailableSourceLiveCapture::Blacklisted => print!("Blacklisted"),
            band_audio_input::AvailableSourceLiveCapture::Errored { error } => print!("Errored: {error}"),
            band_audio_input::AvailableSourceLiveCapture::Live { activity, .. } => print_activity(*activity),
        }
        println!("\x1b[K"); /* Clear to end of line */
    }
    println!();
    println!("Press Enter to exit");
}

fn clear_screen(sources: &[band_audio_input::AvailableSource]) {
    /* Move up to clear previous */
    print!("\x1b[{lines}A", lines = sources.len() + 3);
}

fn print_activity(activity: f32) {
    const GRADIENT: [u8; 12] = [21, 27, 33, 39, 45, 49, 46, 118, 226, 214, 202, 196];
    const BAR_WIDTH: usize = 40;
    let activity = activity.clamp(0.0, 1.0);
    let activity = level_to_bar(activity);
    let filled = (activity * BAR_WIDTH as f32).round() as usize;

    print!("[");
    for i in 0..BAR_WIDTH {
        if i < filled {
            /* Get color code from gradient */
            let color_index = i * GRADIENT.len() / BAR_WIDTH;
            print!("\x1b[38;5;{}m/", GRADIENT[color_index]);
        } else {
            print!(" ");
        }
    }
    /* Reset the color */
    print!("\x1b[0m]");
}

/// Map a linear level (0..=1) to a bar fraction via dBFS.
/// Window: -60 dB (near silence) .. 0 dB (full scale).
fn level_to_bar(linear: f32) -> f32 {
    let db = 20.0 * linear.log10(); /* 0.004 -> ~ -48 dB */
    ((db + 60.0) / 60.0).clamp(0.0, 1.0) /* -60..0 dB -> 0..1  */
}
