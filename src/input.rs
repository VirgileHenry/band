use cpal::traits::{DeviceTrait, HostTrait};

/// Lists capture devices, lets the user pick one by number, and returns the
/// chosen device + its default input config. Look for a name ending in
/// `.monitor` — that's the playback monitor carrying Spotify's audio.
pub fn select_input() -> (cpal::Device, cpal::SupportedStreamConfig) {
    let host = cpal::default_host();

    let devices: Vec<cpal::Device> = host
        .input_devices()
        .expect("failed to enumerate input devices")
        .collect();

    if devices.is_empty() {
        panic!("no input devices found — is PipeWire/Pulse running?");
    }

    println!("\nAvailable capture devices:");
    for (i, d) in devices.iter().enumerate() {
        let name = d.name().unwrap_or_else(|_| "<unknown>".into());
        println!("  [{i}] {name}");
    }

    // prompt loop
    let device = loop {
        println!("Select device number: ");

        let mut line = String::new();
        std::io::stdin()
            .read_line(&mut line)
            .expect("failed to read stdin");

        match line.trim().parse::<usize>() {
            Ok(index) => match devices.get(index) {
                Some(device) => break device.clone(),
                None => println!(" Invalid provided number"),
            },
            _ => println!(" Expected a number !"),
        }
    };

    let name = device.name().unwrap_or_else(|_| "<unknown>".into());

    let config = device
        .default_input_config()
        .expect("device has no default input config");

    println!(
        "\nSelected: {name}\n  {} Hz, {} ch, {:?}",
        config.sample_rate().0,
        config.channels(),
        config.sample_format()
    );

    (device, config)
}
