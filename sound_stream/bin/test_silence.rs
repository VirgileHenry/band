use sound_stream::MelExtractor;

fn main() {
    // 3 seconds of silence at 44.1kHz.
    let samples: Vec<f32> = vec![0.0; 44100 * 3];

    let mut ex = MelExtractor::new(44100).unwrap();
    ex.push(&samples);
    let frames: Vec<[f32; config::MEL_FREQ_COUNT]> = ex.collect();

    if frames.is_empty() {
        println!("no frames");
        return;
    }

    let mid = &frames[frames.len() / 2];
    println!("middle frame mel values:");
    println!("  bin 0   (lowest):  {:>7.2} dB", mid[0]);
    println!("  bin 60  (mid):     {:>7.2} dB", mid[60]);
    println!("  bin 100 (high):    {:>7.2} dB", mid[100]);
    println!("  bin 127 (highest): {:>7.2} dB", mid[127]);
    println!();
    println!(
        "min: {:.2}  max: {:.2}  mean: {:.2}",
        mid.iter().cloned().fold(f32::INFINITY, f32::min),
        mid.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        mid.iter().sum::<f32>() / mid.len() as f32,
    );
}
