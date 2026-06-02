mod audio;
mod chunk;
mod data;
mod midi;

/// Preprocess the dataset.
#[derive(argh::FromArgs)]
struct PreprocessArgs {
    /// raw dataset directory.
    #[argh(option, short = 'i')]
    dataset_dir: Option<String>,

    /// target directory to write the processed data.
    #[argh(option, short = 'o')]
    output_dir: Option<String>,
}

const DEFAULT_DATASET_DIR: &str = "dataset/e-gmd-v1.0.0";
const DEFAULT_OUTPUT_DIR: &str = "dataset/processed";

fn main() -> std::io::Result<()> {
    let default_level = "info";
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_level)).init();

    let args: PreprocessArgs = argh::from_env();
    let dataset_dir = match args.dataset_dir.as_ref() {
        Some(dir) => dir.as_str(),
        None => DEFAULT_DATASET_DIR,
    };
    let output_dir = match args.output_dir.as_ref() {
        Some(dir) => dir.as_str(),
        None => DEFAULT_OUTPUT_DIR,
    };

    log::info!("Preprocessing data: {dataset_dir} -> {output_dir}");
    let dataset_path = format!("{dataset_dir}/e-gmd-v1.0.0.csv");
    let mut dataset = csv::Reader::from_path(&dataset_path)?;

    let mut successes: usize = 0;
    let mut failures: usize = 0;

    for record in dataset.deserialize() {
        let record: data::EgmdRow = record?;
        log::info!("Processing {record}");

        let filename = match record.audio_filename.strip_suffix(".wav") {
            Some(filename) => filename,
            None => {
                log::warn!("Audio filename does not end with .wav: {}", record.audio_filename);
                failures += 1;
                continue;
            }
        };
        let wav_path = std::path::PathBuf::from(&format!("{dataset_dir}/{}", record.audio_filename));
        let midi_path = std::path::PathBuf::from(&format!("{dataset_dir}/{}", record.midi_filename));
        let output_path = std::path::PathBuf::from(&format!("{output_dir}/{filename}.bin"));
        std::fs::create_dir_all(output_path.parent().unwrap())?;

        if let Err(e) = process_one(&wav_path, &midi_path, &output_path) {
            log::warn!("Failed to process {filename}: {e}");
            failures += 1;
            continue;
        }

        successes += 1;
    }

    log::info!(
        "Finished: processed {} records, {} succeeded, {} failed.",
        successes + failures,
        successes,
        failures
    );

    Ok(())
}

fn process_one(wav: &std::path::Path, midi: &std::path::Path, output_path: &std::path::Path) -> std::io::Result<()> {
    let decoded = audio::decode_to_mono(wav)?;
    log::debug!("{wav:?}: {} samples @ {} Hz", decoded.samples.len(), decoded.sample_rate);

    let mut ex = sound_stream::MelExtractor::new(decoded.sample_rate as usize)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
    ex.push(&decoded.samples);
    let mels: Vec<_> = ex.collect();

    let hits = midi::parse(midi)?;
    let (onsets, actives) = chunk::build_labels(&hits, mels.len());

    let n = chunk::write(output_path, &mels, &onsets, &actives)?;
    log::debug!("  wrote {n} frames -> {output_path:?}");
    Ok(())
}
