type WgpuBackend = burn::backend::Wgpu<f32, i32>;
type AutodiffBackend = burn::backend::Autodiff<WgpuBackend>;

fn main() -> std::io::Result<()> {
    /* Info in debug, warning in release. Can be overriden by the env. */
    let default_level = if cfg!(debug_assertions) { "info" } else { "warn" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_level)).init();

    let artifacts_dir = "./training_artifacts";

    let training_config = TrainingConfig::new(burn::optim::AdamConfig::new());

    let device = burn::backend::wgpu::WgpuDevice::default();
    let model = band::TrainingModel::<AutodiffBackend>::init(&device);

    create_artifact_dir(artifacts_dir);
    train::<AutodiffBackend>(artifacts_dir, training_config, device, model)?;

    Ok(())
}

#[derive(burn::prelude::Config, Debug)]
pub struct TrainingConfig {
    pub optimizer: burn::optim::AdamConfig,
    #[config(default = 5)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: burn::tensor::backend::AutodiffBackend>(
    artifact_dir: &str,
    training_config: TrainingConfig,
    device: B::Device,
    model: band::TrainingModel<B>,
) -> std::io::Result<()> {
    B::seed(&device, training_config.seed);

    let dataset_root = "dataset/processed";
    let training_dataset = band::EgmdDataset::load_train(dataset_root)?;
    let validation_dataset = band::EgmdDataset::load_valid(dataset_root)?;

    let dataloader_train = burn::data::dataloader::DataLoaderBuilder::new(band::ChunkBatcher)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(training_dataset);

    let dataloader_valid = burn::data::dataloader::DataLoaderBuilder::new(band::ChunkBatcher)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(validation_dataset);

    let training = burn::train::SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metrics((burn::train::metric::LossMetric::new(),))
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .num_epochs(training_config.num_epochs)
        .summary();

    let learner = burn::train::Learner::new(model, training_config.optimizer.init(), training_config.learning_rate);
    let result = training.launch(learner);

    result.model.save_encoder("models/decoder");

    Ok(())
}
