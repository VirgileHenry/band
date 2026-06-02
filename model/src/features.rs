#[derive(Debug, Clone)]
pub struct ChunkFeatures<B: burn::prelude::Backend> {
    /// Features for an entire batch of tracks.
    ///
    /// The shape is `[batch_size, chunk_length, mel_freq_count]`
    pub features: burn::Tensor<B, 3>,
}

impl<B: burn::prelude::Backend> ChunkFeatures<B> {
    pub fn from_data(device: &B::Device, data: Vec<f32>, batch_size: usize) -> Self {
        let chunk_length = config::TIME_CHUNK_LENGTH;
        let mel_freq_count = config::MEL_FREQ_COUNT;

        if data.len() != batch_size * chunk_length * mel_freq_count {
            log::error!(
                "Invalid array size: expected {} (batch_size * chunk_length * mel_count), found {}",
                batch_size * chunk_length * mel_freq_count,
                data.len()
            );
        }

        let features_data = burn::tensor::TensorData::new(data, [batch_size, chunk_length, mel_freq_count]);
        let features = burn::Tensor::<B, 3>::from_floats(features_data, device);
        Self { features }
    }
}
