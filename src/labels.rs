#[derive(Debug, Clone)]
pub struct ChunkLabels<B: burn::prelude::Backend> {
    /// Onset labels for an entire batch of tracks.
    ///
    /// The shape is `[batch_size, chunk_length, instrument_count]`
    pub onsets: burn::Tensor<B, 3, burn::tensor::Int>,
    /// Active labels for an entire batch of tracks.
    ///
    /// The shape is `[batch_size, chunk_length, instrument_count]`
    pub actives: burn::Tensor<B, 3, burn::tensor::Int>,
}

impl<B: burn::prelude::Backend> ChunkLabels<B> {
    pub fn from_data(device: &B::Device, onsets: Vec<f32>, actives: Vec<f32>, batch_size: usize) -> Self {
        let chunk_length = crate::config::TIME_CHUNK_LENGTH;
        let instrument_count = crate::config::INSTRUMENT_COUNT;

        if onsets.len() != batch_size * chunk_length * instrument_count {
            log::error!(
                "Invalid array size: expected {} (batch_size * chunk_length * mel_count), found {}",
                batch_size * chunk_length * instrument_count,
                onsets.len()
            );
        }
        if actives.len() != batch_size * chunk_length * instrument_count {
            log::error!(
                "Invalid array size: expected {} (batch_size * chunk_length * mel_count), found {}",
                batch_size * chunk_length * instrument_count,
                onsets.len()
            );
        }

        let onsets_data = burn::tensor::TensorData::new(onsets, [batch_size, chunk_length, instrument_count]);
        let onsets = burn::Tensor::<B, 3, _>::from_ints(onsets_data, device);
        let actives_data = burn::tensor::TensorData::new(actives, [batch_size, chunk_length, instrument_count]);
        let actives = burn::Tensor::<B, 3, _>::from_ints(actives_data, device);
        Self { onsets, actives }
    }
}
