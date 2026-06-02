#[derive(Debug, Clone)]
pub struct ChunkPrediction<B: burn::prelude::Backend> {
    /// Onset labels for an entire batch of tracks.
    ///
    /// The shape is `[batch_size, chunk_length, instrument_count]`
    pub onsets: burn::Tensor<B, 3>,
    /// Active labels for an entire batch of tracks.
    ///
    /// The shape is `[batch_size, chunk_length, instrument_count]`
    pub actives: burn::Tensor<B, 3>,
}
