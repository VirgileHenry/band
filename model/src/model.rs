/// Model that predicts instrumtent kicks from spectograms.
#[derive(Debug)]
#[derive(burn::prelude::Module)]
pub struct Model<B: burn::prelude::Backend> {
    /// First conv layer.
    conv_layer_1: burn::nn::conv::Conv2d<B>,
    /// Second conv layer.
    conv_layer_2: burn::nn::conv::Conv2d<B>,
    /// Activation after convolution layers
    conv_activation: burn::nn::Gelu,

    /// Pool layer to take the max out of the conv outputs
    max_pool: burn::nn::pool::MaxPool2d,

    /// Linear layer for the onset head
    onset_head: burn::nn::Linear<B>,
    /// Linear layer for the active head
    active_head: burn::nn::Linear<B>,
}

impl<B: burn::prelude::Backend> Model<B> {
    pub fn init(device: &B::Device) -> Self {
        let padding = burn::nn::PaddingConfig2d::Same;
        let conv_layer_1_config = burn::nn::conv::Conv2dConfig::new([1, 4], [3, 3]).with_padding(padding.clone());
        let conv_layer_2_config = burn::nn::conv::Conv2dConfig::new([4, 8], [3, 3]).with_padding(padding.clone());

        let pool_config = burn::nn::pool::MaxPool2dConfig::new([1, config::MEL_FREQ_COUNT]);

        let onset_head_config = burn::nn::LinearConfig::new(8, config::INSTRUMENT_COUNT);
        let active_head_config = burn::nn::LinearConfig::new(8, config::INSTRUMENT_COUNT);

        Self {
            conv_layer_1: conv_layer_1_config.init(device),
            conv_layer_2: conv_layer_2_config.init(device),
            conv_activation: burn::nn::Gelu::new(),
            max_pool: pool_config.init(),
            onset_head: onset_head_config.init(device),
            active_head: active_head_config.init(device),
        }
    }

    pub fn forward(&self, chunk_features: &crate::features::ChunkFeatures<B>) -> crate::predictions::ChunkPrediction<B> {
        let input_tensor = chunk_features.features.clone();

        /* Normalize across time dimension */
        let mean = input_tensor.clone().mean_dim(1); // (B, 1, M)
        let var = input_tensor.clone().var(1); // (B, 1, M), unbiased ok
        let output = (input_tensor - mean) / (var.sqrt() + 1e-5);

        /* Epxand channel for convolution */
        let batch_size = output.dims()[0];
        let chunk_length = output.dims()[1];
        let mel_count = output.dims()[2];
        let output: burn::Tensor<B, 4> = output.reshape([batch_size, 1, chunk_length, mel_count]);

        /* First convolution */
        let output: burn::Tensor<B, 4> = self.conv_layer_1.forward(output);
        let output: burn::Tensor<B, 4> = self.conv_activation.forward(output);

        /* Second convolution */
        let output: burn::Tensor<B, 4> = self.conv_layer_2.forward(output);
        let output: burn::Tensor<B, 4> = self.conv_activation.forward(output);

        /* Frequency max pool */
        let output: burn::Tensor<B, 4> = self.max_pool.forward(output);
        let output: burn::Tensor<B, 3> = output.squeeze_dim(3);
        let output: burn::Tensor<B, 3> = output.swap_dims(1, 2);

        /* heads for final prediction */
        let onsets: burn::Tensor<B, 3> = self.onset_head.forward(output.clone());
        let actives: burn::Tensor<B, 3> = self.active_head.forward(output.clone());

        crate::predictions::ChunkPrediction { onsets, actives }
    }
}
