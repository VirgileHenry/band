#[derive(Debug)]
#[derive(burn::prelude::Module)]
pub struct Loss<B: burn::prelude::Backend> {
    onset_bce: burn::nn::loss::BinaryCrossEntropyLoss<B>,
    active_bce: burn::nn::loss::BinaryCrossEntropyLoss<B>,
}

impl<B: burn::prelude::Backend> Loss<B> {
    pub fn init(device: &B::Device) -> Self {
        /* Since onset are rarely present, we need to weight them positively to not learn "zero everywhere" */
        let onset_weights = burn::Tensor::<B, 1>::from_floats([50.0; config::INSTRUMENT_COUNT], device);

        let onset_bce = burn::nn::loss::BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .with_weights(Some(onset_weights.into_data().to_vec().unwrap()))
            .init(device);

        let active_bce = burn::nn::loss::BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(device);

        Self { onset_bce, active_bce }
    }

    pub fn forward(
        &self,
        prediction: &crate::predictions::ChunkPrediction<B>,
        labels: &crate::labels::ChunkLabels<B>,
    ) -> burn::Tensor<B, 1> {
        let onsets_loss = self.onset_bce.forward(prediction.onsets.clone(), labels.onsets.clone());
        let actives_loss = self.active_bce.forward(prediction.actives.clone(), labels.actives.clone());

        onsets_loss + actives_loss * 0.5
    }
}
