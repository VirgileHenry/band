#[derive(burn::prelude::Module, Debug)]
pub struct TrainingModel<B: burn::prelude::Backend> {
    pub model: crate::model::Model<B>,
    pub loss: crate::loss::Loss<B>,
}

impl<B: burn::prelude::Backend> TrainingModel<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            model: crate::model::Model::init(device),
            loss: crate::loss::Loss::init(device),
        }
    }

    fn forward_step(&self, batch: crate::data::ChunkBatch<B>) -> crate::output::Output<B> {
        let pred = self.model.forward(&batch.features);
        let loss = self.loss.forward(&pred, &batch.labels);

        crate::output::Output { loss }
    }

    pub fn save_encoder(self, path: &str) {
        use burn::prelude::Module;

        let recorder = burn::record::NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::new();
        self.model
            .save_file(path, &recorder)
            .expect("Should be able to save the model");
    }
}

impl<B: burn::tensor::backend::AutodiffBackend> burn::train::TrainStep for TrainingModel<B> {
    type Input = crate::data::ChunkBatch<B>;
    type Output = crate::output::Output<B>;
    fn step(&self, batch: crate::data::ChunkBatch<B>) -> burn::train::TrainOutput<Self::Output> {
        let item = self.forward_step(batch);
        let grads = item.loss.backward();

        burn::train::TrainOutput::new(&self.model, grads, item)
    }
}

impl<B: burn::prelude::Backend> burn::train::InferenceStep for TrainingModel<B> {
    type Input = crate::data::ChunkBatch<B>;
    type Output = crate::output::Output<B>;
    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_step(batch)
    }
}
