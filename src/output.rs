/// I have little clue of what I'm doing, so I'm doing like the regression output for now
#[derive(Debug, Clone)]
pub struct Output<B: burn::prelude::Backend> {
    pub loss: burn::Tensor<B, 1>,
}

impl<B: burn::prelude::Backend> burn::train::ItemLazy for Output<B> {
    type ItemSync = Output<burn::backend::NdArray>;

    fn sync(self) -> Self::ItemSync {
        let transaction = burn::tensor::Transaction::default().register(self.loss);

        let tensors = transaction.execute();
        let [loss] = tensors.try_into().expect("Correct amount of tensor data");

        let device = Default::default();
        let loss = burn::tensor::Tensor::from_data(loss, &device);
        Output { loss }
    }
}

impl<B: burn::prelude::Backend> burn::train::metric::Adaptor<burn::train::metric::LossInput<B>> for Output<B> {
    fn adapt(&self) -> burn::train::metric::LossInput<B> {
        burn::train::metric::LossInput::new(self.loss.clone())
    }
}
