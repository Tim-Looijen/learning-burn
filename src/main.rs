use burn::{
    backend::{Autodiff, Cuda},
    optim::AdamConfig,
};
use learning_burn::{
    model::ModelConfig,
    training::{TrainingConfig, train},
};

fn main() {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::cuda::CudaDevice::default();
    let artifact_dir = "log/";
    train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
