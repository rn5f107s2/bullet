/*
This is the result of some experimentation I did to try and train a network
for akimbo with more layers, it ended up being significantly stronger at
fixed-nodes, but unfortunately was too much of a slowdown to pass any
time-controlled test.
*/
use bullet_lib::{
    inputs, loader, lr, optimiser, outputs, wdl, Activation, LocalSettings, Loss, TrainerBuilder, TrainingSchedule,
};

fn main() {
    let mut trainer = TrainerBuilder::default()
        .optimiser(optimiser::AdamW)
        .input(inputs::Chess768)
        .output_buckets(outputs::Single)
        .feature_transformer(256)
        .activate(Activation::SCReLU)
        .add_layer(2)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 133.0,
        ft_regularisation: 0.0,
        batch_size: 16_384,
        batches_per_superbatch: 6104,
        start_superbatch: 1,
        end_superbatch: 200,
        wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 57 },
        loss_function: Loss::SigmoidMSE,
        save_rate: 20,
        optimiser_settings: optimiser::AdamWParams {
            decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            min_weight: -1.98,
            max_weight: 1.98,
        },
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/MolyBig.bullet"]);

    trainer.run(&schedule, &settings, &data_loader);
}
