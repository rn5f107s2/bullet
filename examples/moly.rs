/*
This is about as simple as you can get with a network, the arch is
    (768 -> HIDDEN_SIZE)x2 -> 1
and the training schedule is pretty sensible.
There's potentially a lot of elo available by adjusting the wdl
and lr schedulers, depending on your dataset.
*/
use bullet_lib::{
    inputs, outputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler, Loss
};

const HIDDEN_SIZE: usize = 256;
const SCALE: i32 = 133;
const QA: i32 = 255;
const QB: i32 = 64;

fn main() {
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[QA, QB])
        .input(inputs::Chess768)
        .output_buckets(outputs::Single)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "moly_20240715".to_string(),
        eval_scale: SCALE as f32,
        ft_regularisation: 0.0,
        batch_size: 16384,
        batches_per_superbatch: 23137,
        start_superbatch: 1,
        end_superbatch: 40,
        wdl_scheduler: WdlScheduler::Constant { value: 0.5 },
        lr_scheduler: LrScheduler::Step { start: 0.001, gamma: 0.1, step: 15 },
        loss_function: Loss::SigmoidMSE,
        save_rate: 5,
    };

    let settings =
        LocalSettings { threads: 4, data_file_paths: vec!["data/MolyCurrentShuffled.bullet"], output_directory: "checkpoints" };

    trainer.run(&schedule, &settings);

    for fen in [
        "8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 b - - 2 47", //https://www.chessgames.com/perl/chessgame?gid=1143956, Bh3!
        "1r4nk/1p1qb2p/3p1r2/p1pPp3/2P1Pp2/5P1P/PP1QNBRK/5R2 b - - 3 30", //https://www.chessgames.com/perl/chessgame?gid=1084375, Qxh3
        "4r3/1k3p1p/2pr4/2Bn4/PP6/3B1pP1/R3p2P/4R1K1 b - - 0 33", //Nf4
        "r4rk1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3RK2R b K - 1 16", //https://www.chessgames.com/perl/chessgame?gid=1008361 Be6
        "2r2rk1/1bpR1p2/1pq1pQp1/p3P2p/P1PR3P/5N2/2P2PPK/8 w - - 2 32", //https://www.chessgames.com/perl/chessgame?gid=1124533 Kg3
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", //startpos
        "r1b1k2r/1p2bpp1/1qn1p3/p1ppPn2/5P1p/1P1P1N1P/PBPQN1P1/1K1R1B1R b kq - 1 13",
        "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/2KR1B1R b - - 4 9", //d5
        "8/p4p2/5pkp/1pr5/2P1KP2/6P1/P1R4P/8 b - - 1 32", //Rxc4 0-1
        "1rqb1rk1/3b1ppp/3p4/1p1Np3/p3P3/P1PQ4/1P2BPPP/3R1RK1 w - - 6 22"
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 133.0 * eval);
    }
}
