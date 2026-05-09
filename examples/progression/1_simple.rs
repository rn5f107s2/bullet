use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    nn::optimiser::AdamWParams,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder},
};

use bullet_lib::value::loader::ViriBinpackLoader;
use viriformat::dataformat::Filter;

fn main() {
    // hyperparams to fiddle with
    let hl_size = 12 * 64 * 16;
    let initial_lr = 0.001;
    let final_lr = 0.001_f32.powf(5.0);
    let superbatches = 300;
    let wdl_proportion = 0.5;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(403),
            SavedFormat::id("l0b").round().quantise::<i16>(403),
            SavedFormat::id("l1w").round().quantise::<i16>(64),
            SavedFormat::id("l1b").round().quantise::<i16>(403 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            // weights
            let l0 = builder.new_affine("l0", 768, hl_size);
            let l1 = builder.new_affine("l1", 2 * hl_size, 1);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
        });

    let stricter_clipping =  AdamWParams { max_weight: 1.27, min_weight: -1.27, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l1w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l1b", stricter_clipping);

    let schedule = TrainingSchedule {
        net_id: "QA403_2".to_string(),
        eval_scale: 133.0,
        steps: TrainingSteps {
            batch_size: 16384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: 40,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let dataloader = ViriBinpackLoader::new(
        "/data/moly_oraclegcp_5ks_12khtempmix_fixed_4mntemp.vf",
        8192,
        8,
        Filter {
            min_ply: 23,
            min_pieces: 4,
            max_eval: 32000,
            filter_tactical: true,
            filter_check: true,
            filter_castling: false,
            max_eval_incorrectness: u32::MAX,
            random_fen_skipping: true,
            random_fen_skip_probability: 0.90,
            wdl_filtered: false,
            wdl_model_params_a: [0.0; 4],
            wdl_model_params_b: [0.0; 4],
            material_min: 1,
            material_max: 100000,
            mom_target: 58,
            wdl_heuristic_scale: 1.5,
        }
    );

    trainer.run(&schedule, &settings, &dataloader);

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
