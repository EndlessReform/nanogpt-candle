use candle_core::{Device, Module, Result};
use candle_datasets::Batcher;
use candle_nn::{loss, VarBuilder, VarMap};
use nanogpt::config::Config;
use nanogpt::dataloader::TextDatasetIterator;
use nanogpt::datasets::TextDataset;
use nanogpt::models::bigram;
use nanogpt::tokenizer::Tokenizer;
use std::env;
use std::path::PathBuf;

struct TrainingArgs {
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    /// Safetensors filename to load weights from. Will be passed through to hf_hub
    load_from: Option<String>,
    /// Safetensors filename to save weights to.
    save_to: Option<String>,
}

fn training_loop(
    train_iter: TextDatasetIterator,
    args: &TrainingArgs,
    device: &Device,
) -> Result<()> {
    // TERRIBLE do not do this
    let vocab_size = 63;
    let mut train_batcher = Batcher::new_r2(train_iter).batch_size(args.batch_size);

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F16, device);
    let model = bigram::Bigram::new(vs, &bigram::Config { vocab_size })?;

    if let Some(load_from) = &args.load_from {
        varmap.load(load_from)?;
    }

    for _epoch in 0..args.epochs {
        if let Some(Ok((xs, ys))) = train_batcher.next() {
            let logprobs = model.forward(&xs)?;
            println!(
                "Forward pass complete. Logprobs shape: {:?}",
                logprobs.shape()
            );
            let loss = loss::nll(&logprobs, &ys.flatten_all()?)?;
            println!("Loss: {:?}", loss);
        }
    }
    // Sample one
    Ok(())
}

fn main() {
    // Load config
    let cwd = env::current_dir().unwrap();
    let config_path: PathBuf = [cwd.clone(), "config".into(), "model-config.json".into()]
        .iter()
        .collect();
    let config = Config::from_json_file(&config_path).unwrap();

    let tokenizer_path: PathBuf = [
        cwd.clone(),
        "models".into(),
        format!("{}-tokenizer.json", config.tokenizer_id).into(),
    ]
    .iter()
    .collect();
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
    println!("Vocab: {:?}", tokenizer.get_vocab_size());

    let dataset_path: PathBuf = [cwd.clone(), "corpus".into(), "shakespeare.txt".into()]
        .iter()
        .collect();
    let base_dataset = TextDataset::new(&[dataset_path], |s| tokenizer.encode(s)).unwrap();
    let (train_dataset, _test_dataset) = base_dataset.train_test_split(0.2).unwrap();

    let device: Device;
    #[cfg(feature = "cuda")]
    {
        device = Device::new_cuda(0).unwrap();
    }

    #[cfg(feature = "metal")]
    {
        device = Device::new_metal(0).unwrap();
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        device = Device::Cpu;
    }
    let train_iter =
        TextDatasetIterator::new(&train_dataset, config.context_size as usize, &device);

    if let Ok(train_iter) = train_iter {
        // TODO: stop hard-coding this
        training_loop(
            train_iter,
            &TrainingArgs {
                learning_rate: 0.0,
                epochs: 1,
                load_from: None,
                save_to: None,
                batch_size: 1,
            },
            &device,
        )
        .unwrap();
    }
}
