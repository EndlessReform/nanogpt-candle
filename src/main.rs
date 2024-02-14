use candle_core::{Device, Tensor};
use nanogpt::config::Config;
use nanogpt::tokenizer::Tokenizer;
use std::env;
use std::path::PathBuf;

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

    // Example encode
    let device = Device::Cpu;
    let tokens = tokenizer.encode("Hiii!").unwrap();
    let tensor = Tensor::new(tokens.ids, &device).unwrap();
    println!("Tensor: {:?}", tensor);
}
