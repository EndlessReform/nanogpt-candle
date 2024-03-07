use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use nanogpt::config::pretrained_config::PretrainedConfig;
use nanogpt::models::bigram;
use nanogpt::tokenizer::Tokenizer;
use std::env;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, long_about=None)]
struct Args {
    /// Name of folder in model directory.
    #[clap(short, long = "model-id", default_value = "bigram")]
    model_id: String,

    #[arg(short, long)]
    prompt: Option<String>,
}

fn main() {
    let args = Args::parse();

    // Load config. Assume models are in ./models for now
    let cwd = env::current_dir().unwrap();
    let config_path: PathBuf = cwd.join(format!("models/{}/config.json", args.model_id));
    let config = PretrainedConfig::from_json_file(&config_path).unwrap();
    println!("Loading {} model", args.model_id);

    // Assume tokenizers are at top level for now
    let tokenizer_path: PathBuf =
        cwd.join(format!("models/{}-tokenizer.json", config.tokenizer_id));
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
    println!("Loaded tokenizer; vocab: {:?}", tokenizer.get_vocab_size());

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
}
