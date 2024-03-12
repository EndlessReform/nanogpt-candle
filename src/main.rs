use candle_core::{Device, Error, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use nanogpt::config::pretrained_config::PretrainedConfig;
use nanogpt::models::{Model, ModelWrapper, WhichModel};
use nanogpt::tokenizer::Tokenizer;
use std::env;
use std::path::PathBuf;
use std::process;

#[derive(Parser, Debug)]
#[command(version, long_about=None)]
struct Args {
    /// Name of folder in model directory.
    #[arg(short, long, default_value = "transformer")]
    model_type: WhichModel,

    #[arg(short, long)]
    prompt: Option<String>,

    #[arg(short, long)]
    n_tokens: Option<usize>,
}

fn generate<M: Model>(
    tokenizer: &Tokenizer,
    model: &mut M,
    prompt: &str,
    device: &Device,
    max_tokens: usize,
) -> Result<String> {
    let input_encoding = tokenizer
        .encode(prompt)
        .map_err(|_| Error::Msg("Tokenizer error".into()))?;
    let idx_1d = Tensor::new(input_encoding.ids, device)?;
    let idx = idx_1d.reshape((1, idx_1d.dims1()?))?;
    println!("Tokenized");
    let data = model.generate(&idx, max_tokens)?;
    println!("Generated");
    let (b, c) = data.dims2()?;
    let data_1d = data.reshape(b * c)?;

    Ok(tokenizer
        .decode(&data_1d.to_vec1()?)
        .map_err(|_| Error::Msg("Could not decode".into()))?)
}

fn main() {
    let args = Args::parse();

    let cwd = env::current_dir().unwrap();
    // Hardcode to bigram for now
    let model_name: String = args.model_type.into();
    let config_path: PathBuf = cwd.join(format!("models/{}/config.json", model_name));
    if !config_path.exists() {
        eprintln!(
            "Error: config file not found at expected path: {:?}",
            config_path
        );
        process::exit(1);
    }
    let config: PretrainedConfig = match PretrainedConfig::from_json_file(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Failed to load config from {:?}: {}", config_path, e);
            process::exit(1);
        }
    };

    let tokenizer_path: PathBuf =
        cwd.join(format!("models/{}-tokenizer.json", config.tokenizer_id));

    if !tokenizer_path.exists() {
        eprintln!(
            "Error: Tokenizer not found at expected path: {:?}",
            tokenizer_path
        );
        process::exit(1);
    }
    let tokenizer: Tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(e) => {
            eprintln!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e);
            process::exit(1);
        }
    };

    let device = nanogpt::util::get_device();

    // TODO: Factor this out once we make this multi-model
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let mut model: ModelWrapper = ModelWrapper::from_config(vs, &config).unwrap();

    // Get weights if exist, else bail
    let weight_path = cwd.join(format!("models/{}/model.safetensors", model_name));
    if weight_path.exists() {
        println!("Loading {} model", model_name);
        varmap.load(weight_path).unwrap();
    } else {
        println!("Fail!");
    }

    let prompt = args.prompt.unwrap_or(" ".to_string());
    let max_tokens = args.n_tokens.unwrap_or(20);
    println!(
        "{:?}",
        generate(&tokenizer, &mut model, &prompt, &device, max_tokens)
    )
}
