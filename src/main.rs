use candle_core::{Device, Error, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use nanogpt::config::pretrained_config::PretrainedConfig;
use nanogpt::models::bigram::{self, Bigram};
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

    #[arg(short, long)]
    n_tokens: Option<usize>,
}

fn generate(
    tokenizer: &Tokenizer,
    model: &mut Bigram,
    prompt: &str,
    device: &Device,
    max_tokens: usize,
) -> Result<String> {
    let input_encoding = tokenizer
        .encode(&prompt)
        .map_err(|_| Error::Msg("Tokenizer error".into()))?;
    let idx_1d = Tensor::new(input_encoding.ids, &device)?;
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

    // Load config. Assume models are in ./models for now
    let cwd = env::current_dir().unwrap();
    let config_path: PathBuf = cwd.join(format!("models/{}/config.json", args.model_id));
    let config = PretrainedConfig::from_json_file(&config_path).unwrap();

    // Assume tokenizers are at top level for now
    let tokenizer_path: PathBuf =
        cwd.join(format!("models/{}-tokenizer.json", config.tokenizer_id));
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
    println!("Loaded tokenizer; vocab: {:?}", tokenizer.get_vocab_size());

    let device = nanogpt::util::get_device();

    // TODO: Factor this out once we make this multi-model
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let mut model = bigram::Bigram::new(
        vs,
        &bigram::Config {
            vocab_size: tokenizer.get_vocab_size(),
        },
    )
    .unwrap();

    // Get weights if exist, else bail
    let weight_path = cwd.join(format!(
        "models/{}/{}.safetensors",
        args.model_id, args.model_id
    ));
    if weight_path.exists() {
        println!("Loading {} model", args.model_id);
        varmap.load(weight_path).unwrap();
    } else {
        println!("Fail!");
    }

    let prompt = args.prompt.unwrap_or_else(|| " ".to_string());
    let max_tokens = args.n_tokens.unwrap_or_else(|| 100);
    println!(
        "{:?}",
        generate(&tokenizer, &mut model, &prompt, &device, max_tokens)
    )
}
