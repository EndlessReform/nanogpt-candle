pub mod train;

use clap::Parser;
use std::{collections::HashMap, env, path::PathBuf, vec};

use nanogpt::tokenizer::{
    models::{character::Character, ModelWrapper},
    Tokenizer,
};

#[derive(Parser, Debug)]
#[command(version, long_about=None)]
struct Args {
    /// Path of file to train tokenizer
    #[arg(short, long)]
    infile: PathBuf,

    /// Path to persist trained tokenizer to
    #[arg(short, long)]
    outdir: PathBuf,
}

fn main() {
    let args = Args::parse();

    // Init model and trainer
    let model = Character::new(HashMap::new());
    let mut tokenizer = Tokenizer::new(ModelWrapper::Character(model));

    // Load contents
    let cwd = env::current_dir().unwrap();
    let input_file_path: PathBuf = [&cwd, &args.infile].iter().collect();

    tokenizer.train_from_files(vec![input_file_path]).unwrap();

    // Persist
    let out_dir: PathBuf = [&cwd, &args.outdir].iter().collect();
    let out_paths = tokenizer
        .save(&out_dir, Some("shakespeare-tokenizer".into()))
        .unwrap();

    println!(
        "Tokenizer saved to {:?}, vocab size: {}",
        out_paths,
        tokenizer.get_vocab_size()
    );
}
