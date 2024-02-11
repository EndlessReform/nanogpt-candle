use clap::Parser;
use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use nanogpt::tokenizer::{
    models::{character::Character, Model},
    trainer::Trainer,
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
    let mut model = Character::new(HashMap::new());
    let mut trainer = model.get_trainer();

    // Load contents
    let cwd = env::current_dir().unwrap();
    let input_file_path: PathBuf = [&cwd, &args.infile].iter().collect();
    let file = File::open(&input_file_path).unwrap();
    let reader = BufReader::new(file);

    // Do training
    let fake_processor = |s: &str| Ok(vec![s.to_string()]);
    trainer
        .feed(reader.lines().map(|l| l.unwrap()), fake_processor)
        .unwrap();
    trainer
        .feed(vec!["\n".to_string()].iter(), fake_processor)
        .unwrap();
    trainer.train(&mut model).unwrap();

    // Persist
    let out_dir: PathBuf = [&cwd, &args.outdir].iter().collect();
    let out_paths = model.save(&out_dir, None).unwrap();

    println!(
        "Tokenizer saved to {:?}, vocab size: {}",
        out_paths,
        model.get_vocab_size()
    );
}
