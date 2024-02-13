use candle_core::{Device, Tensor};
use nanogpt::tokenizer::Tokenizer;
use std::env;

fn main() {
    let mut path = env::current_dir().unwrap();
    path.push("models");
    path.push("vocab.json");
    let tokenizer = Tokenizer::from_file(&path).unwrap();
    println!("Vocab: {:?}", tokenizer.get_vocab_size());

    // Example encode
    let device = Device::Cpu;
    let tokens = tokenizer.encode("Hiii!").unwrap();
    let tensor = Tensor::new(tokens.ids, &device).unwrap();
    println!("Tensor: {:?}", tensor);
}
