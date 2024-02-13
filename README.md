## NanoGPT, in Candle, spelled out

<img src="./docs/candle.png" width="400px" />

This repo is an implementation of [nanoGPT](https://github.com/karpathy/nanoGPT) in [Candle](https://github.com/huggingface/candle), a Rust-based PyTorch alternative.

If you use this in production, well, that's on you buddy.

## Installation

Install Rust if you haven't already done so. Then in root:

```bash
cargo build
# If on Nvidia platform
cargo build --features cuda
# If on Apple Silicon
cargo build --features metal
```

Add TinyShakespeare dataset:

```bash
mkdir corpus
curl 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt' -o corpus/shakespeare.txt
```

## Usage

Train TinyShakespeare tokenizer:

```bash
cargo run --bin train_tokenizer -- -i corpus/shakespeare.txt -o models
```
