use candle_core::{Error, IndexOp, Result, Tensor};
use candle_nn::{embedding, linear, ops, Embedding, Linear, Module, VarBuilder};
use rand::{distributions::Distribution, thread_rng};
use serde::Deserialize;

use super::Model;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Number of n-grams
    pub vocab_size: usize,
    pub hidden_dim: usize,
}

#[derive(Debug, Clone)]
pub struct Transformer {
    wte: Embedding,
    lm_head: Linear,
    rng: rand::rngs::ThreadRng,
}

impl Transformer {
    pub fn new(vs: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            wte: embedding(cfg.vocab_size, cfg.hidden_dim, vs.pp("wte"))?,
            lm_head: linear(cfg.hidden_dim, cfg.vocab_size, vs.pp("lm_head"))?,
            rng: thread_rng(),
        })
    }
}

impl Module for Transformer {
    /// Returns logprobs
    fn forward(&self, xs: &candle_core::Tensor) -> Result<Tensor> {
        // Remove this eventually. Here for reference
        // println!("Input tensor dims: {:?}", tensor.shape());
        // println!("Input values: {:?}", tensor.to_string());
        // println!(
        //     "Embeddings: {:?}",
        //     self.token_embedding_table.embeddings().to_string()
        // );
        //let logits = self.token_embedding_table.forward(&xs.flatten_all()?)?;
        let tok_emb = self.wte.forward(&xs)?;
        let logits = self.lm_head.forward(&tok_emb)?;
        Ok(logits)
    }
}

impl Model for Transformer {
    fn from_config(
        vs: VarBuilder,
        cfg: &crate::config::pretrained_config::PretrainedConfig,
    ) -> Result<Self> {
        Ok(Self {
            wte: embedding(
                cfg.vocab_size as usize,
                cfg.hidden_size as usize,
                vs.pp("wte"),
            )?,
            lm_head: linear(
                cfg.hidden_size as usize,
                cfg.vocab_size as usize,
                vs.pp("lm_head"),
            )?,
            rng: thread_rng(),
        })
    }

    fn generate(&mut self, idx: &Tensor, max_new_tokens: usize) -> Result<Tensor> {
        let mut preds = idx.clone();
        for _ in 0..max_new_tokens {
            let logits = self.forward(&preds)?;
            // Get logprobs for last time step
            let logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            //let logprobs = ops::softmax_last_dim(&logits)?;
            let logprobs = ops::softmax(&logits, 1)?;

            // Forced to do this because Candle doesn't have `torch.multinomial` built in.
            // May need to PR this in myself eventually.
            let prs: Vec<Vec<f32>> = logprobs.to_dtype(candle_core::DType::F32)?.to_vec2()?;
            let next_tokens: Result<Vec<u32>> = prs
                .iter()
                .map(|p| {
                    let distr = rand::distributions::WeightedIndex::new(p).map_err(Error::wrap)?;
                    let next_token = distr.sample(&mut self.rng) as u32;
                    Ok(next_token)
                })
                .collect();
            let next_tokens = next_tokens?;
            let b = &next_tokens.len();
            let next_tokens_tensor = Tensor::new(next_tokens, idx.device())?.reshape(&[*b, 1])?;
            preds = Tensor::cat(&[preds, next_tokens_tensor], 1)?;
        }
        // TODO: Delete!
        Ok(preds)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    use super::Model;
    use super::{Config, Transformer};

    #[test]
    fn test_generate() {
        let device = Device::Cpu;
        let start_idx = Tensor::zeros((2, 1), DType::U32, &device).unwrap();

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F16, &device);
        let mut model = Transformer::new(
            vs,
            &Config {
                vocab_size: 4,
                hidden_dim: 32,
            },
        )
        .unwrap();

        // Generate complete gibberish
        let preds = model.generate(&start_idx, 100).unwrap();
        println!("Tokens: {:?}", preds.to_vec2::<u32>().unwrap())
    }
}
