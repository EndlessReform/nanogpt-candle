use self::{bigram::Bigram, transformer::Transformer};

use super::config::pretrained_config::PretrainedConfig;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use clap::ValueEnum;

pub mod bigram;
pub mod transformer;

pub trait Model: Sized {
    fn generate(&mut self, idx: &Tensor, max_new_tokens: usize) -> Result<Tensor>;
    fn from_config(vs: VarBuilder, cfg: &PretrainedConfig) -> Result<Self>;
}

pub enum ModelWrapper {
    Bigram(bigram::Bigram),
    Transformer(transformer::Transformer),
}

impl Model for ModelWrapper {
    fn generate(&mut self, idx: &Tensor, max_new_tokens: usize) -> Result<Tensor> {
        match self {
            Self::Bigram(b) => b.generate(idx, max_new_tokens),
            Self::Transformer(t) => t.generate(idx, max_new_tokens),
        }
    }
    fn from_config(vs: VarBuilder, cfg: &PretrainedConfig) -> Result<Self> {
        match cfg.architecture.as_str() {
            "bigram" => Ok(ModelWrapper::Bigram(Bigram::from_config(vs, cfg)?)),
            "transformer" => Ok(ModelWrapper::Transformer(Transformer::from_config(
                vs, cfg,
            )?)),
            _ => Err(candle_core::Error::Msg("Invalid architecture type".into())),
        }
    }
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
pub enum WhichModel {
    Bigram,
    Transformer,
}

impl Into<String> for WhichModel {
    fn into(self) -> String {
        match self {
            WhichModel::Bigram => "bigram".into(),
            WhichModel::Transformer => "transformer".into(),
        }
    }
}
