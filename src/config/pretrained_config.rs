use std::fs::{self, read_to_string};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error(transparent)]
    ParseError(serde_json::Error),

    #[error(transparent)]
    IoError(std::io::Error),
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct PretrainedConfig {
    pub architecture: String,
    pub context_size: u32,
    /// Hidden dimension of model
    pub vocab_size: u32,
    pub hidden_size: u32,
    /// MLP hidden dimension. Recommend 4 * hidden_size
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub hidden_layers: u32,
    /// Name of tokenizer used
    pub tokenizer_id: String,
}

impl PretrainedConfig {
    pub fn from_json_file(path: &PathBuf) -> Result<Self, ConfigError> {
        let contents = read_to_string(path).map_err(ConfigError::IoError)?;
        serde_json::from_str(&contents).map_err(ConfigError::ParseError)
    }
    pub fn to_json_file(&self, path: &PathBuf) -> Result<(), ConfigError> {
        let data: String = serde_json::to_string_pretty(&self).map_err(ConfigError::ParseError)?;
        fs::write(path, data).map_err(ConfigError::IoError)
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;

    #[test]
    fn test_load_from_file() {
        let out_path: PathBuf = env::temp_dir().join("config-test.json");
        // Gosh, what a huge model!
        let sample_config = super::PretrainedConfig {
            architecture: "GPT-69".into(),
            context_size: 1,
            hidden_size: 1,
            vocab_size: 1,
            intermediate_size: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            hidden_layers: 1,
            tokenizer_id: "rick-astley-base-100k".into(),
        };
        sample_config.to_json_file(&out_path).unwrap();
        let read_config = PretrainedConfig::from_json_file(&out_path).unwrap();
        assert_eq!(sample_config, read_config)
    }
}
