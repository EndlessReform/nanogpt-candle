use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    /// Safetensors filename to load weights from. Will be passed through to hf_hub
    pub load_from: Option<String>,
    /// Safetensors filename to save weights to.
    pub save_to: Option<String>,
}

impl TrainingConfig {
    pub fn bigram_default() -> Self {
        Self {
            learning_rate: 1e-3,
            epochs: 3,
            batch_size: 32,
            load_from: None,
            save_to: Some("models/bigram/model.safetensors".into()),
        }
    }

    pub fn transformer_default() -> Self {
        Self {
            learning_rate: 1e-3,
            epochs: 2,
            batch_size: 32,
            load_from: None,
            save_to: Some("models/transformer/model.safetensors".into()),
        }
    }

    pub fn from_json_file(path: &PathBuf) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let config = serde_json::from_str(&contents)?;
        Ok(config)
    }

    pub fn to_json_file(&self, path: &PathBuf) -> Result<()> {
        let data: String = serde_json::to_string_pretty(&self)?;
        fs::write(path, data)?;
        Ok(())
    }
}
