use std::{fs, path::PathBuf};

use crate::tokenizer;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error(transparent)]
    TokenizerError(tokenizer::TokenizerError),

    #[error(transparent)]
    IoError(std::io::Error),
}

#[derive(Debug, Clone)]
pub struct TextDataset {
    token_ids: Vec<u32>,
}

impl TextDataset {
    pub fn new<F>(data_files: &[PathBuf], tokenize: F) -> Result<Self, DatasetError>
    where
        F: Fn(&str) -> Result<tokenizer::Encoding, tokenizer::TokenizerError>,
    {
        let mut concat_ids: Vec<u32> = Vec::new();

        for file_path in data_files {
            let contents = fs::read_to_string(file_path).map_err(DatasetError::IoError)?;
            let encoding = tokenize(&contents).map_err(DatasetError::TokenizerError)?;
            concat_ids.extend(encoding.ids);
        }
        Ok(Self {
            token_ids: concat_ids,
        })
    }

    pub fn get_window(&self, start: usize, window_size: usize) -> Vec<u32> {
        self.token_ids[start..start + window_size].to_vec()
    }
}
