use std::{fs, path::PathBuf};

use crate::tokenizer;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error(transparent)]
    TokenizerError(tokenizer::TokenizerError),

    #[error(transparent)]
    IoError(std::io::Error),

    #[error("Invalid split percentage: {0}")]
    InvalidSplit(String),
}

#[derive(Debug, Clone)]
pub struct TextDataset {
    pub token_ids: Vec<u32>,
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

    pub fn get_window(&self, start: usize, window_size: usize) -> Option<Vec<u32>> {
        self.token_ids
            .get(start..start + window_size)
            .map(|slice| slice.to_vec())
    }

    pub fn train_test_split(
        &self,
        test_pct: f64,
    ) -> Result<(TextDataset, TextDataset), DatasetError> {
        if test_pct > 1.0 || test_pct < 0.0 {
            return Err(DatasetError::InvalidSplit(format!(
                "{} is not in 0..1",
                test_pct
            )));
        }

        let test_amt = (test_pct * self.token_ids.len() as f64).floor();
        let start_idx: usize = self.token_ids.len() - test_amt as usize;
        Ok((
            TextDataset {
                token_ids: self.token_ids[0..start_idx].to_vec(),
            },
            TextDataset {
                token_ids: self.token_ids[start_idx..self.token_ids.len()].to_vec(),
            },
        ))
    }

    pub fn len(&self) -> usize {
        self.token_ids.len()
    }
}
