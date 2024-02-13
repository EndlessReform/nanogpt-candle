use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use thiserror::Error;

use self::models::{Model, ModelWrapper};
use self::trainer::{Trainer, TrainerError};

pub mod models;
pub mod trainer;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Unsupported character: {0}")]
    UnsupportedCharacter(String),

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error(transparent)]
    OtherError(#[from] Box<dyn std::error::Error>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
}

#[derive(Debug, Clone)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub offsets: Vec<(usize, usize)>,
}

impl From<Vec<Token>> for Encoding {
    fn from(value: Vec<Token>) -> Self {
        let (ids, offsets): (Vec<u32>, Vec<(usize, usize)>) =
            value.into_iter().map(|t| (t.id, t.offsets)).unzip();
        Encoding {
            type_ids: vec![0; ids.len()],
            ids,
            // Ignore type id for now
            offsets,
        }
    }
}

impl Token {
    pub fn new(id: u32, value: String, offsets: (usize, usize)) -> Self {
        Token { id, value, offsets }
    }
}

pub struct Tokenizer {
    model_wrapper: ModelWrapper,
}

impl Tokenizer {
    pub fn new(model_wrapper: ModelWrapper) -> Self {
        Self { model_wrapper }
    }
    pub fn encode(&self, input: &str) -> Result<Encoding, TokenizerError> {
        Ok(self.model_wrapper.tokenize(input)?.into())
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let res: Result<String, TokenizerError> = ids
            .iter()
            .map(|id| {
                self.model_wrapper
                    .id_to_token(*id)
                    .ok_or(TokenizerError::UnsupportedCharacter(
                        "Unknown not handled yet!".into(),
                    ))
            })
            .collect();
        res
    }
    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.model_wrapper.get_vocab()
    }
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model_wrapper.token_to_id(token)
    }
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.model_wrapper.id_to_token(id)
    }
    pub fn train<I, S>(&mut self, sequences: I) -> Result<&mut Self, TrainerError>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        let mut trainer = self.model_wrapper.get_trainer();
        // Ignore processing for now
        trainer.feed(sequences, |p| Ok(vec![p.to_string()]))?;
        trainer.train(&mut self.model_wrapper)?;
        Ok(self)
    }
    pub fn train_from_files(&mut self, files: Vec<PathBuf>) -> Result<&mut Self, TrainerError> {
        // Ingest files
        for path in files {
            let file = File::open(&path).map_err(TrainerError::IoError)?;
            let reader = BufReader::with_capacity(1_000_000, file);
            let line_iter = reader.lines().filter_map(|line_result| {
                match line_result {
                    Ok(line) => Some(line), // Convert line into T here, if necessary
                    Err(_) => None,         // Skip or handle errors differently
                }
            });
            let mut trainer = self.model_wrapper.get_trainer();
            trainer.feed(line_iter, |p| Ok(vec![p.to_string()]))?;
            trainer.train(&mut self.model_wrapper)?;
        }
        Ok(self)
    }
    /// Just persist model for now
    pub fn save(
        &self,
        folder: &PathBuf,
        name: Option<&str>,
    ) -> Result<Vec<PathBuf>, std::io::Error> {
        self.model_wrapper.save(folder, name)
    }
    pub fn from_file(path: &PathBuf) -> Result<Self, TokenizerError> {
        let bytes = fs::read(path).map_err(TokenizerError::IoError)?;
        let json_string = String::from_utf8(bytes)
            .map_err(|_| TokenizerError::InvalidInput("Cannot parse file".into()))?;
        let model = serde_json::from_str::<ModelWrapper>(&json_string)
            .map_err(|_| TokenizerError::InvalidInput("Fuck my life".into()))?;
        Ok(Self {
            model_wrapper: model,
        })
    }
}
