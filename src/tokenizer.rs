use std::collections::HashMap;

use thiserror::Error;

use self::models::{Model, ModelWrapper};

pub mod models;
pub mod trainer;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Unsupported character: {0}")]
    UnsupportedCharacter(String),
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
    pub fn new(model: impl Into<ModelWrapper>) -> Self {
        Self {
            model_wrapper: model.into(),
        }
    }
    pub fn encode(&self, input: &str) -> Result<Encoding, TokenizerError> {
        Ok(self.model_wrapper.tokenize(input)?.into())
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let res: Result<String, TokenizerError> = ids
            .into_iter()
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
}
