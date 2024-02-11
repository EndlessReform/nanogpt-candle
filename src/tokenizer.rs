use thiserror::Error;

pub mod models;
pub mod trainer;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Unsupported character: {0}")]
    UnsupportedCharacter(String),
}

#[derive(Debug, Clone)]
pub struct Encoding {
    ids: Vec<u32>,
    type_ids: Vec<u32>,
    offsets: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
}

impl Token {
    pub fn new(id: u32, value: String, offsets: (usize, usize)) -> Self {
        Token { id, value, offsets }
    }
}

trait Tokenizer {
    fn encode(&self, input: &str) -> Result<Encoding, TokenizerError>;
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError>;
}
