use crate::tokenizer::trainer::Trainer;
use crate::tokenizer::Token;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use super::TokenizerError;

pub mod character;

pub trait Model {
    type Trainer: Trainer;
    // Required methods
    fn tokenize(&self, text: &str) -> Result<Vec<Token>, TokenizerError>;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    fn id_to_token(&self, id: u32) -> Option<String>;
    fn get_vocab(&self) -> HashMap<String, u32>;
    fn get_vocab_size(&self) -> usize;
    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>, io::Error>;
    fn get_trainer(&self) -> Self::Trainer;
}
