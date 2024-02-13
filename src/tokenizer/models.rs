use crate::tokenizer::trainer::Trainer;
use crate::tokenizer::Token;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use super::trainer::TrainerWrapper;
use super::TokenizerError;

pub mod character;
use character::Character;

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

#[derive(Serialize, Deserialize)]
pub enum ModelWrapper {
    Character(Character),
}

impl From<Character> for ModelWrapper {
    fn from(c: Character) -> Self {
        Self::Character(c)
    }
}

impl Model for ModelWrapper {
    type Trainer = TrainerWrapper;
    fn tokenize(&self, tokens: &str) -> Result<Vec<Token>, TokenizerError> {
        match self {
            Self::Character(c) => c.tokenize(tokens),
        }
    }
    fn get_trainer(&self) -> Self::Trainer {
        match self {
            Self::Character(c) => TrainerWrapper::CharacterTrainer(c.get_trainer()),
        }
    }
    fn get_vocab(&self) -> HashMap<String, u32> {
        match self {
            Self::Character(c) => c.get_vocab(),
        }
    }
    fn get_vocab_size(&self) -> usize {
        match self {
            Self::Character(c) => c.get_vocab_size(),
        }
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        match self {
            Self::Character(c) => c.id_to_token(id),
        }
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Character(c) => c.token_to_id(token),
        }
    }
    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>, io::Error> {
        match self {
            Self::Character(c) => c.save(folder, name),
        }
    }
}
