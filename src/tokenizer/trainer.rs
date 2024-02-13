use thiserror::Error;

use super::models::{character::trainer::CharacterTrainer, Model, ModelWrapper};

#[derive(Error, Debug)]
pub enum TrainerError {
    #[error(transparent)]
    ProcessorError(#[from] Box<dyn std::error::Error>),
    #[error("Mismatching model: {0}")]
    InvalidModel(String),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

pub trait Trainer {
    type Model: Model;
    fn train(&self, model: &mut Self::Model) -> Result<(), TrainerError>;
    /// Process an iterator of strings and feed the tokens to the model
    fn feed<I, S, F>(&mut self, iterator: I, processor: F) -> Result<(), TrainerError>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
        F: Fn(&str) -> Result<Vec<String>, Box<dyn std::error::Error>>;
}

pub enum TrainerWrapper {
    CharacterTrainer(CharacterTrainer),
}

impl Trainer for TrainerWrapper {
    type Model = ModelWrapper;
    fn feed<I, S, F>(&mut self, iterator: I, processor: F) -> Result<(), TrainerError>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
        F: Fn(&str) -> Result<Vec<String>, Box<dyn std::error::Error>>,
    {
        match self {
            Self::CharacterTrainer(c) => c.feed(iterator, processor),
        }
    }
    fn train(&self, model: &mut Self::Model) -> Result<(), TrainerError> {
        match self {
            Self::CharacterTrainer(c) => match model {
                ModelWrapper::Character(m) => c.train(m),
            },
        }
    }
}
