use std::error::Error;

use super::models::Model;

pub type TrainerError = Box<dyn Error>;

pub trait Trainer {
    type Model: Model;
    fn train(&self, model: &mut Self::Model) -> Result<(), TrainerError>;
    /// Process an iterator of strings and feed the tokens to the model
    fn feed<I, S, F>(&mut self, iterator: I, processor: F) -> Result<(), TrainerError>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
        F: Fn(&str) -> Result<Vec<String>, Box<dyn Error>>;
}
