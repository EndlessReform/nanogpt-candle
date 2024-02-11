use super::Character;
use crate::tokenizer::trainer::{Trainer, TrainerError};
use crate::tokenizer::Token;
use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub struct CharacterTrainer {
    pub special_tokens: Vec<Token>,
    /// All chars encountered during tokenizer training
    chars: HashSet<String>,
}

impl CharacterTrainer {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
    pub fn do_train(
        &self,
        chars: &HashSet<String>,
        model: &mut Character,
    ) -> Result<(), TrainerError> {
        // Create token index for char entries, alphabetized
        let mut items: Vec<String> = chars.iter().cloned().collect();
        items.sort();
        let token_map: HashMap<String, u32> =
            HashMap::from_iter(items.into_iter().enumerate().map(|(i, t)| (t, i as u32)));

        // Persist to model
        model.vocab = token_map;
        Ok(())
    }
}

impl Trainer for CharacterTrainer {
    type Model = Character;
    fn train(&self, model: &mut Self::Model) -> Result<(), TrainerError> {
        self.do_train(&self.chars, model)
    }
    fn feed<I, S, F>(&mut self, iterator: I, processor: F) -> Result<(), TrainerError>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
        F: Fn(&str) -> Result<Vec<String>, Box<dyn std::error::Error>>,
    {
        let chars: Result<Vec<Vec<char>>, TrainerError> = iterator
            .map(|seq| {
                processor(seq.as_ref()).map(|strings| {
                    strings
                        .into_iter()
                        .flat_map(|s| s.chars().collect::<Vec<char>>())
                        .collect::<Vec<char>>()
                })
            })
            .collect();
        let char_map: HashSet<String> =
            chars?
                .into_iter()
                .flatten()
                .fold(HashSet::new(), |mut acc, c| {
                    acc.insert(c.to_string());
                    acc
                });
        self.chars = char_map;
        Ok(())
    }
}
