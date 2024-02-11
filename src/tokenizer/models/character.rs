use crate::tokenizer::{models::Model, Token, TokenizerError};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub mod trainer;
use serde::{Deserialize, Serialize};
use serde_json;
use trainer::CharacterTrainer;

pub type Vocab = HashMap<String, u32>;
type VocabR = HashMap<u32, String>;

#[derive(Serialize)]
pub struct Character {
    pub vocab: Vocab,
    #[serde(skip_serializing, skip_deserializing)]
    vocab_r: VocabR,
}

impl<'de> Deserialize<'de> for Character {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vocab: Vocab = Deserialize::deserialize(deserializer)?;
        let vocab_r: VocabR = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Ok(Character { vocab, vocab_r })
    }
}

impl Model for Character {
    type Trainer = CharacterTrainer;
    fn get_trainer(&self) -> Self::Trainer {
        Self::Trainer::new()
    }
    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }
    fn tokenize(&self, text: &str) -> Result<Vec<Token>, TokenizerError> {
        text.chars()
            .map(|c| c.to_string())
            .enumerate()
            .map(|(i, c)| {
                Ok(Token {
                    id: self
                        .token_to_id(&c)
                        .ok_or(TokenizerError::UnsupportedCharacter(
                            "Unknown token without <unk>".into(),
                        ))?,
                    value: c,
                    offsets: (i, i),
                })
            })
            .collect()
    }
    fn save(
        &self,
        folder: &std::path::Path,
        name: Option<&str>,
    ) -> Result<Vec<std::path::PathBuf>, std::io::Error> {
        let fname = match name {
            Some(n) => format!("{}-vocab.json", n),
            None => "vocab.json".to_string(),
        };
        let vocab_path: PathBuf = [folder, Path::new(fname.as_str())].iter().collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let contents: String = serde_json::to_string(&self)?;
        vocab_file.write_all(contents.as_bytes())?;
        Ok(vec![vocab_path])
    }
}
