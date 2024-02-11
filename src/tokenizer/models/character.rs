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

impl Character {
    pub fn new(vocab: Vocab) -> Self {
        let vocab_r: VocabR = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self { vocab, vocab_r }
    }
}

// Temporary struct for deserialization
#[derive(Deserialize)]
struct TempCharacter {
    vocab: Vocab,
}

impl From<TempCharacter> for Character {
    fn from(temp: TempCharacter) -> Self {
        Character::new(temp.vocab)
    }
}

impl<'de> Deserialize<'de> for Character {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let temp_character: TempCharacter = Deserialize::deserialize(deserializer)?;
        Ok(temp_character.into())
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

#[cfg(test)]
mod tests {
    use std::{env::temp_dir, fs};

    use super::*;
    use serde_json;

    #[test]
    fn test_tokenize() {
        let vocab: Vocab = [("a".into(), 0), ("b".into(), 1), ("c".into(), 2)]
            .iter()
            .cloned()
            .collect();

        let model = Character::new(vocab);
        let test_string: String = "cab".into();
        let tokens = model.tokenize(&test_string).unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(
            tokens[0],
            Token {
                id: 2,
                value: "c".into(),
                offsets: (0, 0)
            }
        )
    }

    #[test]
    fn test_save() {
        let tmp = temp_dir();
        let vocab: Vocab = [("a".into(), 0), ("b".into(), 1), ("c".into(), 2)]
            .iter()
            .cloned()
            .collect();

        // Persist to JSON
        let model = Character::new(vocab);
        let dest_paths = model.save(&tmp, Some("foo")).unwrap();
        assert_eq!(dest_paths.len(), 1);

        // Load from JSON, verify correctness
        let bytes = fs::read(&dest_paths[0]).unwrap();
        let json_string = String::from_utf8(bytes).unwrap();
        let model2: Character = serde_json::from_str(&json_string).unwrap();
        assert_eq!(model2.get_vocab_size(), 3);
    }
}
