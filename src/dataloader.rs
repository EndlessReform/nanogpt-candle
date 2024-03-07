use crate::datasets::TextDataset;
use candle_core::error::Error;
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TextDatasetIteratorError {
    #[error("Not enough data for one batch: {0}")]
    TooShort(String),
}

pub struct TextDatasetIterator<'a> {
    dataset: &'a TextDataset,
    /// Position in the vec of indices
    current_pos: usize,
    shuffled_indices: Vec<usize>,
    pub context_len: usize,
    device: &'a Device,
}

impl<'a> Iterator for TextDatasetIterator<'a> {
    type Item = Result<(Tensor, Tensor), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(&start_idx) = self.shuffled_indices.get(self.current_pos) {
            if self.context_len * (self.current_pos + 1) > self.dataset.len() {
                // Can't get a full batch of labels + targets; swallowing it
                return None;
            }
            let maybe_window = self.dataset.get_window(start_idx, self.context_len + 1);
            match maybe_window {
                Some(window) => {
                    self.current_pos += 1;
                    let x = Tensor::new(&window[..self.context_len], self.device);
                    let y = Tensor::new(&window[1..], self.device);
                    Some(x.and_then(|x| y.map(|y| (x, y))))
                }
                None => None,
            }
        } else {
            None
        }
    }
}

impl<'a> TextDatasetIterator<'a> {
    pub fn new(
        dataset: &'a TextDataset,
        context_len: usize,
        device: &'a Device,
    ) -> Result<Self, TextDatasetIteratorError> {
        // Shuffle indices
        if context_len >= dataset.len() {
            // Not enough for batch and preds
            return Err(TextDatasetIteratorError::TooShort(format!(
                "{} < {}",
                context_len,
                dataset.len(),
            )));
        }
        // Leave out final batch
        let mut start_indices: Vec<usize> = (0..dataset.len() - context_len)
            .step_by(context_len)
            .collect();
        start_indices.shuffle(&mut thread_rng());
        Ok(TextDatasetIterator {
            dataset,
            device,
            shuffled_indices: start_indices,
            context_len,
            current_pos: 0,
        })
    }
}

// pub fn to_batcher(
//     d: TextDataset,
//     context_len: usize,
//     batch_size: usize,
//     device: &candle_core::Device,
// ) -> Batcher<IterResult2<Box<dyn Iterator<Item = Result<(Tensor, Tensor), Error>>>>> {
//     // Heavily borrowed from llama2-c implementation
//     // https://github.com/huggingface/candle/blob/de11623752edbeb42c233256dfc83f56b688e61b/candle-examples/examples/llama2-c/main.rs#L194C5-L234C8
//     // TODO: shuffle
//     let iter = start_indices.into_iter().flat_map(|start_idx| {
//         if start_idx + context_len + 1 > d.len() {
//             // Can't fill full batch plus next token: just throw out the data
//             None
//         } else {
//             let window = d.get_window(start_idx, context_len + 1);
//             let x = Tensor::new(&window[..context_len], device);
//             let y = Tensor::new(&window[1..], device);
//             // Fail if x works; else fold x into
//             Some(x.and_then(|x| y.map(|y| (x, y))))
//         }
//     });
//     Batcher::new_r2(iter).batch_size(batch_size)
// }

#[cfg(test)]
mod tests {
    use super::TextDatasetIterator;
    use crate::datasets::TextDataset;
    use candle_core::{error::Error, Tensor};
    use candle_datasets::Batcher;

    #[test]
    fn test_correctness() {
        let dataset = TextDataset {
            token_ids: (0..65).collect(),
        };
        let device = candle_core::Device::cuda_if_available(0).unwrap();
        let mut iterator = TextDatasetIterator::new(&dataset, 8, &device).unwrap();
        let first_batch = iterator.next();

        // We can retrieve a single batch
        assert!(first_batch.is_some());
        let (x, y) = first_batch.unwrap().unwrap();
        assert_eq!(x.shape().dims(), [8]);
        assert_eq!(y.shape().dims(), [8]);
        println!("{:?}", x);

        // There are 7 batches left
        let rest: Vec<Result<(Tensor, Tensor), Error>> = iterator.collect();
        for t in rest.iter() {
            println!("{:?}", t)
        }

        assert_eq!(rest.len(), 7);
    }

    #[test]
    fn test_batch() {
        let dataset = TextDataset {
            token_ids: (0..65).collect(),
        };
        let device = candle_core::Device::cuda_if_available(0).unwrap();
        let iterator = TextDatasetIterator::new(&dataset, 8, &device).unwrap();
        let mut batcher = Batcher::new_r2(iterator).batch_size(8);
        assert!(batcher.next().is_some());
    }
}
