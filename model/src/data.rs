use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

const FRAME_BYTES: usize = config::MEL_FREQ_COUNT * 4 + config::INSTRUMENT_COUNT * 2;
const ONSETS_OFFSET: usize = config::MEL_FREQ_COUNT * 4;
const ACTIVES_OFFSET: usize = ONSETS_OFFSET + config::INSTRUMENT_COUNT;

#[derive(Clone, Debug)]
pub struct ChunkRef {
    pub file_idx: usize,
    pub frame_offset: usize,
}

struct TrackFile {
    path: PathBuf,
    mmap: Mmap,
    n_frames: usize,
}

pub struct EgmdDataset {
    files: Arc<[TrackFile]>,
    items: Vec<ChunkRef>,
}

impl EgmdDataset {
    pub fn load<P: AsRef<Path>>(root: P) -> std::io::Result<Self> {
        let mut files = Vec::new();
        let mut items = Vec::new();

        for entry in walkdir::WalkDir::new(&root) {
            let entry = entry.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().map_or(true, |e| e != "bin") {
                continue;
            }

            let file = std::fs::File::open(path)?;
            let size = file.metadata()?.len() as usize;
            if size % FRAME_BYTES != 0 {
                log::error!("skipping {path:?}: size {size} not multiple of {FRAME_BYTES}");
                continue;
            }
            let n_frames = size / FRAME_BYTES;
            if n_frames < config::TIME_CHUNK_LENGTH {
                continue;
            }

            let mmap = unsafe { Mmap::map(&file)? };
            let file_idx = files.len();
            files.push(TrackFile {
                path: path.to_owned(),
                mmap,
                n_frames,
            });

            let last = n_frames - config::TIME_CHUNK_LENGTH;
            let mut off = 0;
            while off <= last {
                items.push(ChunkRef {
                    file_idx,
                    frame_offset: off,
                });
                off += config::TIME_CHUNK_STRIDE;
            }
        }

        log::info!("Indexed {} files → {} chunks", files.len(), items.len());
        Ok(Self {
            files: files.into(),
            items,
        })
    }

    pub fn split(self, ratio: f32, seed: u64) -> (Self, Self) {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut items = self.items;
        items.shuffle(&mut rng);

        let cut = (items.len() as f32 * ratio) as usize;
        let valid_items = items.split_off(cut);

        let train = Self {
            files: Arc::clone(&self.files),
            items,
        };
        let valid = Self {
            files: self.files,
            items: valid_items,
        };
        (train, valid)
    }

    fn files(&self) -> Arc<[TrackFile]> {
        Arc::clone(&self.files)
    }
}

impl burn::data::dataset::Dataset<ChunkRef> for EgmdDataset {
    fn len(&self) -> usize {
        self.items.len()
    }
    fn get(&self, index: usize) -> Option<ChunkRef> {
        self.items.get(index).cloned()
    }
}

#[derive(Clone)]
pub struct ChunkBatcher {
    files: Arc<[TrackFile]>,
}

impl ChunkBatcher {
    pub fn new(dataset: &EgmdDataset) -> Self {
        Self { files: dataset.files() }
    }

    /// Read one chunk by slicing into the file's mmap. Returns (mels, onsets, actives)
    /// as flat `Vec<f32>`s already shaped for the tensor reshape downstream.
    fn read_chunk(&self, item: &ChunkRef) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let track = &self.files[item.file_idx];
        let start_byte = item.frame_offset * FRAME_BYTES;
        let end_byte = start_byte + config::TIME_CHUNK_LENGTH * FRAME_BYTES;
        let bytes = &track.mmap[start_byte..end_byte];

        let n_mel_values = config::TIME_CHUNK_LENGTH * config::MEL_FREQ_COUNT;
        let n_label_values = config::TIME_CHUNK_LENGTH * config::INSTRUMENT_COUNT;

        let mut mels = Vec::with_capacity(n_mel_values);
        let mut onsets = Vec::with_capacity(n_label_values);
        let mut actives = Vec::with_capacity(n_label_values);

        for frame in 0..config::TIME_CHUNK_LENGTH {
            let base = frame * FRAME_BYTES;

            // mels: read MEL_FREQ_COUNT f32s little-endian
            for m in 0..config::MEL_FREQ_COUNT {
                let off = base + m * 4;
                mels.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
            }

            // onsets and actives: u8 → f32 (0.0 or 1.0)
            let onsets_start = base + ONSETS_OFFSET;
            for c in 0..config::INSTRUMENT_COUNT {
                onsets.push(bytes[onsets_start + c] as f32);
            }
            let actives_start = base + ACTIVES_OFFSET;
            for c in 0..config::INSTRUMENT_COUNT {
                actives.push(bytes[actives_start + c] as f32);
            }
        }

        (mels, onsets, actives)
    }
}

use burn::data::dataloader::batcher::Batcher;
impl<B: burn::prelude::Backend> Batcher<B, ChunkRef, ChunkBatch<B>> for ChunkBatcher {
    fn batch(&self, items: Vec<ChunkRef>, device: &B::Device) -> ChunkBatch<B> {
        let batch_size = items.len();

        let mut spec_buf = Vec::with_capacity(batch_size * config::TIME_CHUNK_LENGTH * config::MEL_FREQ_COUNT);
        let mut onsets_buf = Vec::with_capacity(batch_size * config::TIME_CHUNK_LENGTH * config::INSTRUMENT_COUNT);
        let mut actives_buf = Vec::with_capacity(batch_size * config::TIME_CHUNK_LENGTH * config::INSTRUMENT_COUNT);

        for item in &items {
            let (mels, onsets, actives) = self.read_chunk(item);
            spec_buf.extend(mels);
            onsets_buf.extend(onsets);
            actives_buf.extend(actives);
        }

        let features = crate::features::ChunkFeatures::from_data(device, spec_buf, batch_size);
        let labels = crate::labels::ChunkLabels::from_data(device, onsets_buf, actives_buf, batch_size);

        ChunkBatch { features, labels }
    }
}

#[derive(Debug, Clone)]
pub struct ChunkBatch<B: burn::prelude::Backend> {
    pub features: crate::features::ChunkFeatures<B>,
    pub labels: crate::labels::ChunkLabels<B>,
}
