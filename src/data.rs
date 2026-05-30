use std::path::{Path, PathBuf};

/// One sample = "load file at this path, take chunk starting at this frame".
#[derive(Clone, Debug)]
pub struct ChunkRef {
    pub path: PathBuf,
    pub frame_offset: usize,
}

pub struct EgmdDataset {
    items: Vec<ChunkRef>,
}

impl EgmdDataset {
    /// Walks the split directory, peeks at each npz to learn its frame count,
    /// and precomputes a list of (path, offset) chunks covering the data.
    pub fn load<P: AsRef<Path>>(track_dir: P) -> std::io::Result<Self> {
        let mut items = Vec::new();
        let mut n_files = 0usize;

        for entry in std::fs::read_dir(&track_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let path = entry.path();

            n_files += 1;
            let n_frames = match get_frame_count(&path) {
                Ok(n) => n,
                Err(e) => {
                    log::error!("skipping {path:?}: {e}");
                    continue;
                }
            };
            if n_frames < crate::config::TIME_CHUNK_LENGTH {
                continue;
            }

            // Sliding window with configured stride.
            // last valid offset = n_frames - chunk_frames
            let last = n_frames - crate::config::TIME_CHUNK_LENGTH;
            let mut off = 0;
            while off <= last {
                items.push(ChunkRef {
                    path: path.clone(),
                    frame_offset: off,
                });
                off += crate::config::TIME_CHUNK_STRIDE;
            }
        }

        log::info!(
            "Indexed {} files → {} chunks ({}-frame chunks, stride {})",
            n_files,
            items.len(),
            crate::config::TIME_CHUNK_LENGTH,
            crate::config::TIME_CHUNK_STRIDE
        );
        Ok(Self { items })
    }

    pub fn load_train<P: AsRef<Path>>(root: P) -> std::io::Result<Self> {
        Self::load(root.as_ref().join("train"))
    }
    pub fn load_valid<P: AsRef<Path>>(root: P) -> std::io::Result<Self> {
        Self::load(root.as_ref().join("valid"))
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

/// Just reads the npz header to learn the number of frames in `spec`.
fn get_frame_count(path: &Path) -> std::io::Result<usize> {
    let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(path)?)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    let spec: ndarray::Array2<f32> = npz
        .by_name("spec")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    Ok(spec.nrows())
}

#[derive(Clone, Default)]
pub struct ChunkBatcher;

impl ChunkBatcher {
    fn read_chunk(&self, item: &ChunkRef) -> std::io::Result<(ndarray::Array2<f32>, ndarray::Array2<f32>, ndarray::Array2<f32>)> {
        let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(&item.path)?)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let spec: ndarray::Array2<f32> = npz
            .by_name("spec")
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let onsets: ndarray::Array2<f32> = npz
            .by_name("onsets")
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let actives: ndarray::Array2<f32> = npz
            .by_name("actives")
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let s_offset = item.frame_offset;
        let e_offset = s_offset + crate::config::TIME_CHUNK_LENGTH;
        Ok((
            spec.slice(ndarray::s![s_offset..e_offset, ..]).to_owned(),
            onsets.slice(ndarray::s![s_offset..e_offset, ..]).to_owned(),
            actives.slice(ndarray::s![s_offset..e_offset, ..]).to_owned(),
        ))
    }
}

use burn::data::dataloader::batcher::Batcher;
impl<B: burn::prelude::Backend> Batcher<B, ChunkRef, ChunkBatch<B>> for ChunkBatcher {
    fn batch(&self, items: Vec<ChunkRef>, device: &B::Device) -> ChunkBatch<B> {
        let batch_size = items.len();

        let mut spec_buf = Vec::with_capacity(batch_size * crate::config::TIME_CHUNK_LENGTH * crate::config::MEL_FREQ_COUNT);
        let mut onsets_buf = Vec::with_capacity(batch_size * crate::config::TIME_CHUNK_LENGTH * crate::config::INSTRUMENT_COUNT);
        let mut actives_buf = Vec::with_capacity(batch_size * crate::config::TIME_CHUNK_LENGTH * crate::config::INSTRUMENT_COUNT);

        for item in &items {
            let (spec, onsets, actives) = match self.read_chunk(item) {
                Ok(data) => data,
                Err(e) => {
                    log::error!("Failed to read {:?}: {e}", item.path);
                    continue;
                }
            };

            match spec.as_slice() {
                Some(spec_data) => spec_buf.extend_from_slice(spec_data),
                None => {
                    log::error!("Failed to read {:?}: spec data not contiguous", item.path);
                    continue;
                }
            }
            match onsets.as_slice() {
                Some(onsets_data) => onsets_buf.extend_from_slice(onsets_data),
                None => {
                    log::error!("Failed to read {:?}: onsets data not contiguous", item.path);
                    continue;
                }
            }
            match actives.as_slice() {
                Some(actives_data) => actives_buf.extend_from_slice(actives_data),
                None => {
                    log::error!("Failed to read {:?}: actives data not contiguous", item.path);
                    continue;
                }
            }
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
