pub struct SpectralFrame<const FREQ_BIN_COUNT: usize> {
    index: usize,
    bins: [f32; FREQ_BIN_COUNT],
}

impl<const FREQ_BIN_COUNT: usize> SpectralFrame<FREQ_BIN_COUNT> {
    pub fn from_complex(index: usize, complex: &[realfft::num_complex::Complex<f32>; FREQ_BIN_COUNT]) -> Self {
        let window_size: usize = (FREQ_BIN_COUNT - 1) * 2;
        let norm = 2.0 / (window_size as f32 * 0.5);
        let bins = std::array::from_fn(|i| complex[i].norm() * norm);
        Self { index, bins }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn bins(&self) -> &[f32; FREQ_BIN_COUNT] {
        &self.bins
    }
}
