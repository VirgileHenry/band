#[derive(Debug, Clone)]
pub struct SpectralFrame<const N_BINS: usize> {
    index: usize,
    bins: [f32; N_BINS],
}

impl<const N_BINS: usize> SpectralFrame<N_BINS> {
    pub const ZERO: Self = Self {
        index: 0,
        bins: [0.0; N_BINS],
    };

    pub fn from_complex(index: usize, complex: &[realfft::num_complex::Complex<f32>; N_BINS]) -> Self {
        let window_size: usize = (N_BINS - 1) * 2;
        let norm = 2.0 / (window_size as f32 * 0.5);
        let bins = std::array::from_fn(|i| complex[i].norm() * norm);
        Self { index, bins }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn bins(&self) -> &[f32; N_BINS] {
        &self.bins
    }
}
