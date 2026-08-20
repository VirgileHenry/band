mod frame_extractor;
mod spectral_frame;

pub use frame_extractor::FrameExtractor;
pub use frame_extractor::NextFrameResult;
pub use spectral_frame::SpectralFrame;

pub type StandardFrameExtractor<I> = FrameExtractor<I, 22100, 2048, 256, 1025>;
