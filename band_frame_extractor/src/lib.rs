mod frame_buffer;
mod frame_extractor;
mod spectral_frame;

pub use frame_buffer::FrameBuffer;
pub use frame_extractor::FrameExtractor;
pub use frame_extractor::FrameExtractorError;
pub use frame_extractor::NextFrameResult;
pub use spectral_frame::SpectralFrame;

pub type StandardFrameBuffer<I> = FrameBuffer<I, 256, 22050, 2048, 256, 1025>;
pub type StandardFrameExtractor<I> = FrameExtractor<I, 22050, 2048, 256, 1025>;
