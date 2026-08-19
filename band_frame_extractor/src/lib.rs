mod frame_extractor;

pub use frame_extractor::FrameExtractor;
pub type StandardFrameExtractor<I> = FrameExtractor<I, 22100>;
