/// The frame buffer allows to keep multiple extracted frames
/// and expose a sliding window over each frames.
pub struct FrameBuffer<
    I,
    const BUFFER_SIZE: usize,
    const SAMPLE_RATE: usize,
    const WINDOW_SIZE: usize,
    const HOP: usize,
    const N_BINS: usize,
> where
    I: band_audio_input::AudioInputStream,
{
    /// Inner frame extractor used to pull spectrum frame
    frame_extractor: crate::FrameExtractor<I, SAMPLE_RATE, WINDOW_SIZE, HOP, N_BINS>,
    /// Double ring buffer for the spectral frames.
    /// The length is BUFFER_SIZE * 2
    double_ring_buffer: Box<[crate::SpectralFrame<N_BINS>]>,
    /// Next slot to write in the double ring buffer.
    /// This also is the first index of the window slice in the double ring buffer.
    next_frame_slot: usize,
}

impl<I, const BUFFER_SIZE: usize, const SAMPLE_RATE: usize, const WINDOW_SIZE: usize, const HOP: usize, const N_BINS: usize>
    FrameBuffer<I, BUFFER_SIZE, SAMPLE_RATE, WINDOW_SIZE, HOP, N_BINS>
where
    I: band_audio_input::AudioInputStream,
{
    pub fn new(frame_extractor: crate::FrameExtractor<I, SAMPLE_RATE, WINDOW_SIZE, HOP, N_BINS>) -> Self {
        let double_ring_buffer = vec![crate::SpectralFrame::ZERO; BUFFER_SIZE * 2].into_boxed_slice();

        Self {
            frame_extractor,
            double_ring_buffer,
            next_frame_slot: 0,
        }
    }

    /// Get the next window. This will call the frame extractor underlying next_frame function.
    /// When a new frame is available, this will advance the window and return a view of it.
    /// Otherwise, this will return the unavailable result.
    pub fn next_window(&mut self) -> Result<FrameWindowResult<'_, BUFFER_SIZE, N_BINS>, crate::FrameExtractorError<I>> {
        let next_frame = match self.frame_extractor.next_frame()? {
            crate::NextFrameResult::Unavailable => return Ok(FrameWindowResult::Unavailable),
            crate::NextFrameResult::EndOfInput => return Ok(FrameWindowResult::EndOfInput),
            crate::NextFrameResult::Frame(frame) => frame,
        };

        /* Store the frame at the next available pos and the mirrored position */
        self.double_ring_buffer[self.next_frame_slot] = next_frame.clone();
        self.double_ring_buffer[self.next_frame_slot + BUFFER_SIZE] = next_frame;

        /* Bump the window position */
        self.next_frame_slot = (self.next_frame_slot + 1) % BUFFER_SIZE;

        /* The full window is available at next_frame_slot..next_frame_slot + BUFFER_SIZE */
        let window = &self.double_ring_buffer[self.next_frame_slot..self.next_frame_slot + BUFFER_SIZE];
        let window: &[crate::SpectralFrame<N_BINS>; BUFFER_SIZE] = window.try_into().unwrap();

        Ok(FrameWindowResult::Window(window))
    }
}

pub enum FrameWindowResult<'slice, const BUFFER_SIZE: usize, const N_BINS: usize> {
    Window(&'slice [crate::SpectralFrame<N_BINS>; BUFFER_SIZE]),
    Unavailable,
    EndOfInput,
}
