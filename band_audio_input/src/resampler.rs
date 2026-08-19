mod error;
#[cfg(test)]
mod test;

pub use error::ResamplerConstructionError;
pub use error::ResamplerError;

pub struct Resampler<I, const TARGET_SAMPLE_RATE: usize>
where
    I: crate::AudioInputStream,
{
    input_stream: I,
    resampler: rubato::Fft<f32>,

    input_buffer: Vec<f32>,
    input_buffer_end: usize,

    output_buffer: Vec<f32>,
    output_buffer_start: usize,
    output_buffer_end: usize,
}

impl<I, const TARGET_SAMPLE_RATE: usize> Resampler<I, TARGET_SAMPLE_RATE>
where
    I: crate::AudioInputStream,
{
    /// Create a new resampler from the given input stream.
    pub fn new(input_stream: I) -> Result<Self, ResamplerConstructionError> {
        use rubato::Resampler;

        let resampler = rubato::Fft::new(
            input_stream.sample_rate(),
            TARGET_SAMPLE_RATE,
            512, /* Bigger is less cpu but more latency, smaller is more cpu but less latency */
            1,
            rubato::FixedSync::Input,
        )?;
        let input_buffer = vec![0.0; resampler.input_frames_max()];
        let output_buffer = vec![0.0; resampler.output_frames_max()];

        tracing::info!("Created resampler<{} -> {}>", input_stream.sample_rate(), TARGET_SAMPLE_RATE);

        Ok(Self {
            input_stream,
            resampler,

            input_buffer,
            input_buffer_end: 0,

            output_buffer,
            output_buffer_start: 0,
            output_buffer_end: 0,
        })
    }

    /// Attempt to drain the resampler output buffer into the provided buffer.
    ///
    /// If the output buffer is fully empty, return true.
    /// Otherwise, if we still have some samples in the output buffer, return false.
    ///
    /// This also returns the total number of samples written in bost cases.
    fn drain_output_buffer(&mut self, buffer: &mut [f32]) -> usize {
        let available_samples = self.output_buffer_end - self.output_buffer_start;

        let samples_to_write = available_samples.min(buffer.len());

        let next_output_buffer_start = self.output_buffer_start + samples_to_write;

        let dest_buffer = &mut buffer[0..samples_to_write];
        let src_buffer = &self.output_buffer[self.output_buffer_start..next_output_buffer_start];
        dest_buffer.copy_from_slice(src_buffer);

        self.output_buffer_start = next_output_buffer_start;

        return samples_to_write;
    }

    /// Call the FFT resampler with the input / output buffers.
    ///
    /// When calling this function, the output buffer must be empty (it's content will be replaced)
    /// and the input buffer shall be filled (it's content will be read)
    fn resample_buffers(&mut self) -> Result<(), ResamplerError<I>> {
        use rubato::Resampler;
        use rubato::audioadapter_buffers::direct::InterleavedSlice;

        let buffer_in = InterleavedSlice::new(self.input_buffer.as_slice(), 1, self.input_buffer.len())?;
        let out_capacity = self.output_buffer.len();
        let mut buffer_out = InterleavedSlice::new_mut(self.output_buffer.as_mut_slice(), 1, out_capacity)?;

        let (_, frames_written) = self.resampler.process_into_buffer(&buffer_in, &mut buffer_out, None)?;

        /* Empty input buffer (fixed input guarantees we consumed it all) */
        self.input_buffer_end = 0;

        /* Fill output buffer */
        self.output_buffer_start = 0;
        self.output_buffer_end = frames_written;

        Ok(())
    }

    /// Call the FFT resampler with the incomplete input buffer.
    ///
    /// This shall only be used when the underlying audio input returned end of input,
    /// and we still have frames to process.
    fn resample_final_frames(&mut self) -> Result<(), ResamplerError<I>> {
        use rubato::Resampler;
        use rubato::audioadapter_buffers::direct::InterleavedSlice;

        let indexing = rubato::Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: Some(self.input_buffer_end),
            active_channels_mask: None,
        };
        let buffer_in = InterleavedSlice::new(self.input_buffer.as_slice(), 1, self.input_buffer.len())?;
        let out_capacity = self.output_buffer.len();
        let mut buffer_out = InterleavedSlice::new_mut(self.output_buffer.as_mut_slice(), 1, out_capacity)?;

        let (_, frames_written) = self
            .resampler
            .process_into_buffer(&buffer_in, &mut buffer_out, Some(&indexing))?;

        self.input_buffer_end = 0;

        /* Fill output buffer */
        self.output_buffer_start = 0;
        self.output_buffer_end = frames_written;

        Ok(())
    }
}

impl<I, const TARGET_SAMPLE_RATE: usize> crate::AudioInputStream for Resampler<I, TARGET_SAMPLE_RATE>
where
    I: crate::AudioInputStream + std::fmt::Debug,
{
    type InputError = ResamplerError<I>;

    fn sample_rate(&self) -> usize {
        TARGET_SAMPLE_RATE
    }

    fn next_samples(&mut self, buffer: &mut [f32]) -> Result<crate::NextSamplesResult, Self::InputError> {
        let mut written_sample_count: usize = 0;

        /* Loop while we have available samples */
        loop {
            if self.output_buffer_end > 0 {
                /* First, attempt to empty the output buffer */
                let frames_written = self.drain_output_buffer(&mut buffer[written_sample_count..]);
                written_sample_count += frames_written;

                if written_sample_count == buffer.len() {
                    /* buffer full, return */
                    return Ok(crate::NextSamplesResult::Some(written_sample_count));
                }
            }

            /* Ask for more samples for the next resampling */
            match self
                .input_stream
                .next_samples(&mut self.input_buffer[self.input_buffer_end..])
                .map_err(ResamplerError::InputStreamError)?
            {
                crate::NextSamplesResult::Some(samples) => {
                    #[cfg(debug_assertions)]
                    if samples == 0 {
                        tracing::error!(samples, "Inner stream returned Some(0) samples");
                    }
                    self.input_buffer_end += samples;
                    if self.input_buffer_end < self.input_buffer.len() {
                        /* Not enough for resampling, ask for more */
                        continue;
                    }
                }
                crate::NextSamplesResult::Unavailable => {
                    if written_sample_count == 0 {
                        return Ok(crate::NextSamplesResult::Unavailable);
                    } else {
                        return Ok(crate::NextSamplesResult::Some(written_sample_count));
                    }
                }
                crate::NextSamplesResult::EndOfInput => {
                    if self.input_buffer_end > 0 {
                        self.resample_final_frames()?;
                        continue;
                    } else {
                        if written_sample_count == 0 {
                            return Ok(crate::NextSamplesResult::EndOfInput);
                        } else {
                            return Ok(crate::NextSamplesResult::Some(written_sample_count));
                        }
                    }
                }
            }

            /* We both emptied the output buffer and filled the input buffer */
            /* We can ask for resampling */
            self.resample_buffers()?;

            /* Next loop iteration will start sending the output buffer, then resample again, etc */
        }
    }
}
