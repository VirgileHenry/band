/// Dummy input stream that continuously generate a sin wave signal.
///
/// This is only intended for testing.
#[derive(Debug)]
pub struct SinInputStream {
    frequency: f32,
    sample_rate: usize,
    max_samples: Option<usize>,
    generated_samples: usize,
    phase: f32,
}

impl SinInputStream {
    pub fn new(frequency: f32, sample_rate: usize, max_samples: Option<usize>) -> Self {
        tracing::info!(frequency, sample_rate, max_samples, "Created sinusoidal input stream");
        Self {
            frequency,
            sample_rate,
            max_samples,
            generated_samples: 0,
            phase: 0.0,
        }
    }

    fn next_sample(&mut self) -> Option<f32> {
        if let Some(max_samples) = self.max_samples {
            if self.generated_samples >= max_samples {
                return None;
            }
        }
        self.phase += std::f32::consts::TAU * self.frequency / self.sample_rate as f32;
        if self.phase >= std::f32::consts::TAU {
            self.phase -= std::f32::consts::TAU;
        }
        let sample = self.phase.sin();
        self.generated_samples += 1;
        Some(sample)
    }
}

impl crate::AudioInputStream for SinInputStream {
    type InputError = std::convert::Infallible;

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn next_samples(&mut self, buffer: &mut [f32]) -> Result<crate::NextSamplesResult, Self::InputError> {
        let mut written = 0;
        for elem in buffer.iter_mut() {
            let next_sample = match self.next_sample() {
                Some(sample) => sample,
                None => {
                    if written == 0 {
                        return Ok(crate::NextSamplesResult::EndOfInput);
                    } else {
                        return Ok(crate::NextSamplesResult::Some(written));
                    }
                }
            };
            *elem = next_sample;
            written += 1;
        }
        Ok(crate::NextSamplesResult::Some(written))
    }
}
