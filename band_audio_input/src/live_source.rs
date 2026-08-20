mod available_source;
mod error;

pub use available_source::AvailableSource;
pub use available_source::AvailableSourceLiveCapture;
use cpal::traits::StreamTrait;
use ringbuf::traits::Consumer;
use ringbuf::traits::Producer;

pub struct LiveSource {
    name: String,
    _stream: cpal::Stream,
    _source_format: LiveSourceFormat,
    sample_rate: usize,
    buffer_consumer: ringbuf::HeapCons<f32>,
    shared_state: std::sync::Arc<LiveSourceSharedState>,
}

impl LiveSource {
    pub fn new(device: cpal::Device) -> Result<Self, error::LiveSourceConstructionError> {
        use cpal::traits::DeviceTrait;
        use ringbuf::traits::Split;

        let description = device.description()?;
        let name = description.name().to_string();

        let config = device.default_input_config()?;
        let sample_rate: usize = config.sample_rate() as usize;
        let channels: usize = match config.channels() {
            0 => return Err(error::LiveSourceConstructionError::InvalidChannelCount { channel_count: 0 }),
            channels => usize::from(channels),
        };
        let channels_f32: f32 = channels as f32;

        let source_format = match config.sample_format() {
            cpal::SampleFormat::F32 => LiveSourceFormat::F32,
            other => return Err(error::LiveSourceConstructionError::UnsupportedFormat(other)),
        };

        /* 1 sec of buffer capacity is plenty, make adjustable later */
        let buffer_capacity = sample_rate;
        let buffer = ringbuf::HeapRb::new(buffer_capacity);
        let (mut buffer_producer, buffer_consumer) = buffer.split();

        /* Shared state */
        let shared_state = std::sync::Arc::new(LiveSourceSharedState {
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
            source_errored: std::sync::atomic::AtomicBool::new(false),
        });

        /* Data callback function, called by the audio source and pushes samples into the ring buffer */
        let data_callback_st = shared_state.clone();
        let data_callback = move |data: &[f32], _info: &cpal::InputCallbackInfo| {
            for chunk in data.chunks_exact(channels) {
                let sample: f32 = chunk.iter().cloned().sum::<f32>() / channels_f32;
                if let Err(_) = buffer_producer.try_push(sample) {
                    let order = std::sync::atomic::Ordering::Relaxed;
                    data_callback_st.dropped_frames.fetch_add(1, order);
                }
            }
        };
        /* Error callback function, only set the error flag for now */
        let error_callback_st = shared_state.clone();
        let error_callback = move |_error: cpal::Error| {
            let order = std::sync::atomic::Ordering::Relaxed;
            error_callback_st.source_errored.store(true, order);
        };

        let stream = device.build_input_stream(config.into(), data_callback, error_callback, None)?;
        stream.play()?;

        Ok(Self {
            name,
            _stream: stream,
            _source_format: source_format,
            sample_rate,
            buffer_consumer,
            shared_state,
        })
    }
}

impl crate::AudioInputStream for LiveSource {
    type InputError = error::LiveSourceError;

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn next_samples(&mut self, buffer: &mut [f32]) -> Result<crate::NextSamplesResult, Self::InputError> {
        /* Check for dropped frames */
        let order = std::sync::atomic::Ordering::Relaxed;
        let dropped_frames = self.shared_state.dropped_frames.swap(0, order);
        if dropped_frames > 0 {
            tracing::warn!(
                name = self.name,
                dropped_frames,
                "Live source ({}) overrun, {dropped_frames} frames dropped",
                self.name
            );
        }

        /* The ringbuf method does all the work: read everything or until buffer is full, exactly what we want */
        let written = self.buffer_consumer.pop_slice(buffer);

        if written > 0 {
            Ok(crate::NextSamplesResult::Some(written))
        } else {
            /* 0 samples written, check source state for info */
            let order = std::sync::atomic::Ordering::Relaxed;
            let source_errored = self.shared_state.source_errored.load(order);
            if source_errored {
                /* Get the error back at some point through a channel ? */
                Err(error::LiveSourceError::SourceDied)
            } else {
                Ok(crate::NextSamplesResult::Unavailable)
            }
        }
    }
}

impl std::fmt::Debug for LiveSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Live Source: {}", self.name)
    }
}

/// All currently supported formats
pub enum LiveSourceFormat {
    F32,
}

struct LiveSourceSharedState {
    dropped_frames: std::sync::atomic::AtomicU32,
    source_errored: std::sync::atomic::AtomicBool,
}
