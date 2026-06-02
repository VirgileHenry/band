//! Decode an audio file to mono f32 samples + report its sample rate.

use std::fs::File;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub fn decode_to_mono(path: &std::path::Path) -> std::io::Result<DecodedAudio> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no audio track"))?
        .clone();

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no sample rate"))?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let mut mono = Vec::<f32>::new();
    let mut sbuf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => {
                log::warn!("packet error in {:?}: {}", path, e);
                break;
            }
        };

        match decoder.decode(&packet) {
            Ok(decoded) => {
                if sbuf.is_none() {
                    let spec = *decoded.spec();
                    let dur = decoded.capacity() as u64;
                    sbuf = Some(SampleBuffer::<f32>::new(dur, spec));
                }
                let sb = sbuf.as_mut().unwrap();
                sb.copy_interleaved_ref(decoded);

                // Downmix interleaved → mono by averaging channels.
                for frame in sb.samples().chunks(channels) {
                    let sum: f32 = frame.iter().copied().sum();
                    mono.push(sum / channels as f32);
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => {
                log::warn!("decode error in {:?}: {}", path, e);
                break;
            }
        }
    }

    Ok(DecodedAudio {
        samples: mono,
        sample_rate,
    })
}
