pub struct AvailableSource {
    device: cpal::Device,
    description: cpal::DeviceDescription,
    live_capture: AvailableSourceLiveCapture,
}

impl AvailableSource {
    pub fn list() -> Vec<Self> {
        use cpal::traits::HostTrait;
        let hosts: Vec<cpal::Host> = cpal::available_hosts()
            .into_iter()
            .filter_map(|host_id| match cpal::host_from_id(host_id) {
                Ok(host) => Some(host),
                Err(e) => {
                    tracing::warn!("Failed to load host {host_id}, skipping: {e}");
                    None
                }
            })
            .collect();

        let devices: Vec<cpal::Device> = hosts
            .iter()
            .filter_map(|host| match host.input_devices() {
                Ok(devices) => Some(devices),
                Err(e) => {
                    tracing::warn!("Failed to list devices for host {}, skipping: {e}", host.id());
                    None
                }
            })
            .flatten()
            .collect();

        devices.into_iter().filter_map(Self::from_device).collect()
    }

    pub fn to_live_source(self) -> Result<crate::LiveSource, crate::live_source::error::LiveSourceConstructionError> {
        crate::LiveSource::new(self.device)
    }

    fn from_device(device: cpal::Device) -> Option<Self> {
        use cpal::traits::DeviceTrait;

        let device_description = match device.description() {
            Ok(desc) => desc,
            Err(e) => {
                tracing::warn!("Failed to get device description, skipping: {e}");
                return None;
            }
        };
        Some(Self {
            device,
            description: device_description,
            live_capture: AvailableSourceLiveCapture::NotStarted,
        })
    }

    pub fn description(&self) -> &cpal::DeviceDescription {
        &self.description
    }

    pub fn live_capture(&self) -> &AvailableSourceLiveCapture {
        &self.live_capture
    }

    pub fn start_capture(&mut self) {
        /* Guard to not restart live captures */
        match self.live_capture {
            AvailableSourceLiveCapture::Live { .. } => return,
            _ => {}
        }

        /* Null devices dump a LOT of 0 and cause overrun */
        const BLACKLIST: &[&'static str] = &["Discard all samples (playback) or generate zero samples (capture)"];
        if BLACKLIST.contains(&self.description.name()) {
            self.live_capture = AvailableSourceLiveCapture::Blacklisted;
            return;
        }

        match crate::LiveSource::new(self.device.clone()) {
            Ok(live_source) => {
                self.live_capture = AvailableSourceLiveCapture::Live {
                    live_source,
                    activity: 0.0,
                }
            }
            Err(e) => self.live_capture = AvailableSourceLiveCapture::Errored { error: e.to_string() },
        }
    }

    pub fn refresh_live_capture(&mut self) {
        match &mut self.live_capture {
            AvailableSourceLiveCapture::Live { live_source, activity } => {
                /* Fetch the samples from the live source and set the activity */
                let mut sample_buffer = [0.0f32; 2048];
                let mut sample_sum: f32 = 0.0;
                let mut total_sample_count: usize = 0;

                loop {
                    use crate::AudioInputStream;
                    match live_source.next_samples(&mut sample_buffer) {
                        Ok(crate::NextSamplesResult::Some(sample_count)) => {
                            sample_sum += sample_buffer[..sample_count].iter().map(|s| s.abs()).sum::<f32>();
                            total_sample_count += sample_count;
                        }
                        Ok(crate::NextSamplesResult::Unavailable) => break,
                        Ok(crate::NextSamplesResult::EndOfInput) => {
                            /* Switching to error would be better */
                            *activity = 0.0;
                            return;
                        }
                        Err(_) => {
                            /* Switching to error would be better */
                            *activity = 0.0;
                            return;
                        }
                    }
                }

                *activity = if total_sample_count > 0 {
                    sample_sum / (total_sample_count as f32)
                } else {
                    0.0
                };
            }
            _ => {}
        }
    }
}

pub enum AvailableSourceLiveCapture {
    NotStarted,
    Blacklisted,
    Errored { error: String },
    Live { live_source: crate::LiveSource, activity: f32 },
}
