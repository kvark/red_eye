//! The headset's microphone through AAudio, for voice commands.
//!
//! Opened only while the app has focus, and read on the voice thread; audio
//! goes straight to the voice activity detector and is never stored.
//! `android.permission.RECORD_AUDIO` must be granted at runtime.

use std::ffi::{CStr, c_char, c_void};

use red_eye::voice::vad::{self, SAMPLE_RATE};

#[repr(C)]
struct AAudioStreamBuilder([u8; 0]);
#[repr(C)]
struct AAudioStream([u8; 0]);

const AAUDIO_OK: i32 = 0;
const AAUDIO_DIRECTION_INPUT: i32 = 1;
const AAUDIO_FORMAT_PCM_FLOAT: i32 = 2;
const AAUDIO_SHARING_MODE_SHARED: i32 = 1;
const AAUDIO_PERFORMANCE_MODE_POWER_SAVING: i32 = 11;
/// Tuned for recognition: no echo cancellation or noise suppression that
/// would smear the speech.
const AAUDIO_INPUT_PRESET_VOICE_RECOGNITION: i32 = 6;

#[link(name = "aaudio")]
unsafe extern "C" {
    fn AAudio_createStreamBuilder(builder: *mut *mut AAudioStreamBuilder) -> i32;
    fn AAudioStreamBuilder_setDirection(builder: *mut AAudioStreamBuilder, direction: i32);
    fn AAudioStreamBuilder_setSampleRate(builder: *mut AAudioStreamBuilder, rate: i32);
    fn AAudioStreamBuilder_setChannelCount(builder: *mut AAudioStreamBuilder, count: i32);
    fn AAudioStreamBuilder_setFormat(builder: *mut AAudioStreamBuilder, format: i32);
    fn AAudioStreamBuilder_setSharingMode(builder: *mut AAudioStreamBuilder, mode: i32);
    fn AAudioStreamBuilder_setPerformanceMode(builder: *mut AAudioStreamBuilder, mode: i32);
    fn AAudioStreamBuilder_setInputPreset(builder: *mut AAudioStreamBuilder, preset: i32);
    fn AAudioStreamBuilder_openStream(
        builder: *mut AAudioStreamBuilder,
        stream: *mut *mut AAudioStream,
    ) -> i32;
    fn AAudioStreamBuilder_delete(builder: *mut AAudioStreamBuilder) -> i32;
    fn AAudioStream_requestStart(stream: *mut AAudioStream) -> i32;
    fn AAudioStream_requestStop(stream: *mut AAudioStream) -> i32;
    fn AAudioStream_read(
        stream: *mut AAudioStream,
        buffer: *mut c_void,
        frames: i32,
        timeout_nanos: i64,
    ) -> i32;
    fn AAudioStream_getSampleRate(stream: *mut AAudioStream) -> i32;
    fn AAudioStream_getChannelCount(stream: *mut AAudioStream) -> i32;
    fn AAudioStream_getFormat(stream: *mut AAudioStream) -> i32;
    fn AAudioStream_close(stream: *mut AAudioStream) -> i32;
    fn AAudio_convertResultToText(result: i32) -> *const c_char;
}

fn describe(result: i32) -> String {
    let text = unsafe { AAudio_convertResultToText(result) };
    if text.is_null() {
        return format!("AAudio error {result}");
    }
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}

/// An open, started input stream.
pub struct Microphone {
    stream: *mut AAudioStream,
    rate: u32,
    channels: usize,
    buffer: Vec<f32>,
}

// The stream is only touched by the thread that owns this.
unsafe impl Send for Microphone {}

impl Microphone {
    /// Open and start the microphone, asking for 16 kHz mono float. The
    /// device may give another rate or channel count; `read` converts.
    pub fn open() -> Result<Self, String> {
        unsafe {
            let mut builder = std::ptr::null_mut();
            let result = AAudio_createStreamBuilder(&mut builder);
            if result != AAUDIO_OK {
                return Err(describe(result));
            }
            AAudioStreamBuilder_setDirection(builder, AAUDIO_DIRECTION_INPUT);
            AAudioStreamBuilder_setSampleRate(builder, SAMPLE_RATE as i32);
            AAudioStreamBuilder_setChannelCount(builder, 1);
            AAudioStreamBuilder_setFormat(builder, AAUDIO_FORMAT_PCM_FLOAT);
            AAudioStreamBuilder_setSharingMode(builder, AAUDIO_SHARING_MODE_SHARED);
            AAudioStreamBuilder_setPerformanceMode(builder, AAUDIO_PERFORMANCE_MODE_POWER_SAVING);
            AAudioStreamBuilder_setInputPreset(builder, AAUDIO_INPUT_PRESET_VOICE_RECOGNITION);
            let mut stream = std::ptr::null_mut();
            let result = AAudioStreamBuilder_openStream(builder, &mut stream);
            AAudioStreamBuilder_delete(builder);
            if result != AAUDIO_OK || stream.is_null() {
                return Err(format!(
                    "{}; is RECORD_AUDIO granted? adb shell pm grant rust.red_eye_xr \
                     android.permission.RECORD_AUDIO",
                    describe(result)
                ));
            }
            let format = AAudioStream_getFormat(stream);
            if format != AAUDIO_FORMAT_PCM_FLOAT {
                AAudioStream_close(stream);
                return Err(format!("the microphone gives format {format}, not float"));
            }
            let result = AAudioStream_requestStart(stream);
            if result != AAUDIO_OK {
                AAudioStream_close(stream);
                return Err(describe(result));
            }
            let microphone = Self {
                stream,
                rate: AAudioStream_getSampleRate(stream) as u32,
                channels: AAudioStream_getChannelCount(stream).max(1) as usize,
                buffer: Vec::new(),
            };
            log::info!(
                "microphone open: {} Hz, {} channel(s)",
                microphone.rate,
                microphone.channels
            );
            Ok(microphone)
        }
    }

    /// Wait up to `timeout` for about 20 ms of audio, and return it as
    /// 16 kHz mono.
    pub fn read(&mut self, timeout: std::time::Duration) -> Result<Vec<f32>, String> {
        let frames = (self.rate / 50) as usize;
        self.buffer.resize(frames * self.channels, 0.0);
        let read = unsafe {
            AAudioStream_read(
                self.stream,
                self.buffer.as_mut_ptr() as *mut c_void,
                frames as i32,
                timeout.as_nanos() as i64,
            )
        };
        if read < 0 {
            return Err(describe(read));
        }
        let mono: Vec<f32> = self.buffer[..read as usize * self.channels]
            .chunks(self.channels)
            .map(|frame| frame.iter().sum::<f32>() / self.channels as f32)
            .collect();
        Ok(vad::resample(&mono, self.rate, SAMPLE_RATE))
    }
}

impl Drop for Microphone {
    fn drop(&mut self) {
        unsafe {
            AAudioStream_requestStop(self.stream);
            AAudioStream_close(self.stream);
        }
        log::info!("microphone closed");
    }
}
