//! A camera frame, in the layout the passthrough camera produces.
//!
//! Quest's passthrough cameras arrive through Camera2 as YUV420 at
//! 1280×960. Keeping that layout end to end is deliberate:
//!
//! * the colour grade only needs luminance, which is the Y plane verbatim;
//! * model preprocessing (resize, normalise, YUV→RGB) belongs on the GPU,
//!   next to the network, rather than in a per-pixel CPU loop;
//! * a desktop source converts *into* this layout once, so both paths
//!   share everything downstream.

/// Which passthrough camera a frame came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Eye {
    Left,
    Right,
}

/// Planar YUV 4:2:0 (I420), BT.601 full range, tightly packed.
#[derive(Clone, Debug)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// `width * height` luma, then two `(width/2) * (height/2)` chroma
    /// planes, U then V.
    pub data: Vec<u8>,
    /// Capture time on the camera clock, nanoseconds. Zero when unknown.
    /// Needed to pair a frame with the head pose it was seen from.
    pub timestamp_ns: i64,
    pub eye: Eye,
}

impl Frame {
    /// An all-black frame.
    pub fn new(width: u32, height: u32, eye: Eye) -> Self {
        assert!(
            width.is_multiple_of(2) && height.is_multiple_of(2),
            "4:2:0 needs even dimensions"
        );
        let luma = (width * height) as usize;
        let mut data = vec![0u8; luma + luma / 2];
        data[luma..].fill(128);
        Self {
            width,
            height,
            data,
            timestamp_ns: 0,
            eye,
        }
    }

    fn luma_len(&self) -> usize {
        (self.width * self.height) as usize
    }

    pub fn luma(&self) -> &[u8] {
        &self.data[..self.luma_len()]
    }

    pub fn luma_mut(&mut self) -> &mut [u8] {
        let n = self.luma_len();
        &mut self.data[..n]
    }

    /// The U and V planes.
    pub fn chroma(&self) -> (&[u8], &[u8]) {
        let (_, uv) = self.data.split_at(self.luma_len());
        uv.split_at(uv.len() / 2)
    }

    /// Convert interleaved RGB8 into a frame, for desktop sources and tests.
    pub fn from_rgb8(width: u32, height: u32, rgb: &[u8], eye: Eye) -> Self {
        assert_eq!(rgb.len(), (width * height * 3) as usize);
        let (w, h) = (width as usize, height as usize);
        let mut frame = Self::new(width & !1, height & !1, eye);
        let (fw, fh) = (frame.width as usize, frame.height as usize);
        let pixel = |x: usize, y: usize| {
            let i = (y.min(h - 1) * w + x.min(w - 1)) * 3;
            [rgb[i] as f32, rgb[i + 1] as f32, rgb[i + 2] as f32]
        };
        {
            let luma = frame.luma_mut();
            for y in 0..fh {
                for x in 0..fw {
                    let [r, g, b] = pixel(x, y);
                    luma[y * fw + x] = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                }
            }
        }
        let (cw, ch) = (fw / 2, fh / 2);
        let base = fw * fh;
        for cy in 0..ch {
            for cx in 0..cw {
                let mut acc = [0.0f32; 3];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let p = pixel(cx * 2 + dx, cy * 2 + dy);
                    acc.iter_mut().zip(p).for_each(|(a, v)| *a += v * 0.25);
                }
                let [r, g, b] = acc;
                let u = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
                let v = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;
                frame.data[base + cy * cw + cx] = u.round().clamp(0.0, 255.0) as u8;
                frame.data[base + cw * ch + cy * cw + cx] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_rgb_has_neutral_chroma() {
        let rgb = vec![100u8; 4 * 2 * 3];
        let frame = Frame::from_rgb8(4, 2, &rgb, Eye::Left);
        assert!(frame.luma().iter().all(|&y| y == 100));
        let (u, v) = frame.chroma();
        assert_eq!(u.len(), 2);
        assert!(u.iter().chain(v).all(|&c| c == 128));
    }

    #[test]
    fn red_has_positive_v() {
        let rgb = [255u8, 0, 0].repeat(4);
        let frame = Frame::from_rgb8(2, 2, &rgb, Eye::Left);
        let (u, v) = frame.chroma();
        assert!(v[0] > 200 && u[0] < 128);
    }
}
