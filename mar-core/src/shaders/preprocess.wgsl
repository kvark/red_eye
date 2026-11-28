// Camera frame (I420) to a detector's input: fit (letterbox or stretch)
// with bilinear resampling, BT.601 YUV to RGB or BGR, scale the 0..255
// values, and write plain CHW or YOLOX's Focus space-to-depth layout.
// Mirrors `preprocess::preprocess` in src/perception/preprocess.rs;
// tests/gpu.rs holds the two to agreement.

struct Params {
    // frame width, frame height, resized width, resized height
    frame: vec4<u32>,
    // input width, input height, input width / 2, input height / 2
    input: vec4<u32>,
    // frame width / resized width, frame height / resized height, pad, value scale
    scale: vec4<f32>,
    // BGR order, Focus layout, unused, unused
    flags: vec4<u32>,
}
var<uniform> params: Params;

// I420 bytes, four to a word: Y plane, then U, then V.
var<storage, read> frame: array<u32>;
var<storage, read_write> output: array<f32>;

fn byte_at(index: u32) -> f32 {
    let word = frame[index >> 2u];
    return f32((word >> ((index & 3u) * 8u)) & 0xffu);
}

fn sample(base: u32, w: u32, h: u32, x_in: f32, y_in: f32) -> f32 {
    let x = clamp(x_in, 0.0, f32(w - 1u));
    let y = clamp(y_in, 0.0, f32(h - 1u));
    let x0 = u32(x);
    let y0 = u32(y);
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let a = byte_at(base + y0 * w + x0);
    let b = byte_at(base + y0 * w + x1);
    let c = byte_at(base + y1 * w + x0);
    let d = byte_at(base + y1 * w + x1);
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    return top + (bottom - top) * fy;
}

fn input_index(x: u32, y: u32, c: u32) -> u32 {
    if params.flags.y != 0u {
        let phase = (x % 2u) * 2u + (y % 2u);
        return ((phase * 3u + c) * params.input.w + y / 2u) * params.input.z + x / 2u;
    }
    return (c * params.input.y + y) * params.input.x + x;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if x >= params.input.x || y >= params.input.y {
        return;
    }
    var rgb = vec3<f32>(params.scale.z);
    if x < params.frame.z && y < params.frame.w {
        let fw = params.frame.x;
        let fh = params.frame.y;
        let cw = fw / 2u;
        let ch = fh / 2u;
        let src_x = (f32(x) + 0.5) * params.scale.x - 0.5;
        let src_y = (f32(y) + 0.5) * params.scale.y - 0.5;
        let l = sample(0u, fw, fh, src_x, src_y);
        let cx = (src_x - 0.5) * 0.5;
        let cy = (src_y - 0.5) * 0.5;
        let u = sample(fw * fh, cw, ch, cx, cy) - 128.0;
        let v = sample(fw * fh + cw * ch, cw, ch, cx, cy) - 128.0;
        rgb = clamp(
            vec3<f32>(l + 1.402 * v, l - 0.344136 * u - 0.714136 * v, l + 1.772 * u),
            vec3<f32>(0.0),
            vec3<f32>(255.0),
        ) * params.scale.w;
        if params.flags.x != 0u {
            rgb = rgb.zyx;
        }
    }
    output[input_index(x, y, 0u)] = rgb.x;
    output[input_index(x, y, 1u)] = rgb.y;
    output[input_index(x, y, 2u)] = rgb.z;
}
