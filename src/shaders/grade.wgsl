// The red colour grade, per pixel. Mirrors `Grade::map` in src/grade.rs;
// tests/gpu.rs holds the two to agreement with every effect disabled.

struct GradeParams {
    // black, white, gamma, lift
    levels: vec4<f32>,
    // green scale, green power, blue scale, blue power
    channels: vec4<f32>,
    // hot colour (rgb), clip (a)
    hot: vec4<f32>,
    // grain, vignette, scanline depth, scanline period (px)
    effects: vec4<f32>,
    // time (s), linear-target flag
    frame: vec4<f32>,
}
var<uniform> params: GradeParams;

// Where the camera image lands in clip space: xy = scale, zw = offset.
// Lets an eye place a 4:3 camera frame inside an asymmetric frustum.
struct Placement {
    transform: vec4<f32>,
}
var<uniform> placement: Placement;

var luma_texture: texture_2d<f32>;
var luma_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle covering the placed quad: a full-screen triangle in the
// placement's local space, so out-of-image pixels can be discarded in the
// fragment stage instead of being stretched.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let local = vec2<f32>(4.0 * f32(vi & 1u) - 1.0, 4.0 * f32(vi >> 1u) - 1.0);
    var out: VertexOutput;
    out.position = vec4<f32>(local * placement.transform.xy + placement.transform.zw, 0.0, 1.0);
    out.uv = vec2<f32>(0.5 * local.x + 0.5, 0.5 - 0.5 * local.y);
    return out;
}

fn curve(luma: f32) -> f32 {
    let range = max(params.levels.y - params.levels.x, 1e-4);
    let y = pow(clamp((luma - params.levels.x) / range, 0.0, 1.0), params.levels.z);
    return params.levels.w + (1.0 - params.levels.w) * y;
}

fn tone_map(luma: f32) -> vec3<f32> {
    let y = curve(luma);
    let base = vec3<f32>(
        y,
        params.channels.x * pow(y, params.channels.y),
        params.channels.z * pow(y, params.channels.w),
    );
    let s = smoothstep(params.hot.a, 1.0, y);
    return mix(base, params.hot.rgb, s);
}

// Cheap integer hash for grain; quality is irrelevant at this amplitude.
fn hash(p: vec3<u32>) -> f32 {
    var h = p.x * 1664525u + p.y * 1013904223u + p.z * 2654435761u;
    h = (h ^ (h >> 16u)) * 2246822519u;
    h = h ^ (h >> 13u);
    return f32(h & 0xffffu) / 65535.0;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if any(in.uv < vec2<f32>(0.0)) || any(in.uv > vec2<f32>(1.0)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    var luma = textureSampleLevel(luma_texture, luma_sampler, in.uv, 0.0).r;

    let pixel = vec2<u32>(in.position.xy);
    let grain = params.effects.x;
    if grain > 0.0 {
        // Film grain at 24 Hz, before grading so the curve shapes it.
        let tick = u32(params.frame.x * 24.0);
        luma += (hash(vec3<u32>(pixel, tick)) - 0.5) * grain;
    }
    var color = tone_map(luma);

    let vignette = params.effects.y;
    if vignette > 0.0 {
        let r = length(in.uv - vec2<f32>(0.5)) * 1.41421356;
        color *= 1.0 - vignette * smoothstep(0.45, 0.95, r);
    }
    let scan_depth = params.effects.z;
    if scan_depth > 0.0 {
        let phase = in.position.y / params.effects.w;
        color *= 1.0 - scan_depth * (0.5 + 0.5 * sin(phase * 6.28318531));
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    if params.frame.y > 0.5 {
        color = srgb_to_linear(color);
    }
    return vec4<f32>(color, 1.0);
}
