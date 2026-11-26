// The HUD canvas over whatever is already in the target: a placed quad
// sampling the canvas texture, premultiplied alpha. The canvas is sRGB, so
// sampling yields linear values; a target that does not encode sRGB itself
// gets them encoded here, as the grade pass does.

struct Params {
    // Clip-space scale (xy) and offset (zw), as the grade pass places quads.
    placement: vec4<f32>,
    // linear-target flag, opacity, unused, unused
    flags: vec4<f32>,
}
var<uniform> params: Params;

var hud_texture: texture_2d<f32>;
var hud_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let local = vec2<f32>(4.0 * f32(vi & 1u) - 1.0, 4.0 * f32(vi >> 1u) - 1.0);
    var out: VertexOutput;
    out.position = vec4<f32>(local * params.placement.xy + params.placement.zw, 0.0, 1.0);
    out.uv = vec2<f32>(0.5 * local.x + 0.5, 0.5 - 0.5 * local.y);
    return out;
}

fn srgb_from_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample first: textureSample must stay in uniform control flow.
    var color = textureSample(hud_texture, hud_sampler, in.uv) * params.flags.y;
    if any(in.uv < vec2<f32>(0.0)) || any(in.uv > vec2<f32>(1.0)) {
        discard;
    }
    if params.flags.x < 0.5 && color.a > 0.0 {
        // Encode the straight color, then premultiply again.
        color = vec4<f32>(srgb_from_linear(color.rgb / color.a) * color.a, color.a);
    }
    return color;
}
