// Depth contours inside segmentation masks: the HUD's wireframe. Lines of
// constant depth wrap around each object's shape; an outline follows the
// mask's edge; a faint fill marks the silhouette. Drawn over the camera
// image with the grade pass's placement, premultiplied alpha.

struct Params {
    // Clip-space scale (xy) and offset (zw) of the camera image.
    placement: vec4<f32>,
    // display-referred colour (rgb), opacity (a)
    color: vec4<f32>,
    // depth units per contour, line width (px), phase (contours), highlight id
    contour: vec4<f32>,
    // fill alpha, dim factor for non-highlighted instances, linear-target flag, unused
    style: vec4<f32>,
}
var<uniform> params: Params;

// Instance ids as 0..255 / 255, and depth, both spanning the camera image.
var mask_texture: texture_2d<f32>;
var depth_texture: texture_2d<f32>;

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

fn instance_at(texel: vec2<i32>) -> u32 {
    let size = vec2<i32>(textureDimensions(mask_texture));
    let t = clamp(texel, vec2<i32>(0), size - vec2<i32>(1));
    return u32(round(textureLoad(mask_texture, t, 0).r * 255.0));
}

// Bilinear depth by hand: float textures need not be filterable.
fn depth_at(uv: vec2<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depth_texture));
    let p = uv * vec2<f32>(size) - 0.5;
    let base = vec2<i32>(floor(p));
    let f = p - floor(p);
    let hi = size - vec2<i32>(1);
    let d00 = textureLoad(depth_texture, clamp(base, vec2<i32>(0), hi), 0).r;
    let d10 = textureLoad(depth_texture, clamp(base + vec2<i32>(1, 0), vec2<i32>(0), hi), 0).r;
    let d01 = textureLoad(depth_texture, clamp(base + vec2<i32>(0, 1), vec2<i32>(0), hi), 0).r;
    let d11 = textureLoad(depth_texture, clamp(base + vec2<i32>(1, 1), vec2<i32>(0), hi), 0).r;
    return mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Derivatives first, in uniform control flow.
    let level = depth_at(in.uv) / params.contour.x + params.contour.z;
    let width = fwidth(level);

    if any(in.uv < vec2<f32>(0.0)) || any(in.uv > vec2<f32>(1.0)) {
        discard;
    }
    let mask_size = vec2<f32>(textureDimensions(mask_texture));
    let texel = vec2<i32>(in.uv * mask_size);
    let id = instance_at(texel);
    if id == 0u {
        discard;
    }

    // Distance to the nearest contour, in pixels.
    let distance = abs(fract(level + 0.5) - 0.5) / max(width, 1e-6);
    let contour = 1.0 - smoothstep(0.5 * params.contour.y - 0.5, 0.5 * params.contour.y + 0.5, distance);
    // Outline: a neighbouring mask texel belongs to something else.
    let edge = id != instance_at(texel + vec2<i32>(1, 0)) || id != instance_at(texel - vec2<i32>(1, 0))
        || id != instance_at(texel + vec2<i32>(0, 1)) || id != instance_at(texel - vec2<i32>(0, 1));
    var alpha = max(max(contour, select(0.0, 1.0, edge)), params.style.x);
    let highlight = u32(params.contour.w);
    if highlight != 0u && id != highlight {
        alpha *= params.style.y;
    }
    alpha *= params.color.a;

    var rgb = params.color.rgb;
    if params.style.z > 0.5 {
        rgb = srgb_to_linear(rgb);
    }
    return vec4<f32>(rgb * alpha, alpha);
}
