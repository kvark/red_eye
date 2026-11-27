// The headset's depth of the environment, resampled to a fixed grid per
// eye into a buffer the CPU reads back. The texture's size belongs to the
// runtime and is only known here; the grid keeps the CPU side independent
// of it. Nearest samples: averaging across an object's edge would invent a
// surface between it and the background.

struct Params {
    // grid width, grid height, layers, unused
    size: vec4<u32>,
}
var<uniform> params: Params;

var depth_texture: texture_depth_2d_array;
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= params.size.xyz) {
        return;
    }
    let dims = textureDimensions(depth_texture);
    let at = (vec2<f32>(id.xy) + 0.5) * vec2<f32>(dims) / vec2<f32>(params.size.xy);
    let texel = min(vec2<u32>(at), dims - vec2<u32>(1u));
    let depth = textureLoad(depth_texture, texel, id.z, 0);
    output[(id.z * params.size.y + id.y) * params.size.x + id.x] = depth;
}
