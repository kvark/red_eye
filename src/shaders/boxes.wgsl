// Box outlines over the camera image: four thin quads per box, placed with
// the same image-to-clip transform the grade pass uses, so a box drawn in
// normalised image coordinates lands on what the camera saw.

struct Params {
    // Same as the grade pass: clip-space scale (xy) and offset (zw).
    placement: vec4<f32>,
    // viewport width, viewport height, line thickness (px), unused
    viewport: vec4<f32>,
}
var<uniform> params: Params;

struct Box {
    // min x, min y, max x, max y in normalised image coordinates, y down
    rect: vec4<f32>,
    color: vec4<f32>,
}
var<storage, read> boxes: array<Box>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

fn to_clip(uv: vec2<f32>) -> vec2<f32> {
    let local = vec2<f32>(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
    return local * params.placement.xy + params.placement.zw;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertexOutput {
    let b = boxes[ii / 4u];
    let edge = ii % 4u;
    let lo = to_clip(b.rect.xw); // left, bottom
    let hi = to_clip(b.rect.zy); // right, top
    // Half the line thickness in clip units: a pixel is 2 / target wide.
    let half = vec2<f32>(params.viewport.z) / params.viewport.xy;
    var a: vec2<f32>;
    var c: vec2<f32>;
    switch edge {
        case 0u: { // top
            a = vec2<f32>(lo.x - half.x, hi.y - half.y);
            c = vec2<f32>(hi.x + half.x, hi.y + half.y);
        }
        case 1u: { // bottom
            a = vec2<f32>(lo.x - half.x, lo.y - half.y);
            c = vec2<f32>(hi.x + half.x, lo.y + half.y);
        }
        case 2u: { // left
            a = vec2<f32>(lo.x - half.x, lo.y - half.y);
            c = vec2<f32>(lo.x + half.x, hi.y + half.y);
        }
        default: { // right
            a = vec2<f32>(hi.x - half.x, lo.y - half.y);
            c = vec2<f32>(hi.x + half.x, hi.y + half.y);
        }
    }
    // Two triangles: corners (0,0) (1,0) (0,1) / (0,1) (1,0) (1,1).
    let corner_x = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    let corner_y = array<f32, 6>(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    let t = vec2<f32>(corner_x[vi], corner_y[vi]);
    var out: VertexOutput;
    out.position = vec4<f32>(mix(a, c, t), 0.0, 1.0);
    out.color = b.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
