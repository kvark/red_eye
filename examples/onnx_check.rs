//! Run an ONNX export through Meganeura's importer and compare it with
//! onnxruntime, to evaluate a candidate model before committing to it.
//!
//! ```text
//! python tools/onnx_reference.py model.onnx ref/candidate 1,3,640,640 [image]
//! cargo run --release --example onnx_check -- ref/candidate
//! ```
//!
//! Reports an import failure (the first unsupported op or shape), or the
//! worst error of every output against the reference and the time per run.

use std::path::Path;
use std::time::Instant;

use meganeura::load::onnx::load_onnx;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytemuck::pod_collect_to_vec(&bytes)
}

fn main() {
    env_logger::init();
    let dir = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: onnx_check <reference-dir>"),
    );
    let start = Instant::now();
    let model = match load_onnx(&dir.join("model.onnx")) {
        Ok(model) => model,
        Err(e) => {
            println!("import failed: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "imported {} nodes, {} weights in {:.1} s",
        model.graph.nodes().len(),
        model.weights.len(),
        start.elapsed().as_secs_f32()
    );
    // Where the graph's size comes from: the most frequent node kinds.
    let mut kinds = std::collections::HashMap::<String, usize>::new();
    for node in model.graph.nodes() {
        let name = format!("{:?}", node.op);
        let kind = name
            .split([' ', '{', '('])
            .next()
            .unwrap_or("?")
            .to_string();
        *kinds.entry(kind).or_default() += 1;
    }
    let mut kinds: Vec<_> = kinds.into_iter().collect();
    kinds.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    let top: Vec<_> = kinds
        .iter()
        .take(8)
        .map(|(k, n)| format!("{k} {n}"))
        .collect();
    println!("most frequent nodes: {}", top.join(", "));

    let start = Instant::now();
    let (mut session, _) = meganeura::build(
        &model.graph,
        meganeura::SessionConfig {
            mode: meganeura::Mode::Inference,
            ..Default::default()
        },
    );
    println!(
        "compiled {} dispatches in {:.1} s",
        session.plan().dispatches.len(),
        start.elapsed().as_secs_f32()
    );
    for (name, _) in session.plan().param_buffers.clone() {
        match model.weights.get(&name) {
            Some(data) => session.set_parameter(&name, data),
            None => println!("warning: no weight for parameter {name}"),
        }
    }
    let input_name = session.plan().input_buffers[0].0.clone();
    session.set_input(&input_name, &read_f32(&dir.join("input.bin")));

    let mut timings = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        session.step();
        session.wait();
        timings.push(start.elapsed().as_secs_f64() * 1e3);
    }
    println!("run: {timings:.1?} ms (host GPU, not the Quest)");

    let mut index = 0;
    let mut all_close = true;
    while let Ok(bytes) = std::fs::metadata(dir.join(format!("output_{index}.bin"))) {
        let expected = read_f32(&dir.join(format!("output_{index}.bin")));
        assert_eq!(bytes.len() as usize, expected.len() * 4);
        let mut got = vec![0.0f32; expected.len()];
        session.read_output_by_index(index, &mut got);
        let (mut worst, mut at) = (0.0f32, 0);
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            let error = (g - e).abs() / (1.0 + e.abs());
            // NaN must count as the worst error, not slip past it.
            if error > worst || error.is_nan() {
                (worst, at) = (error, i);
            }
        }
        println!(
            "output {index}: {} values, worst relative error {worst:.2e} at {at} \
             (meganeura {}, onnxruntime {})",
            expected.len(),
            got[at],
            expected[at]
        );
        all_close &= worst < 1e-3;
        index += 1;
    }
    if !all_close {
        std::process::exit(2);
    }
}
