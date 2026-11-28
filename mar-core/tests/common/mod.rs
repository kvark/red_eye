//! Shared by the tests that read models and references from `ref/`.

/// Run from where `ref/` is. Cargo starts integration tests in the
/// package's directory, while `ref/` sits at the workspace root; on a
/// device the tests run beside a `ref/` of their own and stay put. Every
/// test calls this first, and all move to the same place, so the
/// process-wide change is safe across test threads.
pub fn at_references() {
    if std::path::Path::new("ref").exists() {
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    std::env::set_current_dir(&root).unwrap_or_else(|e| panic!("{}: {e}", root.display()));
}
