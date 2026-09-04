// OPENVINO EP INTEGRATION TESTS — see docs/testing/openvino-test-plan.md Layer 2.
//
// These tests exercise the REAL OpenVINO EP against a real libonnxruntime.so
// (built with --use_openvino) and a system OpenVINO runtime. They are #[ignore]d
// by default because they need a runtime environment:
//
//   ORT_DYLIB_PATH=<libonnxruntime.so> \
//   LD_LIBRARY_PATH=<openvino libs> \
//   cargo test --features openvino --test openvino_integration -- --ignored
//
// I1/I3 run wherever a CPU OpenVINO device exists (E1+). I2 additionally needs
// an Intel GPU + the legacy1 driver stack (E2) and self-skips otherwise.
#![cfg(feature = "openvino")]

use ort::session::Session;
use ort::value::TensorRef;

fn identity_model() -> Vec<u8> {
    std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/identity.onnx")).unwrap()
}

fn cache_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("xs-ov-cache-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build_session(device: &str, cache: &std::path::Path) -> Session {
    Session::builder()
        .unwrap()
        .with_execution_providers([
            ort::ep::OpenVINO::default()
                .with_device_type(device)
                .with_cache_dir(cache.to_str().unwrap())
                .build(),
        ])
        .unwrap()
        .commit_from_memory(identity_model().as_slice())
        .unwrap()
}

fn infer_identity(sess: &mut Session) {
    // SAME TENSOR-CONSTRUCTION STYLE AS src/ml/detect/rfdetr.rs
    let expect: Vec<f32> = (0..1 * 3 * 8 * 8).map(|i| i as f32 / 64.0 - 0.5).collect();
    let input_tensor = ort::value::Tensor::from_array(([1usize, 3, 8, 8], expect.clone())).unwrap();
    let outputs = sess.run(ort::inputs![input_tensor]).unwrap();
    let (_, out_slice) = outputs[0].try_extract_tensor::<f32>().unwrap();
    for (a, b) in expect.iter().zip(out_slice.iter()) {
        assert!((a - b).abs() < 1e-6, "identity mismatch: {a} vs {b}");
    }
}

// I1 — EP loads, CPU device session commits, inference is numerically exact.
#[test]
#[ignore = "requires ORT_DYLIB_PATH + OpenVINO runtime"]
fn i1_openvino_ep_smoke_cpu_device() {
    let mut sess = build_session("CPU", &cache_dir("i1"));
    infer_identity(&mut sess);
}

// I2 — GPU device session builds and infers (UHD 630 / legacy driver stack).
#[test]
#[ignore = "requires ORT_DYLIB_PATH + OpenVINO runtime + Intel GPU (E2)"]
fn i2_openvino_ep_gpu_session_builds() {
    let mut sess = match std::panic::catch_unwind(|| build_session("GPU", &cache_dir("i2"))) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("SKIP: OpenVINO GPU device unavailable in this environment");
            return;
        }
    };
    infer_identity(&mut sess);
}

// I3 — model caching writes blobs (startup-time matters on Gen9: first compile
// of RF-DETR takes ~15s without cache).
#[test]
#[ignore = "requires ORT_DYLIB_PATH + OpenVINO runtime"]
fn i3_model_cache_dir_populated() {
    let dir = cache_dir("i3");
    build_session("CPU", &dir);
    let populated = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.metadata().map(|m| m.len() > 0).unwrap_or(false));
    assert!(populated, "cache dir {:?} has no blobs", dir);
}
