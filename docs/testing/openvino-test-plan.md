# OpenVINO Execution Provider — Test Plan (Phase 0)

Goal: add an `openvino` execution provider path to the ML engine so Intel
iGPUs (Gen9.5 via OpenVINO 2024.6 legacy runtime) can run the detector,
inpainter, and OCR-det models. Phase 0 scope is device plumbing + validation;
settings-UI plumbing and upstream PR polish are later phases.

Design summary
- `Cargo.toml`: new feature `openvino = ["ort/openvino"]`.
- `src/ml/device.rs`: extract pure decision core `resolve_device_plan(DeviceInputs)`
  from `probe_hardware()`; add OpenVINO branches (explicit `MT_DEVICE=openvino`,
  and auto-detect when no dedicated GPU is present and the runtime is usable).
- Per-model override: PP-OCR **rec** stays on CPU under OpenVINO (2024.6 ONNX
  frontend rejects that model — verified empirically); det/rfdetr/lama use OV.
- Runtime latching mirrors CUDA: `OPENVINO_RUNTIME_FAILED` flips on first failed
  session creation; subsequent `probe_hardware()` calls report CPU.

## Environments
- **E0** dev/CI container: no libopenvino. Runs Layer-1 unit tests only
  (openvino cases compile-gated OFF when the feature is absent).
- **E1** validation container: OpenVINO 2024.6 runtime present (CPU device only).
  Runs Layer-1 + I1, I3.
- **E2** in-cluster iGPU pod (worker-01: UHD 630 + legacy1 driver stack + OV
  2024.6). Runs Layer-1 + I1–I3 + Layer-3 validation.

## Layer 1 — Unit tests (`cargo test`, no GPU, no libopenvino)
Target: `resolve_device_plan` + `effective_providers_for_model` (pure fns).

| id  | test                                             | input                                                       | expect |
|-----|--------------------------------------------------|-------------------------------------------------------------|--------|
| U1  | resolves_cpu_when_override_cpu                   | override=`cpu`                                              | `[CPU]`, label "CPU Multi-threaded" |
| U2  | resolves_openvino_when_requested_and_usable      | override=`openvino`, ov_usable=true                          | `[OpenVINO, CPU]`, label contains "OpenVINO" |
| U3  | openvino_alias_ov_accepted                       | override=`ov`                                                | same as U2 |
| U4  | falls_back_to_cpu_when_openvino_runtime_failed   | override=`openvino`, ov_usable=false                         | `[CPU]` |
| U5  | explicit_openvino_ignored_without_feature        | (no `openvino` cfg) override=`openvino`                      | `[CPU]` |
| U6  | auto_prefers_openvino_when_no_dgpu               | override=``, ov_usable=true, no dgpu                         | `[OpenVINO, CPU]` |
| U7  | auto_prefers_cuda_over_openvino                  | override=``, ov_usable=true, dgpu present, cuda_usable       | `[CUDA, CPU]` |
| U8  | auto_cpu_when_nothing_usable                     | override=``, nothing usable                                  | `[CPU]` |
| U9  | rec_model_stays_cpu_under_openvino               | plan=[OpenVINO,CPU], model=`ppocr_rec`                       | `[CPU]` |
| U10 | non_rec_models_keep_openvino                     | plan=[OpenVINO,CPU], model=`ppocr_det` / `rfdetr` / `lama`   | `[OpenVINO, CPU]` |
| U11 | rec_uses_normal_providers_under_cpu              | plan=[CPU], model=`ppocr_rec`                                | `[CPU]` |
| U12 | regression: cuda/coreml/dml branches unchanged   | existing input matrix                                        | existing outputs |

## Layer 2 — Integration tests (feature `openvino`; need libopenvino)
Location: `tests/openvino_integration.rs`. Shared tiny Identity ONNX model
(const bytes generated once); all tests `#[ignore]` by default, activated by
runner scripts per environment.

| id  | test                              | env | steps                                                        | pass |
|-----|-----------------------------------|-----|--------------------------------------------------------------|------|
| I1  | openvino_ep_smoke_cpu_device      | E1+ | build session with OpenVINO EP, device_type=CPU; infer Identity | infer output == input; no panic |
| I2  | openvino_ep_gpu_session_builds    | E2  | device_type=GPU on UHD 630                                    | session commits; infer ok |
| I3  | model_cache_dir_populated         | E1+ | after I1-style session with cache_dir, list dir               | ≥1 blob file |

## Layer 3 — In-cluster validation (scripted, real app + real models)
| id  | check                                     | pass criteria |
|-----|-------------------------------------------|---------------|
| V1  | boot banner + hardware status with `MT_DEVICE=openvino` on E2 | banner/status reports OpenVINO provider, not CPU |
| V2  | detector+OCR parity on a real book page: CPU build vs OV build (same page) | region count ±1; mean box IoU ≥ 0.95; OCR text equal |
| V3  | e2e retranslate one page via `/api/chapters/1/translate` on OV build | pipeline completes; all regions translated; no new page errors |

## Gates
- Phase-0 "done" = U* green in E0 and E2, I1–I3 green in E2, V1+V3 pass, V2 within thresholds.
- Failure of I2/V1 with a clear "runtime missing/ABI mismatch" error ⇒ trigger
  fallback plan: build ONNX Runtime from source with `--use_openvino` against
  OV 2024.6 in the image (documented escape hatch, adds ~20 min CI).

## Red → green sequence
1. commit 1: Layer-1 tests (red: fns don't exist) → extract pure core + openvino logic → green.
2. commit 2: Layer-2 tests (ignored) + `create_session_from_memory` OV branch + per-model override + runtime latch (unit tests only cover pure parts).
3. commit 3: validation container + E2 run scripts; record V1–V3 results here.
