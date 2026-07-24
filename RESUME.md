# RESUME — remaining work

Status of the pure-C++ training toolchain (yolov5) and what's left to make it a full replacement
for Ultralytics-quality training. Verified items live in [README.md](README.md); this file
is the forward-looking TODO.

## Done (pure C++, no Python at run time)
- Engine + all 4 YOLOs (v5/v8/v11/x), all sizes (n/s/m/l/x), forward/loss/train/infer/mAP/ONNX/`.pt`.
- Real training CLI (`pure/train_cli.cpp`): dataset scan → shuffled mini-batches → epochs →
  assignment → loss → Adam(warmup+cosine+wd) → per-epoch val mAP@0.5 → `best.pt`/`last.pt`.
- Initial-weight `.pt` generated in C++ (`pure/make_init_pt.cpp`, `rand`/`from`), all sizes,
  zero-Python bootstrap; checkpoints load back into Ultralytics/reference (0 unexpected).
- **Standard-YOLO dataset ingestion** — directory scan (`images/`↔`labels/`), normalised
  `cls xc yc w h` labels, arbitrary-size images letterboxed (`pure/dataset.hpp`
  `read_yolo_dataset` / `load_boxes_orig`). Verified: val mAP → 0.97 on the synthetic set.
- **Augmentation** — mosaic + mixup + random-affine (rotate/scale/shear/translate) + HSV +
  flip, with **close-mosaic** (disable for last N epochs). Toggle via `AugCfg` / CLI flags.
- **Unified `yolo` CLI** (`pure/yolo.cpp`) reading `data.yaml`: `train` / `val` / `detect`
  (`export` still delegates to the standalone `onnx_export` — see remaining #3).
  Val reports **mAP@0.5 and mAP@0.5:0.95**.
- GPU/CUDA seam (`pure/backend.hpp`) present in all four; conv/matmul route through `bk::`.
  GPU training done for all four (per the parallel session).

## Remaining (roughly in priority order)
1. **Real-dataset convergence parity** — train on COCO128 (or similar) and compare final
   mAP@0.5:0.95 against Ultralytics. This is the key "results, not just pipeline" validation.
   Nothing beyond synthetic data has been checked for convergence quality yet.
2. **Custom `nc`** — the head is fixed at 80 classes; training on a dataset with `nc != 80`
   needs the cls head resized + re-initialised (make_init_pt could emit an `nc`-sized head).
   Today class ids must be < 80.
3. **`export` in the unified CLI** — fold BN from the `.pt` and emit ONNX in-CLI (today
   `yolo export` points at the standalone, onnxruntime-verified `onnx_export`).
4. **Training-quality features** — EMA weights, resume-from-checkpoint, multi-scale, rect val,
   label smoothing, separate bias/BN LR + warmup-bias-lr. (mAP@0.5:0.95 in val — done.)
5. **Speed** — yolox uses a per-image forward summed per minibatch (batch it like v8).
6. **CPU speed on Apple Silicon** — the CUDA seam doesn't help on Mac (Metal≠CUDA). Add a
   BLAS path to `bk::gemm_hosted` (Apple Accelerate / OpenBLAS) for a big CPU speedup without
   a GPU; a full Metal backend is a much larger, lower-priority effort.

## Notes / gotchas
- Label coords: internally everything is xyxy in the **letterboxed SxS pixel** space; GT and
  decoded detections share it, so val mAP is apples-to-apples. `load_boxes_orig` reads either
  format into original pixels, then `lb_map` applies the letterbox transform.
- `make_init_pt … rand` trains but won't converge on tiny data; use `from <pretrained.pt>`
  for transfer learning.
- Build: MSVC via `C:/prog/claude/cc5.sh`; `scratch/` must pre-exist; vcvars hangs here.

7. **Verify the unified `train_cli`/`yolo.cpp` under a CUDA build** — compile with `nvcc -DUSE_CUDA` and run COCO128 end-to-end on a (free-Colab) T4. The CUDA seam + a training loop were verified on T4, but the new dataset-ingestion + augmentation CLI path has not been built/run under nvcc yet (aug/dataset are host-side; conv/matmul auto-route to `bk::` on GPU). Est. COCO128/640px/100ep: T4 GPU ~7-20 min; CPU ~a day (measured ~5.7 s/image fwd+bwd at 640px, naive GEMM) so a real GPU is the fix.

## Eigen + cuDNN compute backends (ported from yolov8_cpp)  ✅ DONE 2026-07-25

Ported and (CPU side) verified locally: Eigen gemm 6.5×, ThrustCPU dnet5_test MATCH, train_cli
`-DUSE_EIGEN` builds, `nvcc -DUSE_CUDA` clean. cuDNN path is in and guard-isolated; verify it on
Colab via `colab/dnet_cudnn_test.ipynb`. Original porting notes below for reference.


yolov8_cpp gained two **opt-in** compute backends (both on main, verified on a Colab T4). The
seam is identical across all repos (`pure/backend.hpp` `bk::gemm*` + `pure/dtensor.hpp` `dconv2d`),
so port both here. Reference: yolov8_cpp commits "optional Eigen GEMM backend" and "optional cuDNN
conv backend", plus `colab/dnet_cudnn_test.ipynb`.

### A. Eigen GEMM (CPU, `-DUSE_EIGEN`) — fast CPU training, zero external dependency
1. Copy the vendored flat Eigen folder from yolov8_cpp: `pure/third_party/eigen_flat/`
   (130 headers, no subdirs, entry header `Eigen_Core.h`, MPL2). It is fully self-contained.
2. In `pure/backend.hpp`: (a) under the CPU `#else` include block add
   `#ifdef USE_EIGEN  #include "Eigen_Core.h"  #endif`; (b) in the CPU section add an
   `#elif defined(USE_EIGEN)` variant of `gemm` / `gemm_nt` / `gemm_tn` that maps the row-major
   float pointers to `Eigen::Map` and does `Cm.noalias() = Am * Bm` (nt/tn use `.transpose()`;
   beta!=0 → `Cm *= beta; Cm.noalias() += …`). Copy verbatim from yolov8_cpp — it is engine-agnostic.
3. Build train_cli with `-DUSE_EIGEN -Ipure/third_party/eigen_flat` + `/arch:AVX2` (cl) or
   `-march=native` (g++). The scratch engine's conv2d/matmul route through
   `bk::gemm_hosted → bk::gemm`, so train_cli picks it up automatically.
   Verify: grad-check clean; `pure/bench_gemm.cpp` (copy from v8) shows ~10-12× over baseline.
   v8 measured: gemm 12.5×, train_cli end-to-end ~3×, parity 1e-6.

### B. cuDNN conv (GPU, `-DUSE_CUDNN`) — fastest GPU training
1. In `pure/dtensor.hpp`: add the cuDNN header/handle block (`bk::bk_cudnn()` + `CUDNN_CHECK`)
   guarded by `#if defined(USE_CUDA) && defined(USE_CUDNN)`, and an **early-return** cuDNN path at
   the top of `dconv2d`: fwd `cudnnConvolutionForward` (+ `cudnnAddTensor` for bias); backward
   `cudnnConvolutionBackwardData` (dIn) / `BackwardFilter` (dW) / `BackwardBias` (dBias). Use
   `CUDNN_CROSS_CORRELATION` (matches im2col, no kernel flip) and **beta=1** on the bwd calls so
   grads accumulate like the existing `+=`. Copy from yolov8_cpp dtensor.hpp.
2. **[v5] `dconv2d` is groups=1 (no `groups` param) → identical to yolov8, straight copy.**
3. No cudnn.h on the dev machine → build/verify on Colab using v8's `colab/dnet_cudnn_test.ipynb`
   as a template (it auto-detects the cuDNN header/lib from Colab's torch-bundled cuDNN).
   Expect: `dnet*_test` forward MATCH; training loss decreases like the cuBLAS build.
   v8 measured (T4, COCO128 640 b4): cuBLAS 88.7 → **cuDNN 36.0 s/epoch (2.46×)**.

### Guardrails
- Both flags are opt-in add-ons. The default builds (scratch CPU / Thrust-CPU / plain
  `-DUSE_CUDA`) must stay behaviorally identical — verify they still compile after the edits
  (`run_all.sh`; and a `nvcc -DUSE_CUDA` compile WITHOUT `-DUSE_CUDNN` to confirm the guard is clean).
- Add the "Compute backends — one source, five builds" table to README (copy from yolov8_cpp).
