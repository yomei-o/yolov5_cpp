# yolov5_cpp


## Train with your own dataset (standard YOLO) + mosaic

`pure/train_cli.cpp` trains from a **standard Ultralytics/YOLO dataset**: pass an images
directory (or a list file) plus `imgsz`, and it scans `images/`↔`labels/`, reads normalised
`cls xc yc w h` labels, and letterboxes arbitrary-size images. A 7th arg toggles **mosaic**
augmentation (on by default in this mode; flip + brightness always on).

```sh
python pure/ref/make_synth_yolo.py 40      # tiny dataset in the standard images/ + labels/ layout
./make_init_pt init.pt from yolov5n.pt     # initial weights .pt, pure C++
./train_cli pure/ref/data_yolo/images/train pure/ref/data_yolo/images/val 6 4 init.pt 96 0
#   fmt=yolo ... val mAP@0.5 -> 0.93 ; best.pt reloads in the yolov5 reference (0 unexpected)
```

**Remaining work** (real-dataset convergence parity, richer augmentation, `data.yaml`/unified
CLI, EMA/resume) is tracked in **[RESUME.md](RESUME.md)**.

Training **YOLOv5** (the classic anchor-based model) in C++ with **zero external
dependencies** — a from-scratch reverse-mode autograd engine, C++ standard library only
(plus two vendored single-header image libs for the demo). Every step is **verified
numerically against the original Ultralytics YOLOv5** (PyTorch).

日本語: 本物の anchor ベース YOLOv5 を C++ で学習する実験。自作 autograd エンジン
（標準ライブラリのみ）で yolov5n の順伝播・損失・学習・推論を再現し、各段階を本家
（`torch.hub` の ultralytics/yolov5）と数値比較して確かめる。CPU / OpenMP は `-fopenmp`
の有無だけで切り替え。姉妹プロジェクト: **yolov8_cpp**。

The autograd engine (`pure/autograd.hpp` — im2col+GEMM conv, BN, SiLU, pool, upsample,
concat, …), optimizers, dataloader and COCO-mAP are shared with **yolov8_cpp**. What is
new here is YOLOv5's architecture and its **anchor-based loss** (box CIoU + objectness
BCE + class BCE, `build_targets` with wh-ratio anchor matching — no DFL, no TAL).

## Status
| file | milestone | result |
|------|-----------|--------|
| `pure/net5.hpp` + `pure/m1_forward.cpp` | **full yolov5n forward** (Conv / C3 / SPPF / anchor head) | matches yolov5n ~2e-5 |
| `pure/loss5.hpp` + `pure/m2_loss.cpp` | **anchor-based v5 loss** (build_targets + box CIoU + obj BCE + cls BCE) fwd+bwd | matches yolov5 `ComputeLoss`: loss ~6e-8, grads ~3e-9 |
| `pure/m3_train.cpp` | **end-to-end training** (forward → loss → backward → Adam/cosine) | loss 3.3 → 1.1 |
| `pure/infer5.hpp` + `pure/m4_infer.cpp` | **inference: anchor decode + NMS** | dets match yolov5 ~2e-4 |
| `pure/m5_demo.cpp` | **real-image inference** (stb_image → letterbox → detect → annotate) | bus + 3 people |
| `pure/metrics.hpp` + `pure/m6_map.cpp` | **COCO mAP** (AP@0.50, AP@0.50:0.95) | match pycocotools ~3e-7 |
| `pure/net5_unfused.hpp` + `pure/m7_unfused.cpp` | unfused conv+BN+SiLU forward | matches yolov5n ~2e-5 |
| `pure/m8_train_writeback.cpp` | **train (live BN) → write weights back to `.pt`** | serialization exact; yolov5 reloads it |
| `pure/onnx.hpp` + `pure/onnx_export5.cpp` | **ONNX writer** (hand-rolled protobuf, no deps) | onnxruntime runs it, ~1e-5 |
| `pure/onnx_run.hpp` + `pure/m9_onnx_run.cpp` | **ONNX reader + graph interpreter** | pure engine runs the `.onnx`, ~2e-5 |

## Demo — real-image detection, no Python, no libraries
Weights ship in the repo (`weights/yolov5n/`), so the pure detector runs from a checkout
with only a C++ compiler + the two vendored single-header image libs:
```sh
g++ -std=c++20 -O2 -Ipure/third_party pure/m5_demo.cpp -o m5_demo   # or cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\m5_demo.cpp
./m5_demo assets/bus.jpg bus_out.png 640
```
| `assets/bus.jpg` → | `assets/zidane.jpg` → |
|---|---|
| ![bus](assets/bus_detected.png) | ![zidane](assets/zidane_detected.png) |

These match yolov5n's own output (boxes ~2e-4 on the letterboxed input, same classes —
see `pure/m4_infer.cpp`). Decode + NMS are in `pure/infer5.hpp`.

## Train with zero Python — the full loop in C++

`pure/train_cli.cpp` is a real training environment, pure C++, **no Python at run time**:
dataset scan → shuffled mini-batches (hflip + brightness augmentation) over epochs →
`build_targets` (anchor matching) → v5 loss (box CIoU + obj/cls BCE) → Adam (warmup +
cosine LR + weight decay) → **per-epoch validation mAP@0.5** → save `best.pt` / `last.pt`
via the pure-C++ `.pt` writer.

```sh
cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\make_init_pt.cpp   # or g++ ...
cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\train_cli.cpp

./make_init_pt init.pt from yolov5n.pt      # C++ builds the initial-weights .pt (see below)
./train_cli pure/ref/data_synth/list.txt pure/ref/data_synth/val.txt 8 4 init.pt
#   epoch 1/8  loss 2.37  val mAP@0.5 0.039
#   epoch 7/8  loss 1.14  val mAP@0.5 1.000   -> best.pt / last.pt (pure C++)
```

The C++-trained `best.pt` loads straight back into the yolov5 reference model
(`torch.hub.load('ultralytics/yolov5','yolov5n',autoshape=False)`, `m.model.load_state_dict`,
0 unexpected keys) and detects the right classes — train/retrain in C++, drop the result
into any PyTorch pipeline. (Checkpoint keys are paired by **name** via `names.txt`, because
the engine's C3 emit order differs from PyTorch's `state_dict` order.)

### Make the initial-weights `.pt` in C++ — no Python needed to bootstrap

`pure/make_init_pt.cpp` writes a valid `state_dict` `.pt` entirely in C++, driven only by
two tiny text files that ship in the repo — `pure/ref/data_net/manifest_unfused.txt`
(per-layer shapes) and `names.txt` (state_dict keys). No Python, no libtorch:

- **`rand`** — He/Kaiming random init, fully self-contained (needs neither a pretrained
  file nor Python). Loads into the yolov5 reference model (0 unexpected). Trains
  mechanically, but from-scratch convergence needs real data volume + many epochs.
- **`from <pretrained.pt>`** — copies pretrained values read in C++ by `ptio`
  (`load_pt` for a state_dict, `load_pt_module` for a raw checkpoint). This is the
  practical **transfer-learning** init; the only input is the `.pt` file itself (just
  download it — no Python). Verified to reproduce the fine-tune run exactly (mAP → 1.000).

**All sizes.** The generator is size-agnostic — it just reads the arch files. Every size's
`manifest_unfused.txt` + `names.txt` ship under `pure/ref/arch/<model>/` (n/s/m/l/x), so
`./make_init_pt out.pt rand yolov5n.pt pure/ref/arch/yolov5m/` builds an init `.pt` for any
size with zero Python. Each verified to load into its architecture (0 unexpected keys):
n/s 291, m 401, l 511, x 621 tensors. Regenerate them with
`python pure/ref/export_arch_all.py` (built from the `.yaml` configs, no download).

`train_cli` starts from that init `.pt` (`load_net_unfused_pt` in `pure/net5_unfused.hpp`,
arch from the manifest, tensors looked up by `names.txt` key) when it's present, else from
the `.bin` export. So a fresh clone bootstraps and trains with **zero Python**:
`make_init_pt` → `init.pt` → `train_cli`. Regenerate the synthetic set with
`python pure/ref/make_synth.py 96 24` (the one Python touch, only to fabricate demo images).

## Build
```sh
# reference weights (needs: torch, and the yolov5 hub deps pandas/seaborn/… )
python pure/ref/export_yolov5.py 64        # dump yolov5n fused weights + reference forward

# pure track — just a compiler
g++ -std=c++20 -O2            pure/m1_forward.cpp -o m1     # CPU
g++ -std=c++20 -O2 -fopenmp   pure/m1_forward.cpp -o m1     # OpenMP (same result)
cl /std:c++20 /O2 /EHsc pure/m1_forward.cpp                 # MSVC (std::thread parallelism)
./m1
```

## Licenses & attribution
The repository's own code is **BSD 3-Clause** — see [LICENSE](LICENSE). Bundled
third-party components keep their own licenses (Ultralytics YOLOv5 weights **AGPL-3.0**,
stb **public-domain / MIT**) — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Unified CLI + `data.yaml`

`pure/yolo.cpp` is a single `yolo` command reading a standard Ultralytics `data.yaml`
(`path`/`train`/`val`/`nc`/`names`):
```sh
bash /c/prog/claude/cc5.sh -std:c++20 -O2 -EHsc -Ipure/third_party pure/yolo.cpp -Fe:yolo.exe -Fo:scratch/
./yolo train  --data data.yaml --weights init.pt --imgsz 640 --epochs 100 --batch 16 \
              --mosaic 1 --mixup 1 --close-mosaic 10          # HSV/affine/flip on by default
./yolo val    --data data.yaml --weights best.pt --imgsz 640  # -> mAP@0.5 and mAP@0.5:0.95
./yolo detect --weights best.pt --source img.jpg --out out.png --data data.yaml
```
Augmentation (mosaic, mixup, random-affine, HSV, flip, close-mosaic) lives in
`pure/dataset.hpp` (`AugCfg`). `yolo export` points at the standalone ONNX exporter.
Remaining work is tracked in **[RESUME.md](RESUME.md)**.
