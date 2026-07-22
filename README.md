# yolov5_cpp

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

Planned next: anchor-based v5 loss (forward + backward) vs the reference `ComputeLoss`,
training loop, decode + NMS inference, and mAP. See the shared methodology in
yolov8_cpp's PORTING_GUIDE.

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
