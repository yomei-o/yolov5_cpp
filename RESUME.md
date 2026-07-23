# RESUME — remaining work

Status of the pure-C++ YOLOv5 training toolchain and what's left to match Ultralytics-quality
training. Verified items are in [README.md](README.md); this file is the forward-looking TODO.

## Done (pure C++, no Python at run time)
- Pure autograd engine; YOLOv5 forward + **anchor-based loss** (build_targets + CIoU/obj/cls),
  training, inference (anchor decode + NMS), COCO mAP — all verified vs the yolov5 reference.
- All sizes (n/s/m/l/x): arch (`pure/ref/arch/<model>/`) + `make_init_pt` generate initial
  weights in C++ with zero Python; checkpoints load back into the yolov5 reference (0 unexpected).
- Real training CLI (`pure/train_cli.cpp`): dataset scan → shuffled mini-batches → epochs →
  build_targets → loss → Adam(warmup+cosine+wd) → per-epoch val mAP@0.5 → `best.pt`/`last.pt`.
- **Standard-YOLO dataset ingestion** — dir scan (`images/`↔`labels/`), normalised
  `cls xc yc w h`, arbitrary-size images letterboxed (`pure/dataset.hpp`). Verified: val
  mAP@0.5 → 0.93 on the synthetic set; `best.pt` reloads in the reference (0 unexpected).
- **Mosaic augmentation** + horizontal flip + brightness. `train_cli … <imgsz> <mosaic>`.
- GPU/CUDA seam (`pure/backend.hpp`); conv/matmul route through `bk::`.

## Remaining (roughly in priority order)
1. **Real-dataset convergence parity** — train on COCO128 (or similar), compare final
   mAP@0.5:0.95 to Ultralytics yolov5. Only synthetic convergence has been checked.
2. **Richer augmentation** — HSV colour jitter, random affine (scale/translate/rotate/shear),
   mixup, "close mosaic for the last N epochs". Only flip + brightness + mosaic exist today.
3. **`data.yaml` + unified CLI** — parse `data.yaml` (paths, `nc`, `names`) and add
   `train`/`val`/`detect`/`export` subcommands.
4. **Training-quality features** — EMA, resume-from-checkpoint, multi-scale, rect val,
   label smoothing, warmup-bias-lr, mAP@0.5:0.95 in the val loop (only mAP@0.5 printed now).
5. **Speed** — verify the GPU path on real hardware.

## Notes / gotchas
- Everything internal is xyxy in the **letterboxed SxS pixel** space; `load_boxes_orig`
  reads either label format into original pixels, then `lb_map` applies the letterbox.
- The yolov5 hub reference wraps the net at `m.model` — load checkpoints into `m.model`.
- `train_cli` uses unbuffered stdout so progress shows in redirected/background runs.
- Build: MSVC via `C:/prog/claude/cc5.sh`; `scratch/` must pre-exist.
