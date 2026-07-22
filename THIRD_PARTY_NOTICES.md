# Third-party notices / サードパーティ ライセンス表記

This repository bundles third-party components. Each retains its own license and
copyright; the notices below apply to those files only.

このリポジトリには第三者の成果物を同梱しています。各ファイルはそれぞれの
ライセンス・著作権に従います。以下の表記はそれらのファイルにのみ適用されます。

---

## 1. Ultralytics YOLOv5 — model weights & derived data

- **Files:** `weights/yolov5n/*` (BN-folded repack of yolov5n's parameters), any
  `yolov5*.pt`, and reference tensors generated from the model under `pure/ref/`
  (git-ignored).
- **Copyright:** © Ultralytics
- **Source:** https://github.com/ultralytics/yolov5
- **License:** **GNU Affero General Public License v3.0 (AGPL-3.0-or-later)**
  — full text: https://www.gnu.org/licenses/agpl-3.0.html

> Note: these are AGPL-3.0 artifacts. Redistributing this repository (or a network
> service built from it) must comply with the AGPL-3.0 terms for them.
> 注: これらは AGPL-3.0 です。再配布・ネットワーク公開時は AGPL-3.0 の義務を満たす必要があります。

## 2. stb — single-header image libraries

- **Files:** `pure/third_party/stb_image.h`, `pure/third_party/stb_image_write.h`
- **Author:** Sean Barrett and contributors. **Source:** https://github.com/nothings/stb
- **License:** dual-licensed — **Public Domain (Unlicense)** OR **MIT**, at your option.
  Full text is at the bottom of each header file.

---

The project's own source code (`pure/` engine, `ref/` scripts, docs) is licensed under
the **BSD 3-Clause License** — see [`LICENSE`](LICENSE). Distributing it together with
the AGPL-3.0 artifacts above carries the AGPL obligations for those specific files.
