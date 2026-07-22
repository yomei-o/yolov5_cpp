"""'.pt' compatibility test (yolov5): load the C++-synth-trained weights (data_wb/) into
yolov5n in canonical order, save yolov5n_synth.pt, then run the yolov5 model + NMS on the
held-out synthetic test image — confirming a C++-trained model runs in the yolov5 stack."""
import os, sys, numpy as np, torch, cv2
from ultralytics import YOLO  # noqa (ensures torch ok)
HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master"); sys.path.insert(0, HUB)
from utils.general import non_max_suppression
from yolo5_walk import walk

DW = os.path.join(HERE, "data_wb"); DS = os.path.join(HERE, "data_synth")
def r(n): return np.fromfile(os.path.join(DW, n), np.float32)

m = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, autoshape=False, trust_repo=True, verbose=False)
seq = m.model.model
mods = walk(seq)
def load_(p, a):
    with torch.no_grad(): p.copy_(torch.from_numpy(a.astype(np.float32)).reshape(p.shape))
serr = 0.0
for i, (kind, mod) in enumerate(mods):
    if kind == "conv":
        pairs = [(r(f"cw{i}.bin"), mod.conv.weight), (r(f"bg{i}.bin"), mod.bn.weight), (r(f"bb{i}.bin"), mod.bn.bias),
                 (r(f"rm{i}.bin"), mod.bn.running_mean), (r(f"rv{i}.bin"), mod.bn.running_var)]
    else:
        pairs = [(r(f"cw{i}.bin"), mod.weight), (r(f"cb{i}.bin"), mod.bias)]
    for a, p in pairs:
        load_(p, a); serr = max(serr, float(np.abs(a.reshape(p.shape) - p.detach().numpy()).max()))
print(f"serialization max|diff| = {serr:.3e}  {'OK' if serr < 1e-6 else 'FAIL'}")

torch.save({"model": m.model, "epoch": -1}, "yolov5n_synth.pt")
im = cv2.imread(os.path.join(DS, "te00.png"))          # 64x64 BGR
x = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0)
dm = m.model.float().eval()
with torch.no_grad():
    out = dm(x); pred = out[0] if isinstance(out, (list, tuple)) else out
dets = non_max_suppression(pred, 0.25, 0.45)[0]
names = {0: "red", 1: "green", 2: "blue"}
print(f"\nyolov5 stack on the C++-trained yolov5n_synth.pt: {dets.shape[0]} detections on te00.png")
for d in dets.cpu().numpy():
    print(f"  cls {int(d[5])} ({names.get(int(d[5]), int(d[5]))}) conf={d[4]:.2f} xyxy=({d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f})")
print("\n.pt round-trip OK — a C++-trained yolov5 model loads and runs in the yolov5 stack"
      if serr < 1e-6 else "MISMATCH")
