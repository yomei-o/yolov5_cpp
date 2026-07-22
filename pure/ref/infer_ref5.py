"""Inference reference for yolov5. Letterboxes a real image, runs yolov5n eval to get
the decoded predictions (1, N, 5+nc) and runs yolov5 NMS, so the pure C++ decode+NMS
can be checked. Usage: python infer_ref5.py [imgsz] [image]"""
import os, sys, numpy as np, torch, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master")
sys.path.insert(0, HUB)
from utils.augmentations import letterbox
from utils.general import non_max_suppression

D = os.path.join(HERE, "data_infer"); os.makedirs(D, exist_ok=True)
S    = int(sys.argv[1]) if len(sys.argv) > 1 else 320
IMGP = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "assets", "bus.jpg")
CONF, IOU = 0.25, 0.45

m = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, autoshape=False, trust_repo=True, verbose=False)
dm = m.model.float().eval()
det = dm.model[-1]; nc = det.nc

im0 = cv2.imread(IMGP); h0, w0 = im0.shape[:2]
im = letterbox(im0, (S, S), stride=32, auto=False)[0]
r = min(S / h0, S / w0); dw = (S - round(w0 * r)) / 2; dh = (S - round(h0 * r)) / 2
x = im[:, :, ::-1].transpose(2, 0, 1)[None]
x = np.ascontiguousarray(x).astype(np.float32) / 255.0
xt = torch.from_numpy(x)

with torch.no_grad():
    out = dm(xt)
pred = out[0] if isinstance(out, (list, tuple)) else out   # (1, N, 5+nc) decoded
dets = non_max_suppression(pred, CONF, IOU)[0]              # (n,6) xyxy,conf,cls

def save(n, t): (t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)).astype(np.float32).tofile(os.path.join(D, n))
save("x.bin", xt)
save("ref_pred.bin", pred[0])           # (N, 5+nc)
lines = [f"{S} {nc} {pred.shape[1]} {CONF} {IOU}", f"{w0} {h0} {r} {dw} {dh}", str(dets.shape[0])]
for d in dets.cpu().numpy():
    lines.append(f"{d[0]:.4f} {d[1]:.4f} {d[2]:.4f} {d[3]:.4f} {d[4]:.6f} {int(d[5])}")
open(os.path.join(D, "meta.txt"), "w").write("\n".join(lines) + "\n")
print(f"imgsz={S} nc={nc} N={pred.shape[1]} dets={dets.shape[0]}")
for d in dets.cpu().numpy():
    print(f"  cls {int(d[5])} conf={d[4]:.3f} xyxy=({d[0]:.1f},{d[1]:.1f},{d[2]:.1f},{d[3]:.1f})")
