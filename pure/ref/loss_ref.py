"""Reference for the pure yolov5 loss. Builds random head predictions p (3 levels,
(bs,na,ny,nx,no)) and random targets, runs the original yolov5 ComputeLoss, and dumps
p, targets, anchors, the loss components and the gradient dLoss/dp — so the C++ loss can
be checked forward + backward. Usage: python loss_ref.py"""
import os, sys, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_loss"); os.makedirs(D, exist_ok=True)
HUB = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master")
sys.path.insert(0, HUB)
from utils.loss import ComputeLoss

m = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, autoshape=False, trust_repo=True, verbose=False)
model = m.model
det = model.model[-1]
model.hyp = {"box": 0.05, "obj": 1.0, "cls": 0.5, "anchor_t": 4.0,
             "cls_pw": 1.0, "obj_pw": 1.0, "label_smoothing": 0.0, "fl_gamma": 0.0}
model.gr = 1.0
compute_loss = ComputeLoss(model)

torch.manual_seed(0)
BS, S = 2, 64
nl, na, no, nc = det.nl, det.na, det.no, det.nc
grids = [S // int(s.item()) for s in det.stride]        # [8,4,2]
p = [torch.randn(BS, na, g, g, no, requires_grad=True) for g in grids]

# random targets: (nt,6) = [img_idx, cls, xc, yc, w, h] normalized to [0,1]
NT = 8
ti = torch.randint(0, BS, (NT, 1)).float()
tc = torch.randint(0, nc, (NT, 1)).float()
txy = torch.rand(NT, 2) * 0.8 + 0.1
twh = torch.rand(NT, 2) * 0.3 + 0.05
targets = torch.cat([ti, tc, txy, twh], 1)

loss, items = compute_loss(p, targets)                   # items = [lbox, lobj, lcls]
loss.backward()

def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
for i in range(nl):
    save(f"p{i}.bin", p[i]); save(f"dp{i}.bin", p[i].grad)
save("targets.bin", targets)
save("anchors.bin", det.anchors.reshape(nl, na, 2))
open(os.path.join(D, "meta.txt"), "w").write(
    f"{BS} {S} {nl} {na} {no} {nc} {NT} " + " ".join(str(g) for g in grids) + " " +
    " ".join(str(int(s.item())) for s in det.stride) + "\n")
save("loss.bin", torch.cat([loss.reshape(1), items]))    # total, lbox, lobj, lcls
print("loss total=%.6f  lbox=%.6f lobj=%.6f lcls=%.6f" % (loss.item(), *items.tolist()))
