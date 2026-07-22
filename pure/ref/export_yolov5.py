"""Export yolov5n (BN-folded) convs in the exact order the pure C++ forward consumes
them, plus a fixed input and the 3 raw detect-head outputs (bs,255,ny,nx) as reference.
Also dumps anchors + strides. Usage: python export_yolov5.py [imgsz]"""
import os, sys, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_net"); os.makedirs(D, exist_ok=True)
IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 64

m = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, autoshape=False, trust_repo=True, verbose=False)
seq = m.model.model.eval()
det = seq[-1]

convs = []   # (w, b, k, s, pad, act)
def fuse(cv):
    conv, bn = cv.conv, cv.bn
    std = torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    b = bn.bias - bn.weight * bn.running_mean / std
    return w, b
def emit(mod):                                    # Conv (conv+bn+silu)
    w, b = fuse(mod); c = mod.conv
    convs.append((w, b, c.kernel_size[0], c.stride[0], c.padding[0], 1))
def emit_plain(c):                                # nn.Conv2d (detect head, bias, no act)
    convs.append((c.weight, c.bias, c.kernel_size[0], c.stride[0], c.padding[0], 0))
def emit_c3(b):                                   # cv1, (bott.cv1,bott.cv2)*n, cv2, cv3
    emit(b.cv1)
    for bott in b.m: emit(bott.cv1); emit(bott.cv2)
    emit(b.cv2); emit(b.cv3)
def emit_sppf(b): emit(b.cv1); emit(b.cv2)

for i, mod in enumerate(seq):
    t = type(mod).__name__
    if t == "Conv": emit(mod)
    elif t == "C3": emit_c3(mod)
    elif t == "SPPF": emit_sppf(mod)
    elif t == "Detect":
        for cc in mod.m: emit_plain(cc)

def save(n, t): (t.detach() if torch.is_tensor(t) else torch.tensor(t)).contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(convs))]; blob = []
for i, (w, b, k, s, p, act) in enumerate(convs):
    save(f"w{i}.bin", w); save(f"b{i}.bin", b)
    blob.append(w.detach().cpu().numpy().ravel()); blob.append(b.detach().cpu().numpy().ravel())
    lines.append(f"{w.shape[0]} {w.shape[1]} {k} {s} {p} {act}")
open(os.path.join(D, "manifest.txt"), "w").write("\n".join(lines) + "\n")
np.concatenate(blob).astype(np.float32).tofile(os.path.join(D, "weights.bin"))

# reference: capture the 3 detect-head conv outputs (bs,255,ny,nx) via hooks (BN in eval)
feats = [None, None, None]
def mk(i):
    def hook(mod, inp, out): feats[i] = out.detach()
    return hook
for i in range(3): det.m[i].register_forward_hook(mk(i))
torch.manual_seed(0)
x = torch.randn(1, 3, IMG, IMG)
mdl = m.model.float(); mdl.train()
for mod in mdl.modules():
    if isinstance(mod, torch.nn.BatchNorm2d): mod.eval()
with torch.no_grad(): mdl(x)

save("x.bin", x)
ref = np.concatenate([f[0].reshape(f.shape[1], -1).cpu().numpy().ravel() for f in feats])  # (255*sum(ny*nx))
ref.astype(np.float32).tofile(os.path.join(D, "ref_head.bin"))
# anchors (grid units) + strides
anc = det.anchors.reshape(det.nl, det.na, 2).cpu().numpy()   # per level, grid units
save("anchors.bin", anc)                                     # (nl,na,2)
save("strides.bin", det.stride.cpu().numpy())
open(os.path.join(D, "io.txt"), "w").write(
    f"{IMG} {det.nc} {det.na} {det.no} {det.nl} " + " ".join(str(f.shape[2]) for f in feats) + "\n")
print(f"convs={len(convs)} imgsz={IMG} nc={det.nc} na={det.na} no={det.no} feats={[tuple(f.shape) for f in feats]}")
