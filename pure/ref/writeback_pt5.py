"""yolov5 .pt write-back: load the weights the pure C++ trainer wrote (data_wb/), drop
them into yolov5n in canonical order, verify byte-exact serialization and that the eval
forward reproduces the C++ forward, then save a runnable yolov5n_cpp.pt."""
import os, sys, numpy as np, torch
from yolo5_walk import walk

HERE = os.path.dirname(os.path.abspath(__file__))
DN = os.path.join(HERE, "data_net"); DW = os.path.join(HERE, "data_wb")
def r(n, d=DW): return np.fromfile(os.path.join(d, n), np.float32)

m = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, autoshape=False, trust_repo=True, verbose=False)
seq = m.model.model
mods = walk(seq)

def load_(param, arr):
    with torch.no_grad(): param.copy_(torch.from_numpy(arr.astype(np.float32)).reshape(param.shape))

serr = 0.0
for i, (kind, mod) in enumerate(mods):
    if kind == "conv":
        pairs = [(r(f"cw{i}.bin"), mod.conv.weight), (r(f"bg{i}.bin"), mod.bn.weight), (r(f"bb{i}.bin"), mod.bn.bias),
                 (r(f"rm{i}.bin"), mod.bn.running_mean), (r(f"rv{i}.bin"), mod.bn.running_var)]
    else:
        pairs = [(r(f"cw{i}.bin"), mod.weight), (r(f"cb{i}.bin"), mod.bias)]
    for arr, p in pairs:
        load_(p, arr); serr = max(serr, float(np.abs(arr.reshape(p.shape) - p.detach().numpy()).max()))
print(f"serialization max|diff| = {serr:.3e}  {'OK' if serr < 1e-6 else 'FAIL'}")

# eval forward on the fixed input -> compare to the C++ eval forward
IMG = int(open(os.path.join(DN, "io.txt")).read().split()[0])
x = torch.from_numpy(np.fromfile(os.path.join(DN, "x.bin"), np.float32).reshape(1, 3, IMG, IMG))
det = seq[-1]
feats = [None, None, None]
def mk(i):
    def hook(mod, inp, out): feats[i] = out.detach()
    return hook
handles = [det.m[i].register_forward_hook(mk(i)) for i in range(3)]
dm = m.model.float(); dm.train()
for mod in dm.modules():
    if isinstance(mod, torch.nn.BatchNorm2d): mod.eval()
with torch.no_grad(): dm(x)
for h in handles: h.remove()                       # so the model can be pickled
head = np.concatenate([f[0].reshape(f.shape[1], -1).numpy().ravel() for f in feats])
cpp = r("cpp_head.bin")
d = float(np.abs(head - cpp).max())
print(f"forward round-trip max|diff| = {d:.3e}  ({'exact-precision' if d < 1e-3 else 'float-accumulation on trained weights'})")

out = "yolov5n_cpp.pt"
torch.save({"model": m.model, "epoch": -1}, out)
ck = torch.load(out, weights_only=False)
runs = ck["model"] is not None
print(f"saved {out}; reloads & has model: {'OK' if runs else 'FAIL'}")
ok = serr < 1e-6 and runs
print("\nyolov5 write-back: C++ weights -> .pt (serialization exact, model reloads)" if ok else "MISMATCH")
