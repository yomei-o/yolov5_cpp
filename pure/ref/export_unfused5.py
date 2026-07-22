"""Export yolov5n WITHOUT folding BN, in canonical order, so the pure engine can run
conv + BatchNorm2d + SiLU as separate ops (for BN training and .pt write-back). Writes
into data_net/ alongside the fused export (reuses its x.bin / ref_head.bin).
Usage: python export_unfused5.py"""
import os, torch
from yolo5_walk import walk

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_net"); os.makedirs(D, exist_ok=True)

m = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, autoshape=False, trust_repo=True, verbose=False)
mods = walk(m.model.model.eval())

def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(mods))]
for i, (kind, mod) in enumerate(mods):
    if kind == "conv":
        conv, bn = mod.conv, mod.bn
        save(f"cw{i}.bin", conv.weight)
        save(f"bg{i}.bin", bn.weight); save(f"bb{i}.bin", bn.bias)
        save(f"rm{i}.bin", bn.running_mean); save(f"rv{i}.bin", bn.running_var)
        Co, Ci = conv.weight.shape[0], conv.weight.shape[1]
        lines.append(f"1 {Co} {Ci} {conv.kernel_size[0]} {conv.stride[0]} {conv.padding[0]} {bn.eps}")
    else:
        save(f"cw{i}.bin", mod.weight); save(f"cb{i}.bin", mod.bias)
        Co, Ci = mod.weight.shape[0], mod.weight.shape[1]
        lines.append(f"0 {Co} {Ci} {mod.kernel_size[0]} {mod.stride[0]} {mod.padding[0]} 0")
open(os.path.join(D, "manifest_unfused.txt"), "w").write("\n".join(lines) + "\n")
print(f"unfused: {len(mods)} layers")
