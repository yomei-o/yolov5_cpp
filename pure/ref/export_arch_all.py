"""Dump the architecture description (manifest_unfused.txt + names.txt) for EVERY size
n/s/m/l/x, built from yolov5's .yaml config (random init, NO pretrained download). These
tiny text files let the pure-C++ make_init_pt generate initial weights for any size with
zero Python. Writes pure/ref/arch/<model>/{manifest_unfused.txt,names.txt}.
Usage: python export_arch_all.py"""
import os, sys, glob
from yolo5_walk import walk

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master")
sys.path.insert(0, HUB)
from models.yolo import DetectionModel   # noqa: E402

SIZES = sys.argv[1:] or ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]

for MODEL in SIZES:
    cfg = os.path.join(HUB, "models", MODEL + ".yaml")
    model = DetectionModel(cfg=cfg).eval()          # architecture only, random init, no download
    mods = walk(model.model)
    qn = {id(mm): nm for nm, mm in model.named_modules()}   # module -> state_dict prefix (model.N...)
    lines = [str(len(mods))]; names = []
    for kind, mod in mods:
        p = qn[id(mod)]
        if kind == "conv":
            conv, bn = mod.conv, mod.bn
            Co, Ci = conv.weight.shape[0], conv.weight.shape[1]
            lines.append(f"1 {Co} {Ci} {conv.kernel_size[0]} {conv.stride[0]} {conv.padding[0]} {bn.eps}")
            names += [f"{p}.conv.weight", f"{p}.bn.weight", f"{p}.bn.bias", f"{p}.bn.running_mean", f"{p}.bn.running_var"]
        else:
            Co, Ci = mod.weight.shape[0], mod.weight.shape[1]
            lines.append(f"0 {Co} {Ci} {mod.kernel_size[0]} {mod.stride[0]} {mod.padding[0]} 0")
            names += [f"{p}.weight", f"{p}.bias"]
    D = os.path.join(HERE, "arch", MODEL); os.makedirs(D, exist_ok=True)
    open(os.path.join(D, "manifest_unfused.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(D, "names.txt"), "w").write("\n".join(names) + "\n")
    print(f"{MODEL}: {len(mods)} layers, {len(names)} tensors -> arch/{MODEL}/")
