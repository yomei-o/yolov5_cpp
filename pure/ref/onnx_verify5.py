"""Verify the C++-written yolov5n.onnx: onnx.checker + run in onnxruntime and compare the
3 detect-head outputs to the reference forward."""
import os, numpy as np, onnx, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
DN = os.path.join(HERE, "data_net")
onnx_path = "yolov5n.onnx"
m = onnx.load(onnx_path); onnx.checker.check_model(m); print("onnx.checker: OK")

IMG = int(open(os.path.join(DN, "io.txt")).read().split()[0])
x = np.fromfile(os.path.join(DN, "x.bin"), np.float32).reshape(1, 3, IMG, IMG)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
outs = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, {"images": x}))}
out = np.concatenate([outs[f"out{i}"].ravel() for i in range(3)])
ref = np.fromfile(os.path.join(DN, "ref_head.bin"), np.float32)
d = float(np.abs(out - ref).max())
print(f"onnxruntime head max|diff| = {d:.3e}  {'OK' if d < 1e-3 else 'FAIL'}")
print("\nyolov5 ONNX(write): C++ .onnx runs in onnxruntime == yolov5n forward" if d < 1e-3 else "MISMATCH")
