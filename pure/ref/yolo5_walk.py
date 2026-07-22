"""Canonical traversal of yolov5's conv-bearing modules, in the order the pure C++
forward consumes them. Shared by the unfused exporter and the .pt write-back so both
agree. Yields ('conv', Conv) for conv+BN+SiLU modules and ('plain', nn.Conv2d) for the
detect head convs."""

def walk(seq):
    order = []
    def emit(m): order.append(("conv", m))          # ultralytics Conv (has .conv, .bn)
    def plain(c): order.append(("plain", c))         # nn.Conv2d
    def c3(b):
        emit(b.cv1)
        for bott in b.m: emit(bott.cv1); emit(bott.cv2)
        emit(b.cv2); emit(b.cv3)
    def sppf(b): emit(b.cv1); emit(b.cv2)
    for mod in seq:
        t = type(mod).__name__
        if t == "Conv": emit(mod)
        elif t == "C3": c3(mod)
        elif t == "SPPF": sppf(mod)
        elif t == "Detect":
            for cc in mod.m: plain(cc)
    return order
