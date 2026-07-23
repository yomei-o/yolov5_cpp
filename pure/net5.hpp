// Full yolov5n forward on the pure autograd engine (reused from yolov8_cpp). Blocks:
// Conv (conv+BN+SiLU, BN folded), C3 (CSP), SPPF, and the anchor Detect head (one
// 1x1 conv per level -> na*no channels). Consumes convs in the order export_yolov5.py
// emits them. Per-conv padding is stored (yolov5 layer0 is k6/s2/p2, not k/2).
#pragma once
#include "autograd.hpp"
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>

struct ConvW { Tensor w, b; int64_t stride, pad, act; };
struct Provider { std::vector<ConvW> convs; size_t i = 0; ConvW& next() { return convs[i++]; } };

inline std::vector<float> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", p.c_str()); std::exit(1); }
  auto n = f.tellg(); f.seekg(0); std::vector<float> v(n / sizeof(float)); f.read((char*)v.data(), n); return v;
}

inline Provider load_net(const std::string& D) {
  std::ifstream f(D + "manifest.txt"); if (!f) { printf("run: python pure/ref/export_yolov5.py\n"); std::exit(1); }
  int n; f >> n; Provider p;
  for (int i = 0; i < n; ++i) {
    int64_t Co, Ci, k, s, pad, act; f >> Co >> Ci >> k >> s >> pad >> act;
    ConvW c; c.stride = s; c.pad = pad; c.act = act;
    c.w = from_data({Co, Ci, k, k}, rd(D + "w" + std::to_string(i) + ".bin"));
    c.b = from_data({Co}, rd(D + "b" + std::to_string(i) + ".bin"));
    p.convs.push_back(c);
  }
  return p;
}
inline Provider load_net_blob(const std::string& D) {
  std::ifstream f(D + "manifest.txt"); if (!f) { printf("run: python pure/ref/export_yolov5.py\n"); std::exit(1); }
  int n; f >> n; std::vector<float> blob = rd(D + "weights.bin");
  Provider p; size_t off = 0;
  for (int i = 0; i < n; ++i) {
    int64_t Co, Ci, k, s, pad, act; f >> Co >> Ci >> k >> s >> pad >> act;
    int64_t wn = Co * Ci * k * k;
    ConvW c; c.stride = s; c.pad = pad; c.act = act;
    c.w = from_data({Co, Ci, k, k}, std::vector<float>(blob.begin() + off, blob.begin() + off + wn)); off += wn;
    c.b = from_data({Co}, std::vector<float>(blob.begin() + off, blob.begin() + off + Co)); off += Co;
    p.convs.push_back(c);
  }
  return p;
}

inline Tensor conv_apply(const Tensor& x, ConvW& c) {
  auto y = conv2d(x, c.w, c.b, c.stride, c.pad);
  return c.act ? silu(y) : y;
}
inline Tensor cL(const Tensor& x, Provider& p) { return conv_apply(x, p.next()); }

// C3: cv1, n bottlenecks on cv1's output, cv2 on x, cv3 on concat(m_out, cv2).
inline Tensor c3(const Tensor& x, Provider& p, int64_t n_bott, bool shortcut) {
  auto last = conv_apply(x, p.next());             // cv1
  for (int64_t i = 0; i < n_bott; ++i) {
    auto h = conv_apply(last, p.next());           // bottleneck cv1 (1x1)
    h = conv_apply(h, p.next());                   // bottleneck cv2 (3x3)
    last = shortcut ? add(h, last) : h;
  }
  auto y2 = conv_apply(x, p.next());               // cv2
  return conv_apply(concat_ch({last, y2}), p.next());   // cv3(concat)
}
inline Tensor sppf(const Tensor& x, Provider& p) {
  auto x1 = conv_apply(x, p.next());
  auto q1 = maxpool2d(x1, 5, 1, 2), q2 = maxpool2d(q1, 5, 1, 2), q3 = maxpool2d(q2, 5, 1, 2);
  return conv_apply(concat_ch({x1, q1, q2, q3}), p.next());
}

// Full yolov5 forward (n/s/m/l/x). dep = C3 repeat counts in forward order (8 blocks);
// widths are data-driven from the loaded conv shapes. Returns 3 raw detect-head outputs.
inline std::vector<Tensor> yolov5n_forward(const Tensor& x, Provider& p,
                                           const std::vector<int64_t>& dep = {1,2,3,1,1,1,1,1}) {
  auto x0 = cL(x, p);                 // 0 Conv k6s2
  auto x1 = cL(x0, p);               // 1 Conv
  auto x2 = c3(x1, p, dep[0], true);      // 2 C3
  auto x3 = cL(x2, p);              // 3 Conv
  auto x4 = c3(x3, p, dep[1], true);      // 4 C3   (save)
  auto x5 = cL(x4, p);              // 5 Conv
  auto x6 = c3(x5, p, dep[2], true);      // 6 C3   (save)
  auto x7 = cL(x6, p);              // 7 Conv
  auto x8 = c3(x7, p, dep[3], true);      // 8 C3
  auto x9 = sppf(x8, p);             // 9 SPPF
  auto x10 = cL(x9, p);             // 10 Conv (save)
  auto x11 = upsample_nearest(x10, 2);          // 11
  auto x12 = concat_ch({x11, x6});              // 12
  auto x13 = c3(x12, p, dep[4], false);   // 13 C3
  auto x14 = cL(x13, p);            // 14 Conv (save)
  auto x15 = upsample_nearest(x14, 2);          // 15
  auto x16 = concat_ch({x15, x4});              // 16
  auto x17 = c3(x16, p, dep[5], false);   // 17 C3  -> P3
  auto x18 = cL(x17, p);            // 18 Conv
  auto x19 = concat_ch({x18, x14});             // 19
  auto x20 = c3(x19, p, dep[6], false);   // 20 C3  -> P4
  auto x21 = cL(x20, p);            // 21 Conv
  auto x22 = concat_ch({x21, x10});             // 22
  auto x23 = c3(x22, p, dep[7], false);   // 23 C3  -> P5
  std::vector<Tensor> out;
  for (auto& xi : {x17, x20, x23}) out.push_back(conv_apply(xi, p.next()));  // Detect m[i] (plain)
  return out;
}
