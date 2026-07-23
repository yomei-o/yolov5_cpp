// yolov5n forward with conv + BatchNorm2d + SiLU kept SEPARATE (BN not folded), so the
// same weights can be trained and written back to a .pt. Same topology as net5.hpp.
#pragma once
#include "autograd.hpp"
#include "bn.hpp"
#include "net5.hpp"          // rd()
#include <fstream>
#include <string>

struct LayerU { int kind; int64_t stride, pad; float eps; Tensor w, b, gamma, beta; std::vector<float> rm, rv; };
struct ProviderU { std::vector<LayerU> layers; size_t i = 0; LayerU& next() { return layers[i++]; } };

inline ProviderU load_net_unfused(const std::string& D) {
  std::ifstream f(D + "manifest_unfused.txt"); if (!f) { printf("run: python pure/ref/export_unfused5.py\n"); std::exit(1); }
  int n; f >> n; ProviderU p;
  for (int i = 0; i < n; ++i) {
    int kind; int64_t Co, Ci, k, s, pad; float eps; f >> kind >> Co >> Ci >> k >> s >> pad >> eps;
    LayerU L; L.kind = kind; L.stride = s; L.pad = pad; L.eps = eps;
    L.w = from_data({Co, Ci, k, k}, rd(D + "cw" + std::to_string(i) + ".bin"), true);
    if (kind == 1) {
      L.gamma = from_data({Co}, rd(D + "bg" + std::to_string(i) + ".bin"), true);
      L.beta  = from_data({Co}, rd(D + "bb" + std::to_string(i) + ".bin"), true);
      L.rm = rd(D + "rm" + std::to_string(i) + ".bin");
      L.rv = rd(D + "rv" + std::to_string(i) + ".bin");
    } else L.b = from_data({Co}, rd(D + "cb" + std::to_string(i) + ".bin"), true);
    p.layers.push_back(std::move(L));
  }
  return p;
}

inline Tensor applyU(const Tensor& x, LayerU& L, bool tr) {
  if (L.kind == 1) { auto y = conv2d(x, L.w, nullptr, L.stride, L.pad);
    y = batchnorm2d(y, L.gamma, L.beta, L.rm, L.rv, L.eps, tr, 0.03f); return silu(y); }
  return conv2d(x, L.w, L.b, L.stride, L.pad);
}
inline Tensor cLU(const Tensor& x, ProviderU& p, bool tr) { return applyU(x, p.next(), tr); }
inline Tensor c3_u(const Tensor& x, ProviderU& p, int64_t n, bool sc, bool tr) {
  auto last = applyU(x, p.next(), tr);
  for (int64_t i = 0; i < n; ++i) { auto h = applyU(last, p.next(), tr); h = applyU(h, p.next(), tr); last = sc ? add(h, last) : h; }
  auto y2 = applyU(x, p.next(), tr);
  return applyU(concat_ch({last, y2}), p.next(), tr);
}
inline Tensor sppf_u(const Tensor& x, ProviderU& p, bool tr) {
  auto x1 = applyU(x, p.next(), tr);
  auto q1 = maxpool2d(x1, 5, 1, 2), q2 = maxpool2d(q1, 5, 1, 2), q3 = maxpool2d(q2, 5, 1, 2);
  return applyU(concat_ch({x1, q1, q2, q3}), p.next(), tr);
}
inline std::vector<Tensor> yolov5n_forward_u(const Tensor& x, ProviderU& p, bool tr,
                                             const std::vector<int64_t>& d = {1,2,3,1,1,1,1,1}) {
  auto x0 = cLU(x, p, tr); auto x1 = cLU(x0, p, tr);
  auto x2 = c3_u(x1, p, d[0], true, tr); auto x3 = cLU(x2, p, tr);
  auto x4 = c3_u(x3, p, d[1], true, tr); auto x5 = cLU(x4, p, tr);
  auto x6 = c3_u(x5, p, d[2], true, tr); auto x7 = cLU(x6, p, tr);
  auto x8 = c3_u(x7, p, d[3], true, tr); auto x9 = sppf_u(x8, p, tr);
  auto x10 = cLU(x9, p, tr); auto x11 = upsample_nearest(x10, 2);
  auto x12 = concat_ch({x11, x6}); auto x13 = c3_u(x12, p, d[4], false, tr);
  auto x14 = cLU(x13, p, tr); auto x15 = upsample_nearest(x14, 2);
  auto x16 = concat_ch({x15, x4}); auto x17 = c3_u(x16, p, d[5], false, tr);
  auto x18 = cLU(x17, p, tr); auto x19 = concat_ch({x18, x14}); auto x20 = c3_u(x19, p, d[6], false, tr);
  auto x21 = cLU(x20, p, tr); auto x22 = concat_ch({x21, x10}); auto x23 = c3_u(x22, p, d[7], false, tr);
  std::vector<Tensor> out; for (auto& xi : {x17, x20, x23}) out.push_back(applyU(xi, p.next(), tr));
  return out;
}
