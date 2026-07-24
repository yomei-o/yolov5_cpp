// Device-resident yolov5 net (any size) + trainable provider + checkpoint save.
// Mirrors net5_unfused.hpp topology (C3 = cv1 + n bottlenecks + parallel cv2 -> concat ->
// cv3; SPPF; anchor Detect = one plain conv per level) using device ops (dtensor.hpp).
// Per-layer padding is taken from the manifest (yolov5 convs are not all k/2). BN running
// stats are EMA-tracked so a saved .pt is eval-ready. One source CPU/GPU.
#pragma once
#include "net5_unfused.hpp"    // ProviderU, load_net_unfused_pt (reads manifest+names, per-layer pad)
#include "dtensor.hpp"
#include <string>
#include <fstream>

struct DT5 { int kind; int64_t stride, pad; float eps; DT w, gamma, beta, b, rm, rv; };
struct ProvD5 { std::vector<DT5> L; size_t i = 0; DT5& next() { return L[i++]; } };

inline ProvD5 dnet5_build(const std::string& arch, const std::string& weights) {
  ProviderU pu = load_net_unfused_pt(arch, weights);
  ProvD5 p;
  for (auto& L : pu.layers) {
    DT5 d; d.kind = L.kind; d.stride = L.stride; d.pad = L.pad; d.eps = L.eps;
    int64_t Co = L.w->shape[0], Ci = L.w->shape[1], k = L.w->shape[2];
    d.w = dfrom({Co,Ci,k,k}, L.w->data);
    if (L.kind == 1) { d.gamma = dfrom({Co}, L.gamma->data); d.beta = dfrom({Co}, L.beta->data);
                       d.rm = dfrom({Co}, L.rm); d.rv = dfrom({Co}, L.rv); }
    else d.b = dfrom({Co}, L.b->data);
    p.L.push_back(std::move(d));
  }
  return p;
}
inline std::vector<DT> dnet5_params(ProvD5& p) {
  std::vector<DT> ps; for (auto& L : p.L) { ps.push_back(L.w); if (L.kind==1){ps.push_back(L.gamma);ps.push_back(L.beta);} else ps.push_back(L.b); } return ps;
}
inline std::vector<int64_t> dnet5_depths(const std::string& arch) {
  std::vector<int64_t> d; std::ifstream f(arch + "depths.txt"); int64_t v; while (f >> v) d.push_back(v);
  if (d.empty()) d = {1,2,3,1,1,1,1,1}; return d;
}

static inline DT d5_apply(DT x, DT5& L, bool tr) {
  if (L.kind == 1) return dsilu(dbn(dconv2d(x, L.w, DT(), L.stride, L.pad), L.gamma, L.beta, L.eps,
                                    tr ? L.rm : DT(), tr ? L.rv : DT(), 0.03f));
  return dconv2d(x, L.w, L.b, L.stride, L.pad);
}
static inline DT d5_cL(DT x, ProvD5& p, bool tr) { return d5_apply(x, p.next(), tr); }
static inline DT d5_c3(DT x, ProvD5& p, int64_t n, bool sc, bool tr) {   // yolov5 C3 (no channel split)
  DT last = d5_apply(x, p.next(), tr);                                   // cv1
  for (int64_t i=0;i<n;++i){ DT h=d5_apply(last,p.next(),tr); h=d5_apply(h,p.next(),tr); last=sc?dadd(h,last):h; }
  DT y2 = d5_apply(x, p.next(), tr);                                     // cv2 (parallel from x)
  return d5_apply(dconcat({last, y2}), p.next(), tr);                    // cv3
}
static inline DT d5_sppf(DT x, ProvD5& p, bool tr) {
  DT x1=d5_apply(x,p.next(),tr); DT q1=dmaxpool2d(x1,5,1,2),q2=dmaxpool2d(q1,5,1,2),q3=dmaxpool2d(q2,5,1,2);
  return d5_apply(dconcat({x1,q1,q2,q3}), p.next(), tr);
}
inline std::vector<DT> dnet5_forward(DT x, ProvD5& p, const std::vector<int64_t>& d, bool tr) {
  DT x0=d5_cL(x,p,tr),x1=d5_cL(x0,p,tr);
  DT x2=d5_c3(x1,p,d[0],true,tr);DT x3=d5_cL(x2,p,tr);
  DT x4=d5_c3(x3,p,d[1],true,tr);DT x5=d5_cL(x4,p,tr);
  DT x6=d5_c3(x5,p,d[2],true,tr);DT x7=d5_cL(x6,p,tr);
  DT x8=d5_c3(x7,p,d[3],true,tr);DT x9=d5_sppf(x8,p,tr);
  DT x10=d5_cL(x9,p,tr);DT x11=dupsample2x(x10);
  DT x12=dconcat({x11,x6});DT x13=d5_c3(x12,p,d[4],false,tr);
  DT x14=d5_cL(x13,p,tr);DT x15=dupsample2x(x14);
  DT x16=dconcat({x15,x4});DT x17=d5_c3(x16,p,d[5],false,tr);
  DT x18=d5_cL(x17,p,tr);DT x19=dconcat({x18,x14});DT x20=d5_c3(x19,p,d[6],false,tr);
  DT x21=d5_cL(x20,p,tr);DT x22=dconcat({x21,x10});DT x23=d5_c3(x22,p,d[7],false,tr);
  std::vector<DT> out; DT lv[3]={x17,x20,x23}; for (auto& xi : lv) out.push_back(d5_apply(xi, p.next(), tr)); return out;
}
inline void dnet5_save(ProvD5& p, const std::string& arch, const std::string& path) {
  std::vector<std::string> names; { std::ifstream f(arch + "names.txt"); std::string s; while (f >> s) names.push_back(s); }
  std::vector<pt::Tensor> ck; size_t k = 0;
  auto push = [&](DT t){ pt::Tensor o; if (k<names.size()) o.name=names[k]; o.shape.assign(t->shape.begin(),t->shape.end()); o.data=dto_host(t); ck.push_back(std::move(o)); ++k; };
  for (auto& L : p.L) { push(L.w); if (L.kind==1){push(L.gamma);push(L.beta);push(L.rm);push(L.rv);} else push(L.b); }
  pt::save_pt(ck, path);
}
