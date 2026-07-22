// yolov5 anchor-based loss on the pure engine: build_targets (plain, wh-ratio anchor
// matching + 3-neighbour cell expansion) and the differentiable loss (box CIoU +
// objectness BCE + class BCE), faithfully following the original yolov5 ComputeLoss.
#pragma once
#include "ops2d.hpp"
#include <array>

// --- extra differentiable ops ---
inline Tensor gather_rows(const Tensor& a, const std::vector<int64_t>& idx) {
  int64_t C = a->shape.back(), R = (int64_t)idx.size();
  auto o = make_tensor({R, C}, true);
  for (int64_t r = 0; r < R; ++r) for (int64_t c = 0; c < C; ++c) o->data[r * C + c] = a->data[idx[r] * C + c];
  o->parents = {a}; Node* op = o.get();
  o->backward_fn = [a, op, idx, R, C] { for (int64_t r = 0; r < R; ++r) for (int64_t c = 0; c < C; ++c) a->grad[idx[r] * C + c] += op->grad[r * C + c]; };
  return o;
}
inline Tensor slice_cols(const Tensor& a, int64_t c0, int64_t c1) {
  int64_t C = a->shape.back(), R = a->numel() / C, Cs = c1 - c0;
  auto sh = a->shape; sh.back() = Cs; auto o = make_tensor(sh, true);
  for (int64_t r = 0; r < R; ++r) for (int64_t j = 0; j < Cs; ++j) o->data[r * Cs + j] = a->data[r * C + c0 + j];
  o->parents = {a}; Node* op = o.get();
  o->backward_fn = [a, op, R, C, Cs, c0] { for (int64_t r = 0; r < R; ++r) for (int64_t j = 0; j < Cs; ++j) a->grad[r * C + c0 + j] += op->grad[r * Cs + j]; };
  return o;
}

// Detect head output (BS, na*no, ny, nx) -> prediction layout (BS, na, ny, nx, no),
// matching yolov5's view(bs,na,no,ny,nx).permute(0,1,3,4,2). Differentiable.
inline Tensor head_to_pred(const Tensor& h, int64_t na, int64_t no) {
  int64_t BS = h->shape[0], C = h->shape[1], ny = h->shape[2], nx = h->shape[3];
  auto o = make_tensor({BS, na, ny, nx, no}, true);
  for (int64_t b = 0; b < BS; ++b) for (int64_t a = 0; a < na; ++a) for (int64_t y = 0; y < ny; ++y)
    for (int64_t x = 0; x < nx; ++x) for (int64_t k = 0; k < no; ++k)
      o->data[((((b*na+a)*ny+y)*nx+x)*no)+k] = h->data[(((b*C + a*no+k)*ny+y)*nx+x)];
  o->parents = {h}; Node* op = o.get();
  o->backward_fn = [h, op, BS, C, na, no, ny, nx] {
    for (int64_t b = 0; b < BS; ++b) for (int64_t a = 0; a < na; ++a) for (int64_t y = 0; y < ny; ++y)
      for (int64_t x = 0; x < nx; ++x) for (int64_t k = 0; k < no; ++k)
        h->grad[(((b*C + a*no+k)*ny+y)*nx+x)] += op->grad[((((b*na+a)*ny+y)*nx+x)*no)+k];
  };
  return o;
}

inline Tensor sq5(const Tensor& z) { return mul(z, z); }

// CIoU between paired boxes (N,4) xyxy -> (N,1). alpha detached (as yolov5/ultralytics).
inline Tensor ciou_rows5(const Tensor& b1, const Tensor& b2, float eps = 1e-7f) {
  auto x1 = narrow_col(b1, 0), y1 = narrow_col(b1, 1), x2 = narrow_col(b1, 2), y2 = narrow_col(b1, 3);
  auto X1 = narrow_col(b2, 0), Y1 = narrow_col(b2, 1), X2 = narrow_col(b2, 2), Y2 = narrow_col(b2, 3);
  auto w1 = sub(x2, x1), h1 = sub(y2, y1), w2 = sub(X2, X1), h2 = sub(Y2, Y1);
  auto inter = mul(clampmin_scalar(sub(min2(x2, X2), max2(x1, X1)), 0.f),
                   clampmin_scalar(sub(min2(y2, Y2), max2(y1, Y1)), 0.f));
  auto uni = add_scalar(sub(add(mul(w1, h1), mul(w2, h2)), inter), eps);
  auto iou = divi(inter, uni);
  auto cw = sub(max2(x2, X2), min2(x1, X1)), ch = sub(max2(y2, Y2), min2(y1, Y1));
  auto c2 = add_scalar(add(mul(cw, cw), mul(ch, ch)), eps);
  auto rho2 = mul_scalar(add(sq5(sub(add(x1, x2), add(X1, X2))), sq5(sub(add(y1, y2), add(Y1, Y2)))), 0.25f);
  float pi = (float)std::acos(-1.0);
  auto v = mul_scalar(sq5(sub(op_atan(divi(w2, add_scalar(h2, eps))), op_atan(divi(w1, add_scalar(h1, eps))))), 4.f / (pi * pi));
  auto alpha = detach(divi(v, add_scalar(sub(v, iou), 1.f + eps)));
  return sub(iou, add(divi(rho2, c2), mul(v, alpha)));
}

// BCE-with-logits (stable), elementwise, z constant: max(x,0) - x*z + log1p(exp(-|x|)).
inline Tensor bce_logits5(const Tensor& x, const Tensor& z) {
  auto t3 = op_log1p(op_exp(mul_scalar(op_abs(x), -1.f)));
  return add(sub(clampmin_scalar(x, 0.f), mul(x, z)), t3);
}

// One positive target: a cell (b,a,gj,gi) with box (grid units) + class + anchor.
struct Pos { int64_t b, a, gj, gi; float tx, ty, tw, th; int cls; float aw, ah; };

// build_targets for one level. targets (NT,6): [img,cls,xn,yn,wn,hn] normalized.
// anchors_lvl (na*2) grid units. nx,ny grid size. anchor_t threshold (4.0).
inline std::vector<Pos> build_targets_level(const std::vector<float>& targets, int NT,
                                            const std::vector<float>& anchors_lvl, int64_t na,
                                            int64_t nx, int64_t ny, float anchor_t) {
  const float g = 0.5f;
  const float off[5][2] = {{0,0},{1,0},{0,1},{-1,0},{0,-1}};
  // Rows surviving the anchor wh-ratio filter, in yolov5's order: anchor-major, then
  // target. Each row carries its grid box + neighbour masks.
  struct Row { int b, c; int64_t a; float gx, gy, gw, gh, aw, ah; bool use[5]; };
  std::vector<Row> rows;
  for (int64_t a = 0; a < na; ++a) {
    float aw = anchors_lvl[a*2+0], ah = anchors_lvl[a*2+1];
    for (int t = 0; t < NT; ++t) {
      float gw = targets[t*6+4]*nx, gh = targets[t*6+5]*ny;
      float m = std::max(std::max(gw/aw, aw/gw), std::max(gh/ah, ah/gh));
      if (m >= anchor_t) continue;
      float gx = targets[t*6+2]*nx, gy = targets[t*6+3]*ny;
      float gxi = nx - gx, gyi = ny - gy;
      Row r; r.b=(int)targets[t*6+0]; r.c=(int)targets[t*6+1]; r.a=a;
      r.gx=gx; r.gy=gy; r.gw=gw; r.gh=gh; r.aw=aw; r.ah=ah;
      r.use[0]=true;
      r.use[1]=(std::fmod(gx,1.f)<g)&&(gx>1.f);
      r.use[2]=(std::fmod(gy,1.f)<g)&&(gy>1.f);
      r.use[3]=(std::fmod(gxi,1.f)<g)&&(gxi>1.f);
      r.use[4]=(std::fmod(gyi,1.f)<g)&&(gyi>1.f);
      rows.push_back(r);
    }
  }
  // Emit offset-major, then row order (matches yolov5's t.repeat(5,1,1)[j] selection),
  // so duplicate-cell tobj overwrites match.
  std::vector<Pos> out;
  for (int o = 0; o < 5; ++o)
    for (auto& r : rows) {
      if (!r.use[o]) continue;
      int gi = (int)std::floor(r.gx - off[o][0]*g), gj = (int)std::floor(r.gy - off[o][1]*g);
      Pos p; p.b=r.b; p.a=r.a; p.cls=r.c; p.aw=r.aw; p.ah=r.ah;
      p.gj = std::clamp<int64_t>(gj, 0, ny-1); p.gi = std::clamp<int64_t>(gi, 0, nx-1);
      p.tx = r.gx - gi; p.ty = r.gy - gj; p.tw = r.gw; p.th = r.gh;
      out.push_back(p);
    }
  return out;
}

struct Loss5 { Tensor total, lbox, lobj, lcls; };

// p[i]: leaf tensors shaped {BS,na,ny,nx,no} (flattened). anchors (nl*na*2) grid units.
inline Loss5 compute_loss_v5(const std::vector<Tensor>& p, const std::vector<float>& targets, int NT,
                             const std::vector<float>& anchors, const std::vector<int64_t>& grids,
                             int64_t BS, int64_t na, int64_t no, int64_t nc, float anchor_t = 4.0f) {
  const float balance[3] = {4.0f, 1.0f, 0.4f};
  const float H_box = 0.05f, H_obj = 1.0f, H_cls = 0.5f;
  Tensor lbox, lobj, lcls;
  auto acc = [](Tensor& s, const Tensor& v) { s = s ? add(s, v) : v; };

  for (size_t i = 0; i < p.size(); ++i) {
    int64_t g = grids[i], Ncells = BS * na * g * g;
    auto Pi2 = reshape(p[i], {Ncells, no});
    std::vector<float> anc_lvl(anchors.begin() + i*na*2, anchors.begin() + (i+1)*na*2);
    auto pos = build_targets_level(targets, NT, anc_lvl, na, g, g, anchor_t);
    std::vector<float> tobj(Ncells, 0.f);

    if (!pos.empty()) {
      int64_t Np = (int64_t)pos.size();
      std::vector<int64_t> idx(Np); std::vector<float> ancd(Np*2), tbx(Np*4);
      std::vector<int> cls(Np);
      for (int64_t r = 0; r < Np; ++r) {
        auto& q = pos[r];
        idx[r] = ((q.b*na + q.a)*g + q.gj)*g + q.gi;
        ancd[r*2+0]=q.aw; ancd[r*2+1]=q.ah;
        tbx[r*4+0]=q.tx; tbx[r*4+1]=q.ty; tbx[r*4+2]=q.tw; tbx[r*4+3]=q.th;
        cls[r]=q.cls;
      }
      auto Pp = gather_rows(Pi2, idx);                                   // (Np,no)
      auto pxy = add_scalar(mul_scalar(sigmoid(slice_cols(Pp,0,2)), 2.f), -0.5f);      // (Np,2)
      auto sw  = mul_scalar(sigmoid(slice_cols(Pp,2,4)), 2.f);
      auto pwh = mul(mul(sw, sw), from_data({Np,2}, ancd));             // (2*sig)^2 * anchor
      auto pcx=narrow_col(pxy,0), pcy=narrow_col(pxy,1), pw=narrow_col(pwh,0), ph=narrow_col(pwh,1);
      auto pb = concat_cols({sub(pcx, mul_scalar(pw,0.5f)), sub(pcy, mul_scalar(ph,0.5f)),
                             add(pcx, mul_scalar(pw,0.5f)), add(pcy, mul_scalar(ph,0.5f))});
      std::vector<float> tbxyxy(Np*4);
      for (int64_t r=0;r<Np;++r){ float cx=tbx[r*4],cy=tbx[r*4+1],w=tbx[r*4+2],h=tbx[r*4+3];
        tbxyxy[r*4+0]=cx-w/2; tbxyxy[r*4+1]=cy-h/2; tbxyxy[r*4+2]=cx+w/2; tbxyxy[r*4+3]=cy+h/2; }
      auto iou = ciou_rows5(pb, from_data({Np,4}, tbxyxy));             // (Np,1)
      acc(lbox, mean(add_scalar(mul_scalar(iou,-1.f), 1.f)));           // mean(1-iou)
      for (int64_t r=0;r<Np;++r) tobj[idx[r]] = std::max(0.f, iou->data[r]);   // gr=1

      if (nc > 1) {
        auto pcls = slice_cols(Pp, 5, 5+nc);
        std::vector<float> tc(Np*nc, 0.f);
        for (int64_t r=0;r<Np;++r) tc[r*nc + cls[r]] = 1.f;
        acc(lcls, mean(bce_logits5(pcls, from_data({Np,nc}, tc))));
      }
    }
    auto pobj = narrow_col(Pi2, 4);
    acc(lobj, mul_scalar(mean(bce_logits5(pobj, from_data({Ncells,1}, tobj))), balance[i]));
  }
  auto zero = [] { return from_data({1}, {0.f}); };
  if (!lbox) lbox = zero(); if (!lcls) lcls = zero();
  lbox = mul_scalar(lbox, H_box); lobj = mul_scalar(lobj, H_obj); lcls = mul_scalar(lcls, H_cls);
  auto total = mul_scalar(add(add(lbox, lobj), lcls), (float)BS);
  return {total, lbox, lobj, lcls};
}
