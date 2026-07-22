// yolov5 inference postprocess: anchor decode + NMS (plain float), matching yolov5's
// Detect eval decode (xy=(2σ-0.5+grid)·stride, wh=(2σ)²·anchor·stride, obj/cls sigmoid)
// and non_max_suppression (objectness gate, conf=obj·cls, class-aware greedy NMS).
#pragma once
#include "net5.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

struct Det5 { float x1, y1, x2, y2, conf; int cls; };

// heads: nl x (1, na*no, ny, nx). anchors (nl*na*2) grid units. strides per level.
// Returns pred (N, no) = [cx,cy,w,h,obj,cls...] (pixel xywh, sigmoids), N level-major, (a,y,x).
inline std::vector<float> decode5(const std::vector<Tensor>& heads, const std::vector<float>& anchors,
                                  const std::vector<int>& strides, int64_t na, int64_t no, int64_t nc, int64_t& N) {
  N = 0; for (auto& h : heads) N += na * h->shape[2] * h->shape[3];
  std::vector<float> pred(N * no); int64_t row = 0;
  for (size_t i = 0; i < heads.size(); ++i) {
    auto& h = heads[i]; int64_t ny = h->shape[2], nx = h->shape[3]; float st = (float)strides[i];
    const float* Hd = h->data.data();
    for (int64_t a = 0; a < na; ++a) {
      float aw = anchors[(i*na+a)*2+0] * st, ah = anchors[(i*na+a)*2+1] * st;
      for (int64_t y = 0; y < ny; ++y) for (int64_t x = 0; x < nx; ++x) {
        auto sg = [&](int64_t k) { return 1.f / (1.f + std::exp(-Hd[(((a*no+k)*ny)+y)*nx+x])); };
        float* pr = &pred[row * no];
        pr[0] = (sg(0)*2 - 0.5f + x) * st;
        pr[1] = (sg(1)*2 - 0.5f + y) * st;
        float sw = sg(2)*2, sh = sg(3)*2;
        pr[2] = sw*sw*aw; pr[3] = sh*sh*ah;
        pr[4] = sg(4);
        for (int64_t k = 0; k < nc; ++k) pr[5+k] = sg(5+k);
        ++row;
      }
    }
  }
  return pred;
}

static inline float iou5(const Det5& a, const Det5& b) {
  float iw = std::max(0.f, std::min(a.x2,b.x2) - std::max(a.x1,b.x1));
  float ih = std::max(0.f, std::min(a.y2,b.y2) - std::max(a.y1,b.y1));
  float inter = iw*ih, ua = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
  return ua > 0 ? inter/ua : 0.f;
}

inline std::vector<Det5> nms5(const std::vector<float>& pred, int64_t N, int64_t no, int64_t nc,
                              float conf_thres, float iou_thres, int max_det = 300) {
  const float max_wh = 7680.f;
  std::vector<Det5> cand;
  for (int64_t i = 0; i < N; ++i) {
    const float* p = &pred[i * no];
    float obj = p[4]; if (obj <= conf_thres) continue;            // objectness gate
    int best = 0; float bc = p[5];
    for (int64_t k = 1; k < nc; ++k) if (p[5+k] > bc) { bc = p[5+k]; best = (int)k; }
    float conf = obj * bc; if (conf <= conf_thres) continue;
    float cx = p[0], cy = p[1], w = p[2], h = p[3];
    cand.push_back({cx - w*0.5f, cy - h*0.5f, cx + w*0.5f, cy + h*0.5f, conf, best});
  }
  std::sort(cand.begin(), cand.end(), [](const Det5& a, const Det5& b) { return a.conf > b.conf; });
  std::vector<Det5> keep; std::vector<char> removed(cand.size(), 0);
  for (size_t i = 0; i < cand.size() && (int)keep.size() < max_det; ++i) {
    if (removed[i]) continue;
    keep.push_back(cand[i]);
    Det5 bi = cand[i]; float o = cand[i].cls * max_wh; bi.x1+=o; bi.x2+=o; bi.y1+=o; bi.y2+=o;
    for (size_t j = i+1; j < cand.size(); ++j) {
      if (removed[j] || cand[j].cls != cand[i].cls) continue;
      Det5 bj = cand[j]; float o2 = cand[j].cls * max_wh; bj.x1+=o2; bj.x2+=o2; bj.y1+=o2; bj.y2+=o2;
      if (iou5(bi, bj) > iou_thres) removed[j] = 1;
    }
  }
  return keep;
}
