// yolov5 training CLI — a real training loop, pure C++ / no Python at run time:
// dataset scan -> shuffled mini-batches (+ hflip/brightness aug) over epochs -> anchor
// build_targets -> yolov5 loss (CIoU + obj + cls BCE) -> Adam (warmup + cosine) ->
// per-epoch validation mAP@0.5 -> save last.pt / best.pt via the pure-C++ .pt writer.
//   build: bash /c/prog/claude/cc5.sh -std:c++20 -O2 -EHsc -Ipure/third_party pure/train_cli.cpp -Fe:train_cli.exe -Fo:scratch/
//   run:   train_cli <train_list> <val_list> <epochs> <batch> <init.pt>
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"
#include "net5_unfused.hpp"
#include "loss5.hpp"
#include "optim.hpp"
#include "infer5.hpp"
#include "metrics.hpp"
#include "ptio.hpp"
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <random>

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered so per-epoch progress shows in redirected/background runs
  std::string trainL = argc>1?argv[1]:"pure/ref/data_synth/list.txt";
  std::string valL   = argc>2?argv[2]:"pure/ref/data_synth/val.txt";
  int EPOCHS = argc>3?atoi(argv[3]):8, BATCH = argc>4?atoi(argv[4]):4;
  std::string initpt = argc>5?argv[5]:"yolov5n.pt";
  int imgsz = argc>6?atoi(argv[6]):0;                 // >0 => standard-YOLO dataset (dir/list + normalised labels)
  bool mosaic = argc>7?atoi(argv[7])!=0:(imgsz>0);    // mosaic on by default in YOLO mode
  const std::string DN = "pure/ref/data_net/";
  const int64_t NC = 80, NA = 3, NO = 85;

  Dataset tr, va;
  if (imgsz>0) { tr = read_yolo_dataset(trainL, imgsz); va = read_yolo_dataset(valL, imgsz); }
  else { tr = read_dataset(trainL); va = read_dataset(valL); }
  int64_t S = tr.S;
  auto anchors = rd(DN + "anchors.bin");
  std::vector<int64_t> grids = {S/8, S/16, S/32};
  std::vector<int> strides = {8, 16, 32};
  std::vector<int64_t> dep; { std::ifstream f(DN + "depths.txt"); int64_t v; while (f >> v) dep.push_back(v); }
  if (dep.empty()) dep = {1,2,3,1,1,1,1,1};
  printf("train=%zu val=%zu imgsz=%lld batch=%d epochs=%d fmt=%s mosaic=%d\n", tr.items.size(), va.items.size(), (long long)S, BATCH, EPOCHS, tr.yolo?"yolo":"legacy", (int)mosaic);

  // initial weights: from the init .pt (pure-C++ read, arch from the tiny manifest) if it
  // exists, else from the Python-exported .bin files.
  ProviderU prov; { std::ifstream t(initpt); if (t.good()) { printf("init weights <- %s (pure C++)\n", initpt.c_str()); prov = load_net_unfused_pt(DN, initpt); } else prov = load_net_unfused(DN); }
  std::vector<Tensor> params; for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b); }
  Adam opt(params, 1e-3f, 0.9f, 0.999f, 1e-8f, 5e-4f, false);

  // state_dict KEY per engine tensor (engine/C3 emit order != state_dict order; pair by
  // name — load_state_dict matches by key). names.txt is emitted by export_unfused5.py.
  std::vector<std::string> names; { std::ifstream f(DN + "names.txt"); std::string s; while (f >> s) names.push_back(s); }
  auto save_ckpt = [&](const std::string& path) {
    std::vector<pt::Tensor> ck; size_t k = 0;
    auto push = [&](const std::vector<float>& d, std::vector<int64_t> shp){ pt::Tensor t; if (k<names.size()) t.name=names[k]; t.shape=shp; t.data=d; ck.push_back(t); ++k; };
    for (auto& L : prov.layers) { std::vector<int64_t> ws(L.w->shape.begin(),L.w->shape.end());
      push(L.w->data, ws);
      if (L.kind==1){ std::vector<int64_t> c={L.gamma->shape[0]}; push(L.gamma->data,c); push(L.beta->data,c); push(L.rm,c); push(L.rv,c); }
      else push(L.b->data, {L.b->shape[0]}); }
    pt::save_pt(ck, path);
  };

  auto validate = [&]() -> double {
    std::vector<mapeval::Image> imgs;
    for (auto& s : va.items) {
      Letterbox lb; auto xi = load_image_letterbox(s.img, S, lb);   // no aug/mosaic at eval
      prov.i=0; auto hv = yolov5n_forward_u(xi, prov, false, dep);
      int64_t N; auto pred = decode5(hv, anchors, strides, NA, NO, NC, N);
      auto dets = nms5(pred, N, NO, NC, 0.25f, 0.45f, 100);
      mapeval::Image im;
      for (auto& d : dets) im.dts.push_back({d.x1,d.y1,d.x2,d.y2,d.cls,d.conf});
      std::vector<float> gb; std::vector<int64_t> gl; int m = load_boxes_orig(s.lbl, va.yolo, lb.w0, lb.h0, gb, gl); lb_map(gb, lb);
      for (int j=0;j<m;++j) im.gts.push_back({gb[j*4],gb[j*4+1],gb[j*4+2],gb[j*4+3],(int)gl[j]});
      imgs.push_back(im);
    }
    return mapeval::coco_map(imgs).first;   // mAP@0.50
  };

  AugCfg baseAug; baseAug.mosaic = mosaic; baseAug.mixup = mosaic;     // HSV/affine/flip on by default
  int closeMosaic = argc>8 ? atoi(argv[8]) : std::max(1, EPOCHS/10);   // disable mosaic/mixup for last N epochs
  std::vector<int> order(tr.items.size()); std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(0);
  int steps_per_epoch = ((int)tr.items.size() + BATCH - 1) / BATCH, total = EPOCHS * steps_per_epoch, gstep = 0;
  double best = -1;
  for (int ep = 0; ep < EPOCHS; ++ep) {
    AugCfg aug = baseAug; if (ep >= EPOCHS - closeMosaic) { aug.mosaic = false; aug.mixup = false; }
    std::shuffle(order.begin(), order.end(), rng); double eloss = 0; int nb = 0;
    for (size_t off = 0; off < order.size(); off += BATCH) {
      std::vector<int> idx(order.begin()+off, order.begin()+std::min(order.size(), off+BATCH));
      Batch bt = load_minibatch(tr, idx, true, rng(), aug);
      int64_t B = bt.B, M = bt.M;
      // build normalized targets (NT,6) = [img, cls, xn, yn, wn, hn]
      std::vector<float> targets; int NT = 0;
      for (int64_t b=0;b<B;++b) for (int64_t m=0;m<M;++m) if (bt.mask[b*M+m] > 0) {
        float x1=bt.gt_boxes[(b*M+m)*4],y1=bt.gt_boxes[(b*M+m)*4+1],x2=bt.gt_boxes[(b*M+m)*4+2],y2=bt.gt_boxes[(b*M+m)*4+3];
        targets.insert(targets.end(), {(float)b,(float)bt.gt_labels[b*M+m], ((x1+x2)/2)/S, ((y1+y2)/2)/S, (x2-x1)/(float)S, (y2-y1)/(float)S});
        ++NT;
      }
      prov.i=0; auto heads = yolov5n_forward_u(bt.x, prov, true, dep);
      std::vector<Tensor> p; for (auto& h : heads) p.push_back(head_to_pred(h, NA, NO));
      auto L = compute_loss_v5(p, targets, NT, anchors, grids, B, NA, NO, NC);
      backward(L.total);
      opt.lr = cosine_lr(gstep, total, 1e-3f, std::max(1, total/20)); opt.step(); ++gstep;
      eloss += L.total->data[0]; ++nb;
    }
    double m50 = validate();
    printf("epoch %2d/%d  loss %6.3f  val mAP@0.5 %.3f%s\n", ep+1, EPOCHS, eloss/nb, m50, m50>best?"  *best*":"");
    save_ckpt("last.pt"); if (m50 > best) { best = m50; save_ckpt("best.pt"); }
  }
  printf("done. best val mAP@0.5 = %.3f. wrote last.pt / best.pt (pure C++)\n", best);
  return 0;
}
