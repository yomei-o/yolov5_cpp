// Unified pure-C++ CLI for yolov5: `yolo <train|val|detect|export> [--flags]`, reading a
// standard Ultralytics data.yaml. No Python at run time.
//   train  --data d.yaml --weights init.pt [--epochs 100 --batch 16 --imgsz 640
//                                            --mosaic 1 --mixup 1 --close-mosaic 10]
//   val    --data d.yaml --weights best.pt [--imgsz 640]
//   detect --weights best.pt --source img.jpg [--imgsz 640 --conf 0.25 --out out.png]
//   export --weights best.pt                 (delegates to onnx_export — see note)
// build: bash /c/prog/claude/cc5.sh -std:c++20 -O2 -EHsc -Ipure/third_party pure/yolo.cpp -Fe:yolo.exe -Fo:scratch/
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"                  // includes stb_image.h once (with the impl above)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
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
#include <sstream>
#include <map>

static const int64_t NC0 = 80, NA = 3, NO = 85;
static const std::string DN = "pure/ref/data_net/";

// ------- data.yaml (minimal: path/train/val/nc/names, inline-list names) -------
struct DataYaml { std::string path, train, val; int nc = 80; std::vector<std::string> names; };
static std::string trim(std::string s) {
  size_t a = s.find_first_not_of(" \t\r\n'\""); size_t b = s.find_last_not_of(" \t\r\n'\"");
  return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}
static DataYaml parse_yaml(const std::string& p) {
  std::ifstream f(p); if (!f) { printf("cannot open %s\n", p.c_str()); std::exit(1); }
  DataYaml d; std::string line;
  while (std::getline(f, line)) {
    auto h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
    auto c = line.find(':'); if (c == std::string::npos) continue;
    std::string k = trim(line.substr(0, c)), v = trim(line.substr(c + 1));
    if (k == "path") d.path = v; else if (k == "train") d.train = v; else if (k == "val") d.val = v;
    else if (k == "nc") d.nc = atoi(v.c_str());
    else if (k == "names") {
      auto lb = v.find('['), rb = v.rfind(']');
      if (lb != std::string::npos && rb != std::string::npos) {
        std::stringstream ss(v.substr(lb + 1, rb - lb - 1)); std::string tok;
        while (std::getline(ss, tok, ',')) { tok = trim(tok); if (!tok.empty()) d.names.push_back(tok); }
      }
    }
  }
  return d;
}
static std::string join(const std::string& a, const std::string& b) { return a.empty() ? b : a + "/" + b; }

// ------- arg parsing: --key value / --key=value -------
struct Args { std::map<std::string, std::string> m;
  std::string get(const std::string& k, const std::string& def = "") const { auto it = m.find(k); return it == m.end() ? def : it->second; }
  int geti(const std::string& k, int def) const { auto it = m.find(k); return it == m.end() ? def : atoi(it->second.c_str()); }
  float getf(const std::string& k, float def) const { auto it = m.find(k); return it == m.end() ? def : (float)atof(it->second.c_str()); }
};
static Args parse_args(int argc, char** argv, int start) {
  Args a;
  for (int i = start; i < argc; ++i) { std::string s = argv[i];
    if (s.rfind("--", 0) != 0) continue; s = s.substr(2);
    auto eq = s.find('=');
    if (eq != std::string::npos) a.m[s.substr(0, eq)] = s.substr(eq + 1);
    else if (i + 1 < argc && std::string(argv[i+1]).rfind("--", 0) != 0) a.m[s] = argv[++i];
    else a.m[s] = "1";
  }
  return a;
}

static const char* COCO[80] = {
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
  "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
  "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
  "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
  "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
  "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

static ProviderU load_model(const std::string& weights, std::vector<int64_t>& dep) {
  { std::ifstream f(DN + "depths.txt"); int64_t v; while (f >> v) dep.push_back(v); }
  if (dep.empty()) dep = {1,2,3,1,1,1,1,1};
  std::ifstream t(weights);
  if (t.good()) { printf("weights <- %s (pure C++)\n", weights.c_str()); return load_net_unfused_pt(DN, weights); }
  return load_net_unfused(DN);
}

// run val mAP over a Dataset (letterbox eval, format-aware GT) — yolov5 anchor decode.
static double run_val(Dataset& va, ProviderU& prov, const std::vector<int64_t>& dep, int64_t S) {
  auto anchors = rd(DN + "anchors.bin"); std::vector<int> strides = {8,16,32};
  std::vector<mapeval::Image> imgs;
  for (auto& s : va.items) {
    Letterbox lb; auto xi = load_image_letterbox(s.img, S, lb);
    prov.i = 0; auto hv = yolov5n_forward_u(xi, prov, false, dep);
    int64_t N; auto pred = decode5(hv, anchors, strides, NA, NO, NC0, N);
    auto dets = nms5(pred, N, NO, NC0, 0.001f, 0.6f, 300);
    mapeval::Image im;
    for (auto& d : dets) im.dts.push_back({d.x1,d.y1,d.x2,d.y2,d.cls,d.conf});
    std::vector<float> gb; std::vector<int64_t> gl; int m = load_boxes_orig(s.lbl, va.yolo, lb.w0, lb.h0, gb, gl); lb_map(gb, lb);
    for (int j = 0; j < m; ++j) im.gts.push_back({gb[j*4],gb[j*4+1],gb[j*4+2],gb[j*4+3],(int)gl[j]});
    imgs.push_back(im);
  }
  auto mp = mapeval::coco_map(imgs);
  printf("val: mAP@0.5 %.4f   mAP@0.5:0.95 %.4f   (%zu images)\n", mp.first, mp.second, va.items.size());
  return mp.first;
}

// ============================== train ==============================
static int cmd_train(const Args& a) {
  DataYaml dy = parse_yaml(a.get("data"));
  int64_t S = a.geti("imgsz", 640);
  int EPOCHS = a.geti("epochs", 100), BATCH = a.geti("batch", 16);
  std::string weights = a.get("weights", "yolov5n.pt");
  AugCfg baseAug; baseAug.mosaic = a.geti("mosaic", 1) != 0; baseAug.mixup = a.geti("mixup", 1) != 0;
  baseAug.hsv = a.geti("hsv", 1) != 0; baseAug.affine = a.geti("affine", 1) != 0; baseAug.flip = a.geti("flip", 1) != 0;
  int closeMosaic = a.geti("close-mosaic", std::max(1, EPOCHS/10));
  if (dy.nc != NC0) printf("[warn] data.yaml nc=%d but the head is %lld-class; class ids must be < %lld "
                           "(custom-nc head re-init is a RESUME item)\n", dy.nc, (long long)NC0, (long long)NC0);

  Dataset tr = read_yolo_dataset(join(dy.path, dy.train), S);
  Dataset va = read_yolo_dataset(join(dy.path, dy.val),   S);
  printf("train=%zu val=%zu imgsz=%lld batch=%d epochs=%d mosaic=%d mixup=%d close-mosaic=%d\n",
         tr.items.size(), va.items.size(), (long long)S, BATCH, EPOCHS, (int)baseAug.mosaic, (int)baseAug.mixup, closeMosaic);

  std::vector<int64_t> dep; auto prov = load_model(weights, dep);
  auto anchors = rd(DN + "anchors.bin"); std::vector<int64_t> grids = {S/8, S/16, S/32};
  std::vector<Tensor> params; for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b); }
  float lr0 = a.getf("lr", 1e-3f);                    // fine-tune default; 2e-3 destroys pretrained
  int warmup = a.geti("warmup", std::max(1, ((int)tr.items.size()+BATCH-1)/BATCH));  // ~1 epoch
  Adam opt(params, lr0, 0.9f, 0.999f, 1e-8f, 5e-4f, false);

  std::vector<std::string> names; { std::ifstream f(DN + "names.txt"); std::string s; while (f >> s) names.push_back(s); }
  auto save_ckpt = [&](const std::string& path) {
    std::vector<pt::Tensor> ck; size_t k = 0;
    auto push = [&](const std::vector<float>& d, std::vector<int64_t> shp){ pt::Tensor t; if (k<names.size()) t.name=names[k]; t.shape=shp; t.data=d; ck.push_back(t); ++k; };
    for (auto& L : prov.layers) { std::vector<int64_t> ws(L.w->shape.begin(),L.w->shape.end()); push(L.w->data, ws);
      if (L.kind==1){ std::vector<int64_t> c={L.gamma->shape[0]}; push(L.gamma->data,c); push(L.beta->data,c); push(L.rm,c); push(L.rv,c); }
      else push(L.b->data, {L.b->shape[0]}); }
    pt::save_pt(ck, path);
  };

  std::vector<int> order(tr.items.size()); std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(0);
  int total = EPOCHS * (((int)tr.items.size()+BATCH-1)/BATCH), gstep = 0; double best = -1;
  for (int ep = 0; ep < EPOCHS; ++ep) {
    AugCfg aug = baseAug; if (ep >= EPOCHS - closeMosaic) { aug.mosaic = false; aug.mixup = false; }
    std::shuffle(order.begin(), order.end(), rng); double eloss = 0; int nb = 0;
    for (size_t off = 0; off < order.size(); off += BATCH) {
      std::vector<int> idx(order.begin()+off, order.begin()+std::min(order.size(), off+BATCH));
      Batch bt = load_minibatch(tr, idx, true, rng(), aug);
      int64_t B = bt.B, Mx = bt.M;
      std::vector<float> targets; int NT = 0;
      for (int64_t b=0;b<B;++b) for (int64_t m=0;m<Mx;++m) if (bt.mask[b*Mx+m] > 0) {
        float x1=bt.gt_boxes[(b*Mx+m)*4],y1=bt.gt_boxes[(b*Mx+m)*4+1],x2=bt.gt_boxes[(b*Mx+m)*4+2],y2=bt.gt_boxes[(b*Mx+m)*4+3];
        targets.insert(targets.end(), {(float)b,(float)bt.gt_labels[b*Mx+m], ((x1+x2)/2)/S, ((y1+y2)/2)/S, (x2-x1)/(float)S, (y2-y1)/(float)S}); ++NT;
      }
      prov.i=0; auto heads = yolov5n_forward_u(bt.x, prov, true, dep);
      std::vector<Tensor> p; for (auto& h : heads) p.push_back(head_to_pred(h, NA, NO));
      auto L = compute_loss_v5(p, targets, NT, anchors, grids, B, NA, NO, NC0);
      backward(L.total);
      opt.lr = cosine_lr(gstep, total, lr0, warmup); opt.step(); ++gstep;
      eloss += L.total->data[0]; ++nb;
    }
    Dataset vac = va; double m50 = run_val(vac, prov, dep, S);
    printf("epoch %d/%d  loss %.3f  (val above)%s\n", ep+1, EPOCHS, eloss/nb, m50>best?"  *best*":"");
    save_ckpt("last.pt"); if (m50 > best) { best = m50; save_ckpt("best.pt"); }
  }
  printf("done. best val mAP@0.5 = %.4f. wrote last.pt / best.pt (pure C++)\n", best);
  return 0;
}

// ============================== val ==============================
static int cmd_val(const Args& a) {
  DataYaml dy = parse_yaml(a.get("data"));
  int64_t S = a.geti("imgsz", 640);
  std::vector<int64_t> dep; auto prov = load_model(a.get("weights", "yolov5n.pt"), dep);
  Dataset va = read_yolo_dataset(join(dy.path, dy.val), S);
  run_val(va, prov, dep, S);
  return 0;
}

// ============================== detect ==============================
static int cmd_detect(const Args& a) {
  DataYaml dy; if (!a.get("data").empty()) dy = parse_yaml(a.get("data"));
  int64_t S = a.geti("imgsz", 640); float conf = a.getf("conf", 0.25f), iou = a.getf("iou", 0.45f);
  std::string src = a.get("source"), outp = a.get("out", "out.png");
  std::vector<int64_t> dep; auto prov = load_model(a.get("weights", "yolov5n.pt"), dep);
  auto anchors = rd(DN + "anchors.bin"); std::vector<int> strides = {8,16,32};
  int w0,h0,ch; unsigned char* im = stbi_load(src.c_str(), &w0,&h0,&ch,3);
  if (!im) { printf("cannot load %s\n", src.c_str()); return 1; }
  Letterbox lb; auto x = load_image_letterbox(src, S, lb);
  prov.i = 0; auto hv = yolov5n_forward_u(x, prov, false, dep);
  int64_t N; auto pred = decode5(hv, anchors, strides, NA, NO, NC0, N);
  auto dets = nms5(pred, N, NO, NC0, conf, iou, 300);
  auto put=[&](int px,int py,unsigned char r,unsigned char g,unsigned char b){ if(px<0||py<0||px>=w0||py>=h0)return; unsigned char*p=&im[(py*w0+px)*3];p[0]=r;p[1]=g;p[2]=b; };
  printf("%zu detections:\n", dets.size());
  for (auto& d : dets) {
    int x1=(int)std::round((d.x1-lb.left)/lb.r), y1=(int)std::round((d.y1-lb.top)/lb.r);
    int x2=(int)std::round((d.x2-lb.left)/lb.r), y2=(int)std::round((d.y2-lb.top)/lb.r);
    x1=std::clamp(x1,0,w0-1);y1=std::clamp(y1,0,h0-1);x2=std::clamp(x2,0,w0-1);y2=std::clamp(y2,0,h0-1);
    for(int t=0;t<3;++t){for(int px=x1;px<=x2;++px){put(px,y1+t,255,60,60);put(px,y2-t,255,60,60);}for(int py=y1;py<=y2;++py){put(x1+t,py,255,60,60);put(x2-t,py,255,60,60);}}
    const char* nm = (d.cls < (int)dy.names.size()) ? dy.names[d.cls].c_str() : (d.cls<80?COCO[d.cls]:"?");
    printf("  %-14s conf=%.2f  xyxy=(%d,%d,%d,%d)\n", nm, d.conf, x1,y1,x2,y2);
  }
  if (!stbi_write_png(outp.c_str(), w0,h0,3, im, w0*3)) { printf("write failed\n"); return 1; }
  printf("wrote %s\n", outp.c_str()); stbi_image_free(im); return 0;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::string cmd = argc > 1 ? argv[1] : "";
  Args a = parse_args(argc, argv, 2);
  if (cmd == "train")  return cmd_train(a);
  if (cmd == "val")    return cmd_val(a);
  if (cmd == "detect") return cmd_detect(a);
  if (cmd == "export") { printf("export: use `onnx_export <model>` (onnxruntime-verified). "
                                "Unifying ONNX-export-from-.pt into this CLI is a RESUME item.\n"); return 0; }
  printf("usage: yolo <train|val|detect|export> --flags   (see header of pure/yolo.cpp)\n");
  return 1;
}
