// yolov5_cpp: train yolov5n in the pure engine with conv+BN separate, then write the
// updated weights (conv, BN params) back to flat .bin so a Python bridge can drop them
// into yolov5n.pt. Also dumps an eval-mode forward so the round-trip can be checked.
//   run: m8_train_writeback [iters]   (0 = round-trip check on original weights)
#include "net5_unfused.hpp"
#include "loss5.hpp"
#include <fstream>
#include "optim.hpp"
#include <cstdio>
#include <random>
#include <filesystem>

int main(int argc, char** argv) {
  const int ITERS = argc > 1 ? atoi(argv[1]) : 20;
  const std::string D = "pure/ref/data_net/", D2 = "pure/ref/data_wb/";
  std::filesystem::create_directories(D2);
  auto prov = load_net_unfused(D);
  std::vector<int64_t> dep; { std::ifstream f(D + "depths.txt"); int64_t v; while (f >> v) dep.push_back(v); }
  std::vector<Tensor> params;
  for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind == 1) { params.push_back(L.gamma); params.push_back(L.beta); } else params.push_back(L.b); }
  Adam opt(params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.f, false);

  std::ifstream io(D + "io.txt"); int64_t IMG, nc, na, no, nl; io >> IMG >> nc >> na >> no >> nl;
  auto anchors = rd(D + "anchors.bin");
  const int64_t BS = 2, S = IMG;
  std::vector<int64_t> grids = {S/8, S/16, S/32};
  auto img = make_tensor({BS, 3, S, S});
  { std::mt19937 rng(0); std::normal_distribution<float> nd(0,1); for (auto& v : img->data) v = nd(rng); }
  std::vector<float> targets = {0,12,0.30f,0.40f,0.35f,0.55f, 0,40,0.65f,0.30f,0.25f,0.40f, 0,7,0.50f,0.70f,0.30f,0.35f,
                                1,3,0.40f,0.45f,0.45f,0.60f, 1,25,0.60f,0.55f,0.30f,0.30f};
  int NT = (int)targets.size() / 6;

  printf("iter |   total     box      obj      cls\n");
  for (int it = 0; it < ITERS; ++it) {
    prov.i = 0;
    auto heads = yolov5n_forward_u(img, prov, true, dep);
    std::vector<Tensor> p; for (auto& h : heads) p.push_back(head_to_pred(h, na, no));
    auto L = compute_loss_v5(p, targets, NT, anchors, grids, BS, na, no, nc);
    backward(L.total);
    opt.lr = cosine_lr(it, ITERS, 1e-3f, 3); opt.step();
    if (it % 5 == 0) printf("%4d | %8.4f %8.4f %8.4f %8.4f\n", it, L.total->data[0], L.lbox->data[0], L.lobj->data[0], L.lcls->data[0]);
  }
  printf("training done (%d iters).\n", ITERS);

  auto wr = [&](const std::string& n, const std::vector<float>& v) { std::ofstream f(D2 + n, std::ios::binary); f.write((const char*)v.data(), v.size()*sizeof(float)); };
  for (size_t i = 0; i < prov.layers.size(); ++i) {
    auto& L = prov.layers[i]; std::string s = std::to_string(i);
    wr("cw"+s+".bin", L.w->data);
    if (L.kind == 1) { wr("bg"+s+".bin", L.gamma->data); wr("bb"+s+".bin", L.beta->data); wr("rm"+s+".bin", L.rm); wr("rv"+s+".bin", L.rv); }
    else wr("cb"+s+".bin", L.b->data);
  }
  auto x = from_data({1, 3, S, S}, rd(D + "x.bin"));
  prov.i = 0; auto hv = yolov5n_forward_u(x, prov, false, dep);
  std::vector<float> head; for (auto& h : hv) head.insert(head.end(), h->data.begin(), h->data.end());
  wr("cpp_head.bin", head);
  printf("wrote updated weights + eval forward to %s\n", D2.c_str());
  return 0;
}
