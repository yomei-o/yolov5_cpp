// yolov5_cpp M3: end-to-end training of yolov5n in the pure engine. forward -> head
// reshape -> anchor-based v5 loss -> backward -> Adam (cosine LR). Loss must drop.
#include "net5.hpp"
#include "loss5.hpp"
#include "optim.hpp"
#include <cstdio>
#include <random>

int main(int argc, char** argv) {
  const int ITERS = argc > 1 ? atoi(argv[1]) : 30;
  const std::string D = "pure/ref/data_net/";
  auto prov = load_net(D);
  std::vector<Tensor> params;
  for (auto& c : prov.convs) { params.push_back(c.w); params.push_back(c.b); }
  Adam opt(params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.f, false);

  std::ifstream io(D + "io.txt"); int64_t IMG, nc, na, no, nl; io >> IMG >> nc >> na >> no >> nl;
  auto anchors = rd(D + "anchors.bin");
  const int64_t BS = 2, S = IMG;
  std::vector<int64_t> grids = {S/8, S/16, S/32};

  // fixed synthetic batch + a few targets [img,cls,xn,yn,wn,hn]
  auto img = make_tensor({BS, 3, S, S});
  { std::mt19937 rng(0); std::normal_distribution<float> nd(0,1); for (auto& v : img->data) v = nd(rng); }
  std::vector<float> targets = {
    0, 12, 0.30f,0.40f,0.35f,0.55f,  0, 40, 0.65f,0.30f,0.25f,0.40f,  0, 7, 0.50f,0.70f,0.30f,0.35f,
    1,  3, 0.40f,0.45f,0.45f,0.60f,  1, 25, 0.60f,0.55f,0.30f,0.30f };
  int NT = (int)targets.size() / 6;

  printf("iter |   total     box      obj      cls      lr\n");
  for (int it = 0; it < ITERS; ++it) {
    prov.i = 0;
    auto heads = yolov5n_forward(img, prov);
    std::vector<Tensor> p;
    for (auto& h : heads) p.push_back(head_to_pred(h, na, no));
    auto L = compute_loss_v5(p, targets, NT, anchors, grids, BS, na, no, nc);
    backward(L.total);
    opt.lr = cosine_lr(it, ITERS, 1e-3f, 3);
    opt.step();
    if (it % 5 == 0 || it == ITERS - 1)
      printf("%4d | %8.4f %8.4f %8.4f %8.4f  %.2e\n", it, L.total->data[0],
             L.lbox->data[0], L.lobj->data[0], L.lcls->data[0], opt.lr);
  }
  printf("done — trained yolov5n in the pure engine.\n");
  return 0;
}
