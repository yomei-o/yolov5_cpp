// yolov5_cpp M2: verify the pure anchor-based loss (forward + backward) against the
// original yolov5 ComputeLoss — loss components and the gradient dLoss/dp per level.
#include "loss5.hpp"
#include "net5.hpp"      // rd()
#include <cstdio>

int main() {
  const std::string D = "pure/ref/data_loss/";
  std::ifstream mf(D + "meta.txt");
  if (!mf) { printf("run: python pure/ref/loss_ref.py\n"); return 1; }
  int64_t BS, S, nl, na, no, nc, NT; mf >> BS >> S >> nl >> na >> no >> nc >> NT;
  std::vector<int64_t> grids(nl); for (auto& g : grids) mf >> g;

  auto targets = rd(D + "targets.bin");
  auto anchors = rd(D + "anchors.bin");            // (nl*na*2)
  std::vector<Tensor> p(nl);
  for (int64_t i = 0; i < nl; ++i)
    p[i] = from_data({BS, na, grids[i], grids[i], no}, rd(D + "p" + std::to_string(i) + ".bin"), true);

  auto L = compute_loss_v5(p, targets, (int)NT, anchors, grids, BS, na, no, nc);
  backward(L.total);

  auto ref = rd(D + "loss.bin");                   // total, lbox, lobj, lcls
  printf("component     pure        yolov5\n");
  printf("total     %10.6f  %10.6f\n", L.total->data[0], ref[0]);
  printf("lbox      %10.6f  %10.6f\n", L.lbox->data[0], ref[1]);
  printf("lobj      %10.6f  %10.6f\n", L.lobj->data[0], ref[2]);
  printf("lcls      %10.6f  %10.6f\n", L.lcls->data[0], ref[3]);
  double lerr = std::max({std::abs(L.total->data[0]-ref[0]), std::abs(L.lbox->data[0]-ref[1]),
                          std::abs(L.lobj->data[0]-ref[2]), std::abs(L.lcls->data[0]-ref[3])});

  double gerr = 0;
  for (int64_t i = 0; i < nl; ++i) {
    auto dref = rd(D + "dp" + std::to_string(i) + ".bin");
    for (size_t k = 0; k < dref.size(); ++k) gerr = std::max(gerr, (double)std::abs(p[i]->grad[k] - dref[k]));
  }
  printf("\nloss max|diff| = %.3e   grad max|diff| = %.3e\n", lerr, gerr);
  bool ok = lerr < 1e-4 && gerr < 1e-6;
  printf("\n%s\n", ok ? "yolov5 M2: PURE loss == yolov5 ComputeLoss (fwd + bwd)" : "MISMATCH");
  return ok ? 0 : 1;
}
