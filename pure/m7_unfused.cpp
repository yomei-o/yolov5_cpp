// yolov5_cpp: verify the UNFUSED forward (conv + BatchNorm2d(eval) + SiLU) matches
// yolov5n — same reference head outputs as m1, but BN is a live op.
#include "net5_unfused.hpp"
#include <cstdio>

int main() {
  const std::string D = "pure/ref/data_net/";
  auto prov = load_net_unfused(D);
  std::ifstream io(D + "io.txt"); int64_t IMG; io >> IMG;
  auto x = from_data({1, 3, IMG, IMG}, rd(D + "x.bin"));
  auto heads = yolov5n_forward_u(x, prov, false);
  std::vector<float> out; for (auto& h : heads) out.insert(out.end(), h->data.begin(), h->data.end());
  auto ref = rd(D + "ref_head.bin");
  double d = 0; for (size_t i = 0; i < out.size(); ++i) d = std::max(d, (double)std::abs(out[i] - ref[i]));
  printf("consumed %zu/%zu layers, head max|diff| = %.3e  %s\n", prov.i, prov.layers.size(), d, d < 1e-3 ? "OK" : "FAIL");
  printf("\n%s\n", d < 1e-3 ? "yolov5 unfused: conv+BN+SiLU == yolov5n forward" : "MISMATCH");
  return d < 1e-3 ? 0 : 1;
}
