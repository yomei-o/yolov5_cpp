// yolov5_cpp M1: full yolov5n forward in the pure engine, checked against the real net.
#include "net5.hpp"
#include <cstdio>

int main() {
  const std::string D = "pure/ref/data_net/";
  auto prov = load_net(D);
  std::ifstream io(D + "io.txt"); int64_t IMG; io >> IMG;
  auto x = from_data({1, 3, IMG, IMG}, rd(D + "x.bin"));

  auto heads = yolov5n_forward(x, prov);
  printf("consumed %zu/%zu convs, %zu head levels\n", prov.i, prov.convs.size(), heads.size());

  // pack the 3 head outputs (1,C,ny,nx) channel-major per level, concatenated
  std::vector<float> out;
  for (auto& h : heads) out.insert(out.end(), h->data.begin(), h->data.end());
  auto ref = rd(D + "ref_head.bin");
  if (out.size() != ref.size()) { printf("size mismatch %zu vs %zu\n", out.size(), ref.size()); return 1; }
  double d = 0; for (size_t i = 0; i < out.size(); ++i) d = std::max(d, (double)std::abs(out[i] - ref[i]));
  printf("head max|diff| = %.3e  %s\n", d, d < 1e-3 ? "OK" : "FAIL");
  printf("\n%s\n", d < 1e-3 ? "yolov5 M1: PURE ENGINE == yolov5n forward" : "MISMATCH");
  return d < 1e-3 ? 0 : 1;
}
