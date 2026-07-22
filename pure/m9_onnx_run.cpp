// yolov5_cpp: load yolov5n.onnx and run it purely graph-driven (parse + interpret with
// pure ops), no topology/weights hardcoded. Verify vs the reference forward.
#include "onnx_run.hpp"
#include "net5.hpp"       // rd()
#include <cstdio>

int main() {
  const std::string D = "pure/ref/data_net/";
  auto g = onx::load_onnx("yolov5n.onnx");
  std::ifstream io(D + "io.txt"); int64_t IMG; io >> IMG;
  auto x = from_data({1, 3, IMG, IMG}, rd(D + "x.bin"));
  printf("yolov5n.onnx: %zu nodes, %zu float inits\n", g.nodes.size(), g.init_f.size());
  auto vals = onx::run_onnx(g, x);
  std::vector<float> out;
  for (int i = 0; i < 3; ++i) { auto t = vals.at("out" + std::to_string(i)); out.insert(out.end(), t->data.begin(), t->data.end()); }
  auto ref = rd(D + "ref_head.bin");
  double d = 0; for (size_t i = 0; i < out.size(); ++i) d = std::max(d, (double)std::abs(out[i] - ref[i]));
  printf("head max|diff| = %.3e  %s\n", d, d < 1e-3 ? "OK" : "FAIL");
  printf("\n%s\n", d < 1e-3 ? "yolov5 ONNX(read): pure engine runs yolov5n.onnx == forward" : "MISMATCH");
  return d < 1e-3 ? 0 : 1;
}
