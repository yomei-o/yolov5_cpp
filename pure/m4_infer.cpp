// yolov5_cpp M4: pure-engine inference. forward -> anchor decode -> NMS, checked against
// yolov5's eval output (decoded predictions) and non_max_suppression.
#include "net5.hpp"
#include "infer5.hpp"
#include <cstdio>
#include <fstream>

int main() {
  const std::string DN = "pure/ref/data_net/", DI = "pure/ref/data_infer/";
  std::ifstream mf(DI + "meta.txt");
  if (!mf) { printf("run: python pure/ref/infer_ref5.py\n"); return 1; }
  int64_t S, nc, N; float conf, iou; mf >> S >> nc >> N >> conf >> iou;
  float w0, h0, r, dw, dh; mf >> w0 >> h0 >> r >> dw >> dh;
  int ndet; mf >> ndet;
  std::vector<Det5> refd(ndet);
  for (int i = 0; i < ndet; ++i) mf >> refd[i].x1 >> refd[i].y1 >> refd[i].x2 >> refd[i].y2 >> refd[i].conf >> refd[i].cls;

  std::ifstream io(DN + "io.txt"); int64_t IMGe, nce, na, no, nl; io >> IMGe >> nce >> na >> no >> nl;
  auto anchors = rd(DN + "anchors.bin");
  auto prov = load_net(DN);
  auto x = from_data({1, 3, S, S}, rd(DI + "x.bin"));

  printf("forward (%lldx%lld, im2col+GEMM)...\n", (long long)S, (long long)S);
  prov.i = 0;
  auto heads = yolov5n_forward(x, prov);
  std::vector<int> strides = {8, 16, 32};
  int64_t Nn;
  auto pred = decode5(heads, anchors, strides, na, no, nc, Nn);

  // 1) decoded predictions vs yolov5 eval output
  auto refp = rd(DI + "ref_pred.bin");
  double dmax = 0; for (size_t i = 0; i < pred.size(); ++i) dmax = std::max(dmax, (double)std::abs(pred[i] - refp[i]));
  printf("decode max|diff| vs yolov5 = %.3e  (N=%lld)  %s\n", dmax, (long long)Nn, dmax < 1e-2 ? "OK" : "FAIL");

  // 2) NMS vs yolov5
  auto dets = nms5(pred, Nn, no, nc, conf, iou, 300);
  std::sort(dets.begin(), dets.end(), [](const Det5& a, const Det5& b){ return a.conf > b.conf; });
  std::sort(refd.begin(), refd.end(), [](const Det5& a, const Det5& b){ return a.conf > b.conf; });
  printf("NMS: pure=%zu  yolov5=%d\n", dets.size(), ndet);
  bool match = (int)dets.size() == ndet;
  double berr = 0, cerr = 0;
  for (size_t i = 0; i < dets.size() && i < refd.size(); ++i) {
    if (dets[i].cls != refd[i].cls) match = false;
    berr = std::max({berr, (double)std::abs(dets[i].x1-refd[i].x1), (double)std::abs(dets[i].y1-refd[i].y1),
                           (double)std::abs(dets[i].x2-refd[i].x2), (double)std::abs(dets[i].y2-refd[i].y2)});
    cerr = std::max(cerr, (double)std::abs(dets[i].conf-refd[i].conf));
  }
  printf("det box max|diff| = %.3e  conf max|diff| = %.3e\n", berr, cerr);
  for (auto& d : dets) printf("  cls %2d conf=%.3f xyxy=(%.1f,%.1f,%.1f,%.1f)\n", d.cls, d.conf, d.x1, d.y1, d.x2, d.y2);
  bool ok = dmax < 1e-2 && match && berr < 1.0 && cerr < 1e-2;
  printf("\n%s\n", ok ? "yolov5 M4: PURE INFERENCE == yolov5 (decode + NMS)" : "MISMATCH");
  return ok ? 0 : 1;
}
