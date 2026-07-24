// Device-resident yolov5 training over a standard-YOLO dataset (e.g. COCO128), with
// checkpoint save. Device fwd(train BN, EMA running stats)+bwd+Adam; trusted host anchor
// loss (head_to_pred + compute_loss_v5) bridged in. Saves last.pt/best.pt. Times s/epoch.
//   run: dtrain_coco5 <images_dir> <imgsz> <batch> <epochs>
//   GPU: nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA [-DUSE_CUBLAS -lcublas] -Ipure/third_party pure/dtrain_coco5.cpp -o dtrain_coco5
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"
#include "dnet5.hpp"           // device yolov5 + build/forward/params/save
#include "loss5.hpp"           // head_to_pred, compute_loss_v5 (trusted host)
#include "net5.hpp"            // rd()
#include <cstdio>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::string dir = argc>1?argv[1]:"pure/ref/data_yolo/images/train";
  int64_t S = argc>2?atoll(argv[2]):96;
  int BATCH = argc>3?atoi(argv[3]):4, EPOCHS = argc>4?atoi(argv[4]):2;
  std::string model = argc>5?argv[5]:"yolov5n";
  const std::string DN = (model=="yolov5n") ? "pure/ref/data_net/" : "pure/ref/arch/"+model+"/";
  const std::string weights = model + ".pt";
  const int64_t NC = 80, NA = 3, NO = 85;

  ProvD5 prov = dnet5_build(DN, weights);
  std::vector<DT> params = dnet5_params(prov);
  DAdam opt(params, 1e-3f);
  auto dep = dnet5_depths(DN);
  auto anchors = rd(DN + "anchors.bin");
  Dataset tr = read_yolo_dataset(dir, S);
  std::vector<int64_t> grids = {S/8, S/16, S/32};
  printf("%s train=%zu imgsz=%lld batch=%d epochs=%d\n", model.c_str(), tr.items.size(), (long long)S, BATCH, EPOCHS);

  std::vector<int> order(tr.items.size()); std::iota(order.begin(),order.end(),0); std::mt19937 rng(0);
  double best = 1e30;

  for (int ep=0; ep<EPOCHS; ++ep) {
    std::shuffle(order.begin(),order.end(),rng); double eloss=0; int nb=0;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t off=0; off<order.size(); off+=BATCH) {
      std::vector<int> idx(order.begin()+off, order.begin()+std::min(order.size(),off+BATCH));
      Batch bt = load_minibatch(tr, idx, false, rng());
      int64_t B=bt.B, M=bt.M;
      std::vector<float> targets; int NT=0;                     // (NT,6) [img,cls,xn,yn,wn,hn]
      for (int64_t b=0;b<B;++b) for (int64_t m=0;m<M;++m) if (bt.mask[b*M+m]>0) {
        float x1=bt.gt_boxes[(b*M+m)*4],y1=bt.gt_boxes[(b*M+m)*4+1],x2=bt.gt_boxes[(b*M+m)*4+2],y2=bt.gt_boxes[(b*M+m)*4+3];
        targets.insert(targets.end(), {(float)b,(float)bt.gt_labels[b*M+m], ((x1+x2)/2)/S, ((y1+y2)/2)/S, (x2-x1)/(float)S, (y2-y1)/(float)S}); ++NT;
      }
      opt.zero_grad();
      prov.i=0; auto dev = dnet5_forward(dfrom({B,3,S,S}, bt.x->data), prov, dep, true);   // DEVICE
      std::vector<Tensor> heads_cpu;                            // CPU leaf mirrors of head outputs
      for (auto& h : dev) heads_cpu.push_back(from_data({h->shape[0],h->shape[1],h->shape[2],h->shape[3]}, dto_host(h), true));
      std::vector<Tensor> p; for (auto& h : heads_cpu) p.push_back(head_to_pred(h, NA, NO));
      auto L = compute_loss_v5(p, targets, NT, anchors, grids, B, NA, NO, NC);   // TRUSTED host loss
      backward(L.total);
      for (size_t l=0;l<dev.size();++l) thrust::copy(heads_cpu[l]->grad.begin(), heads_cpu[l]->grad.end(), dev[l]->grad.begin());
      dbackward_from(dev);                                      // DEVICE backward through the net
      opt.step();
      eloss += L.total->data[0]; ++nb;
    }
    bk::sync();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double avg = eloss/nb;
    dnet5_save(prov, DN, "last.pt");
    if (avg < best) { best = avg; dnet5_save(prov, DN, "best.pt"); }
    printf("epoch %d/%d  loss %.4f  %.1f s/epoch%s\n", ep+1, EPOCHS, avg, secs, avg<=best?"  *best*":"");
  }
  printf("done. best loss %.4f. wrote last.pt / best.pt (pure C++, %s)\n", best, model.c_str());
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
