// End-to-end test: train yolov5n on a small synthetic dataset (image files + labels),
// then run inference on a held-out image and draw the detections + dump trained weights.
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure/m10_synth.cpp
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "net5_unfused.hpp"
#include "loss5.hpp"
#include "optim.hpp"
#include "infer5.hpp"
#include <cstdio>
#include <string>
#include <filesystem>

static const char* NM[3] = {"red", "green", "blue"};

int main(int argc, char** argv) {
  const int ITERS = argc > 1 ? atoi(argv[1]) : 120;
  const std::string DN = "pure/ref/data_net/", DS = "pure/ref/data_synth/";
  const int64_t NC = 80, NA = 3, NO = 85;

  Batch bt = load_batch(DS + "list.txt");
  int64_t B = bt.B, M = bt.M, S = bt.x->shape[2];
  auto anchors = rd(DN + "anchors.bin");
  std::vector<int64_t> grids = {S/8, S/16, S/32};
  printf("train batch: %lld images %lldpx\n", (long long)B, (long long)S);

  // targets for loss5: (NT,6) = [img, cls, xn, yn, wn, hn] normalized
  std::vector<float> targets; int NT = 0;
  for (int64_t b=0;b<B;++b) for (int64_t m=0;m<M;++m) if (bt.mask[b*M+m] > 0) {
    float x1=bt.gt_boxes[(b*M+m)*4],y1=bt.gt_boxes[(b*M+m)*4+1],x2=bt.gt_boxes[(b*M+m)*4+2],y2=bt.gt_boxes[(b*M+m)*4+3];
    targets.insert(targets.end(), {(float)b,(float)bt.gt_labels[b*M+m], ((x1+x2)/2)/S, ((y1+y2)/2)/S, (x2-x1)/(float)S, (y2-y1)/(float)S});
    ++NT;
  }

  auto prov = load_net_unfused(DN);
  std::vector<Tensor> params;
  for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b); }
  Adam opt(params, 2e-3f, 0.9f, 0.999f, 1e-8f, 0.f, false);

  printf("training %d iters (%d objects)...\n", ITERS, NT);
  for (int it=0; it<ITERS; ++it) {
    prov.i=0; auto heads = yolov5n_forward_u(bt.x, prov, true);
    std::vector<Tensor> p; for (auto& h : heads) p.push_back(head_to_pred(h, NA, NO));
    auto L = compute_loss_v5(p, targets, NT, anchors, grids, B, NA, NO, NC);
    backward(L.total); opt.lr = cosine_lr(it, ITERS, 2e-3f, 5); opt.step();
    if (it%25==0 || it==ITERS-1) printf("  iter %3d  loss %7.3f\n", it, L.total->data[0]);
  }
  printf("training done.\n\n");

  std::filesystem::create_directories("pure/ref/data_wb/");
  auto wr=[&](const std::string&n,const std::vector<float>&v){std::ofstream f("pure/ref/data_wb/"+n,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(float));};
  for (size_t i=0;i<prov.layers.size();++i){auto&L=prov.layers[i];std::string s=std::to_string(i);wr("cw"+s+".bin",L.w->data);if(L.kind==1){wr("bg"+s+".bin",L.gamma->data);wr("bb"+s+".bin",L.beta->data);wr("rm"+s+".bin",L.rm);wr("rv"+s+".bin",L.rv);}else wr("cb"+s+".bin",L.b->data);}
  printf("wrote trained weights to pure/ref/data_wb/\n\n");

  std::string timg = DS + "te00.png";
  int w0,h0,ch; unsigned char* im = stbi_load(timg.c_str(), &w0,&h0,&ch, 3);
  auto x = make_tensor({1,3,S,S});
  for (int c=0;c<3;++c) for (int y=0;y<S;++y) for (int xx=0;xx<S;++xx) x->data[(c*S+y)*S+xx]=im[(y*w0+xx)*3+c]/255.f;
  prov.i=0; auto hv = yolov5n_forward_u(x, prov, false);
  std::vector<int> strides={8,16,32}; int64_t N; auto pred = decode5(hv, anchors, strides, NA, NO, NC, N);
  auto dets = nms5(pred, N, NO, NC, 0.25f, 0.45f, 50);
  printf("inference on te00.png: %zu detections\n", dets.size());
  auto putp=[&](int px,int py){if(px<0||py<0||px>=w0||py>=h0)return;unsigned char*p=&im[(py*w0+px)*3];p[0]=255;p[1]=255;p[2]=0;};
  for (auto& d : dets){ printf("  cls %d(%s) conf %.2f xyxy=(%.0f,%.0f,%.0f,%.0f)\n", d.cls, d.cls<3?NM[d.cls]:"?", d.conf, d.x1,d.y1,d.x2,d.y2);
    for(int t=0;t<2;++t){for(int px=(int)d.x1;px<=(int)d.x2;++px){putp(px,(int)d.y1+t);putp(px,(int)d.y2-t);}for(int py=(int)d.y1;py<=(int)d.y2;++py){putp((int)d.x1+t,py);putp((int)d.x2-t,py);}}}
  stbi_write_png("synth_det.png", w0,h0,3,im,w0*3); printf("wrote synth_det.png\n");
  return 0;
}
