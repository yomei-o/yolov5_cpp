// yolov5 device forward parity vs the trusted CPU engine (yolov5n_forward_u), train-mode
// (batch-stat BN) so no folding needed. Same weights + input; per-level head max|diff|.
#include "net5_unfused.hpp"
#include "dnet5.hpp"
#include <cstdio>
#include <cmath>
static float md(const std::vector<float>& a, const Tensor& b){ float m=0; for(size_t i=0;i<a.size();++i) m=std::max(m,std::abs(a[i]-b->data[i])); return m; }
int main(){
  const std::string DN="pure/ref/data_net/"; const int64_t S=64;
  ProviderU pu = load_net_unfused_pt(DN, "yolov5n.pt");
  ProvD5 pd = dnet5_build(DN, "yolov5n.pt");
  auto dep = dnet5_depths(DN);
  std::vector<float> xh(1*3*S*S); for(size_t i=0;i<xh.size();++i) xh[i]=std::sin(0.05f*i)*0.5f+0.1f;
  pu.i=0; auto cpu = yolov5n_forward_u(from_data({1,3,S,S},xh), pu, true, dep);
  pd.i=0; auto dev = dnet5_forward(dfrom({1,3,S,S},xh), pd, dep, true); bk::sync();
  float worst=0;
  for(int l=0;l<3;++l){ float d=md(dto_host(dev[l]), cpu[l]);
    printf("  level %d: head[%lldx%lldx%lld] max|d|=%.3e\n", l, (long long)cpu[l]->shape[1],(long long)cpu[l]->shape[2],(long long)cpu[l]->shape[3], d); worst=std::max(worst,d); }
  printf("yolov5n forward: worst |device - CPU-engine| = %.3e  %s\n", worst, worst<2e-3f?"MATCH":"MISMATCH");
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
