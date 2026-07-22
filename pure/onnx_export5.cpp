// yolov5_cpp: export the yolov5n forward to a standard .onnx (opset 13) using the
// hand-rolled protobuf writer (no deps). Ops: Conv, Sigmoid+Mul (SiLU), MaxPool (SPPF),
// Resize (upsample), Concat, Add (C3 shortcut). Outputs the 3 raw detect-head tensors.
//   run: onnx_export5   (needs pure/ref/data_net from export_yolov5.py)
#include "net5.hpp"
#include "onnx.hpp"
#include <string>
using namespace onx;

int main() {
  const std::string D = "pure/ref/data_net/";
  auto prov = load_net(D);
  std::ifstream io(D + "io.txt"); int64_t IMG, nc, na, no, nl; io >> IMG >> nc >> na >> no >> nl;

  Graph g; g.opset = 13;
  g.inputs.push_back({"images", {1, 3, IMG, IMG}});
  int uid = 0;
  auto uniq = [&](const char* p) { return std::string(p) + std::to_string(uid++); };

  auto conv = [&](const std::string& in) -> std::string {
    ConvW& c = prov.next();
    int64_t Co = c.w->shape[0], Ci = c.w->shape[1], k = c.w->shape[2];
    std::string wn = uniq("w"), bn = uniq("b"), yn = uniq("conv");
    g.init_f.push_back({wn, {Co, Ci, k, k}, c.w->data});
    g.init_f.push_back({bn, {Co}, c.b->data});
    onx::Node n; n.op_type="Conv"; n.name=yn; n.input={in,wn,bn}; n.output={yn};
    n.attr.push_back({"kernel_shape", A_INTS,0,0,"",{k,k},{}});
    n.attr.push_back({"strides", A_INTS,0,0,"",{c.stride,c.stride},{}});
    n.attr.push_back({"pads", A_INTS,0,0,"",{c.pad,c.pad,c.pad,c.pad},{}});
    n.attr.push_back({"group", A_INT,1,0,"",{},{}});
    g.nodes.push_back(n);
    if (!c.act) return yn;
    std::string sn=uniq("sig"), mn=uniq("silu");
    g.nodes.push_back({"Sigmoid", sn, {yn}, {sn}, {}});
    g.nodes.push_back({"Mul", mn, {yn, sn}, {mn}, {}});
    return mn;
  };
  auto add = [&](const std::string& a, const std::string& b){ std::string yn=uniq("add"); g.nodes.push_back({"Add",yn,{a,b},{yn},{}}); return yn; };
  auto concat = [&](const std::vector<std::string>& xs){ std::string yn=uniq("cat"); onx::Node n{"Concat",yn,xs,{yn},{}}; n.attr.push_back({"axis",A_INT,1,0,"",{},{}}); g.nodes.push_back(n); return yn; };
  auto maxpool = [&](const std::string& in){ std::string yn=uniq("mp"); onx::Node n{"MaxPool",yn,{in},{yn},{}};
    n.attr.push_back({"kernel_shape",A_INTS,0,0,"",{5,5},{}}); n.attr.push_back({"strides",A_INTS,0,0,"",{1,1},{}}); n.attr.push_back({"pads",A_INTS,0,0,"",{2,2,2,2},{}}); g.nodes.push_back(n); return yn; };
  auto resize2x = [&](const std::string& in){ std::string sc=uniq("scales"), yn=uniq("up"); g.init_f.push_back({sc,{4},{1.f,1.f,2.f,2.f}});
    onx::Node n{"Resize",yn,{in,"",sc},{yn},{}}; n.attr.push_back({"mode",A_STRING,0,0,"nearest",{},{}}); n.attr.push_back({"coordinate_transformation_mode",A_STRING,0,0,"asymmetric",{},{}}); n.attr.push_back({"nearest_mode",A_STRING,0,0,"floor",{},{}}); g.nodes.push_back(n); return yn; };
  auto c3 = [&](const std::string& x, int n_bott, bool sc) -> std::string {
    std::string last = conv(x);
    for (int i = 0; i < n_bott; ++i) { std::string h = conv(last); h = conv(h); last = sc ? add(h, last) : h; }
    std::string y2 = conv(x);
    return conv(concat({last, y2}));
  };
  auto sppf = [&](const std::string& x){ std::string x1=conv(x), q1=maxpool(x1), q2=maxpool(q1), q3=maxpool(q2); return conv(concat({x1,q1,q2,q3})); };

  std::string x0=conv("images"), x1=conv(x0), x2=c3(x1,1,true), x3=conv(x2), x4=c3(x3,2,true),
    x5=conv(x4), x6=c3(x5,3,true), x7=conv(x6), x8=c3(x7,1,true), x9=sppf(x8),
    x10=conv(x9), x11=resize2x(x10), x12=concat({x11,x6}), x13=c3(x12,1,false),
    x14=conv(x13), x15=resize2x(x14), x16=concat({x15,x4}), x17=c3(x16,1,false),
    x18=conv(x17), x19=concat({x18,x14}), x20=c3(x19,1,false),
    x21=conv(x20), x22=concat({x21,x10}), x23=c3(x22,1,false);
  std::string ins[3]={x17,x20,x23};
  for (int i = 0; i < 3; ++i) {
    std::string h = conv(ins[i]);                              // detect m[i] (plain)
    std::string on = "out" + std::to_string(i);
    g.nodes.push_back({"Identity", uniq("id"), {h}, {on}, {}});
    int64_t st = 8 << i;
    g.outputs.push_back({on, {1, na*no, IMG/st, IMG/st}});
  }
  save_onnx(g, "yolov5n.onnx");
  printf("wrote yolov5n.onnx  (%zu nodes, %zu inits, consumed %zu/%zu convs)\n",
         g.nodes.size(), g.init_f.size(), prov.i, prov.convs.size());
  return 0;
}
