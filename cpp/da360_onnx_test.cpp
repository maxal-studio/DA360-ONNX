/*
 * DA360 ONNX smoke test (OpenCV + ONNX Runtime).
 *
 * Optional --ply-flip-y / --ply-flip-x adjust world axes for viewers (see README).
 *
 * Run:
 *   ./build/da360_onnx_test ../onnx_models/small.onnx ../input/panorama.jpg \
 *       --preview ../output/preview.png --ply ../output/cloud.ply
 *
 * See README.md for Python deps and Three.js viewer (`web/pointcloud_viewer.html`).
 */

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
constexpr float kDepthFarClip = 200.f;  // matches DA360 saver.save_as_point_cloud mask

int64_t DimOr(int64_t d, int64_t fallback) {
  if (d <= 0) {
    return fallback;
  }
  return d;
}

void PreprocessRgb(const cv::Mat& rgb_u8, int H, int W, std::vector<float>& out_nchw, cv::Mat* resized_rgb_u8_out) {
  if (rgb_u8.type() != CV_8UC3) {
    throw std::runtime_error("Expected CV_8UC3 RGB image");
  }
  cv::Mat resized;
  cv::resize(rgb_u8, resized, cv::Size(W, H), 0, 0, cv::INTER_CUBIC);
  if (resized_rgb_u8_out != nullptr) {
    *resized_rgb_u8_out = resized.clone();
  }
  cv::Mat f;
  resized.convertTo(f, CV_32FC3, 1.0 / 255.0);

  out_nchw.resize(static_cast<size_t>(1) * 3 * H * W);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      const cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
      for (int x = 0; x < W; ++x) {
        float v = row[x][c];
        out_nchw[static_cast<size_t>(c) * H * W + y * W + x] = (v - kMean[c]) / kStd[c];
      }
    }
  }
}

void SaveDisparityPreview(const float* data, int H, int W, const std::string& path) {
  cv::Mat disp(H, W, CV_32F, const_cast<float*>(data));
  double mn = 0.0;
  double mx = 0.0;
  cv::minMaxLoc(disp, &mn, &mx);
  if (mx <= mn + 1e-8) {
    mx = mn + 1.0;
  }
  cv::Mat u8;
  disp.convertTo(u8, CV_8U, 255.0 / (mx - mn), -mn * 255.0 / (mx - mn));
  cv::Mat color;
  cv::applyColorMap(u8, color, cv::COLORMAP_INFERNO);
  if (!cv::imwrite(path, color)) {
    throw std::runtime_error("Failed to write: " + path);
  }
}

void DispToRelativeDepth(const float* disp, int H, int W, std::vector<float>* depth_out) {
  depth_out->assign(static_cast<size_t>(H * W), 0.f);
  float min_pos = std::numeric_limits<float>::infinity();
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float d = disp[y * W + x];
      float dep = (d > 1e-6f) ? (1.0f / d) : 0.f;
      (*depth_out)[static_cast<size_t>(y * W + x)] = dep;
      if (dep > 0.f && std::isfinite(dep)) {
        min_pos = std::min(min_pos, dep);
      }
    }
  }
  if (!std::isfinite(min_pos) || min_pos <= 0.f) {
    min_pos = 1.f;
  }
  for (float& v : *depth_out) {
    if (v > 0.f) {
      v /= min_pos;
    }
  }
}

void WriteAsciiPlyEquirect(const std::vector<float>& depth,
                           const cv::Mat& rgb_hw_uint8,
                           int H,
                           int W,
                           const std::string& path,
                           bool flip_y,
                           bool flip_x) {
  if (rgb_hw_uint8.rows != H || rgb_hw_uint8.cols != W || rgb_hw_uint8.type() != CV_8UC3) {
    throw std::runtime_error("RGB size/type mismatch for point cloud");
  }

  const double pi = static_cast<double>(M_PI);
  std::vector<double> xs, ys, zs;
  std::vector<uint8_t> cr, cg, cb;
  xs.reserve(static_cast<size_t>(H * W) / 4);
  ys.reserve(xs.capacity());
  zs.reserve(xs.capacity());
  cr.reserve(xs.capacity());
  cg.reserve(xs.capacity());
  cb.reserve(xs.capacity());

  // Same Theta/Phi as DA360 saver.py (integer pixel indices).
  for (int y = 0; y < H; ++y) {
    double theta = pi - static_cast<double>(y) * pi / static_cast<double>(H) - pi / (2.0 * static_cast<double>(H));
    double sin_t = std::sin(theta);
    double cos_t = std::cos(theta);
    for (int x = 0; x < W; ++x) {
      float dep = depth[static_cast<size_t>(y * W + x)];
      if (!(dep > 0.f) || !(dep < kDepthFarClip)) {
        continue;
      }
      double phi =
          static_cast<double>(x) * 2.0 * pi / static_cast<double>(W) + pi / static_cast<double>(W) - pi;
      double sin_p = std::sin(phi);
      double cos_p = std::cos(phi);
      double xd = static_cast<double>(dep) * sin_t * sin_p;
      double yd = static_cast<double>(dep) * cos_t;
      double zd = static_cast<double>(dep) * sin_t * cos_p;
      if (flip_x) {
        xd = -xd;
      }
      if (flip_y) {
        yd = -yd;
      }
      xs.push_back(xd);
      ys.push_back(yd);
      zs.push_back(zd);
      const cv::Vec3b& px = rgb_hw_uint8.at<cv::Vec3b>(y, x);
      cr.push_back(px[0]);
      cg.push_back(px[1]);
      cb.push_back(px[2]);
    }
  }

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot write PLY: " + path);
  }
  const size_t n = xs.size();
  out << "ply\nformat ascii 1.0\ncomment DA360-style ERP point cloud\n";
  out << "element vertex " << n << "\n";
  out << "property float x\nproperty float y\nproperty float z\n";
  out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  out << "end_header\n";
  out.precision(9);
  for (size_t i = 0; i < n; ++i) {
    out << xs[i] << " " << ys[i] << " " << zs[i] << " ";
    out << static_cast<int>(cr[i]) << " " << static_cast<int>(cg[i]) << " " << static_cast<int>(cb[i]) << "\n";
  }
}

void PrintUsage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " <model.onnx> <input_image> [--preview out.png] [--ply out.ply] [--ply-flip-y] [--ply-flip-x]\n";
  std::cerr << "  Default preview: pred_disp_preview.png\n";
  std::cerr << "  Point cloud: same Theta/Phi as DA360 saver.py; depth = (1/disp)/min(depth); mask depth < "
            << kDepthFarClip << "\n";
  std::cerr << "  --ply-flip-y  Negate world Y in PLY (DA360 puts image-top at Y<0; many viewers expect +Y up).\n";
  std::cerr << "  --ply-flip-x  Negate world X (try if left/right looks mirrored in 3D).\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string image_path = argv[2];
  std::string out_path = "pred_disp_preview.png";
  std::string ply_path;
  bool ply_flip_y = false;
  bool ply_flip_x = false;

  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ply" && i + 1 < argc) {
      ply_path = argv[++i];
    } else if (a == "--preview" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (a == "--ply-flip-y") {
      ply_flip_y = true;
    } else if (a == "--ply-flip-x") {
      ply_flip_x = true;
    } else {
      std::cerr << "Unknown or incomplete argument: " << a << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    std::cerr << "Failed to read image: " << image_path << "\n";
    return 1;
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "da360_onnx_test");
  Ort::SessionOptions session_opts;
  session_opts.SetIntraOpNumThreads(0);
  session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
  const std::wstring model_w(model_path.begin(), model_path.end());
  Ort::Session session(env, model_w.c_str(), session_opts);
#else
  Ort::Session session(env, model_path.c_str(), session_opts);
#endif

  Ort::TypeInfo in_type = session.GetInputTypeInfo(0);
  auto in_shape_info = in_type.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> in_shape = in_shape_info.GetShape();
  if (in_shape.size() != 4) {
    std::cerr << "Expected 4D input, got rank " << in_shape.size() << "\n";
    return 1;
  }
  const int64_t N = std::max<int64_t>(1, DimOr(in_shape[0], 1));
  const int64_t C = DimOr(in_shape[1], 3);
  const int64_t H = DimOr(in_shape[2], 518);
  const int64_t W = DimOr(in_shape[3], 1036);
  if (C != 3) {
    std::cerr << "Expected 3 input channels, got " << C << "\n";
    return 1;
  }
  std::cout << "Model input spatial size: " << H << " x " << W << " (resizing " << image_path << ")\n";

  cv::Mat resized_rgb_u8;
  std::vector<float> input_tensor;
  PreprocessRgb(rgb, static_cast<int>(H), static_cast<int>(W), input_tensor, &resized_rgb_u8);

  std::vector<int64_t> input_dims = {N, C, H, W};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_val = Ort::Value::CreateTensor<float>(
      mem, input_tensor.data(), input_tensor.size(), input_dims.data(), input_dims.size());

  const char* input_names[] = {"normalized_rgb"};
  const char* output_names[] = {"pred_disp"};

  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_val, 1, output_names, 1);
  if (outputs.empty()) {
    std::cerr << "No outputs from session.Run\n";
    return 1;
  }

  float* out_data = outputs[0].GetTensorMutableData<float>();
  auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
  std::vector<int64_t> out_shape = out_info.GetShape();
  std::cout << "Output shape:";
  for (int64_t d : out_shape) {
    std::cout << " " << d;
  }
  std::cout << "\n";

  int oh = static_cast<int>(H);
  int ow = static_cast<int>(W);
  if (out_shape.size() == 4) {
    oh = static_cast<int>(DimOr(out_shape[2], H));
    ow = static_cast<int>(DimOr(out_shape[3], W));
  } else if (out_shape.size() == 3) {
    oh = static_cast<int>(DimOr(out_shape[1], H));
    ow = static_cast<int>(DimOr(out_shape[2], W));
  }

  const size_t plane = static_cast<size_t>(oh) * static_cast<size_t>(ow);
  if (outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() < plane) {
    std::cerr << "Output element count too small for inferred HxW\n";
    return 1;
  }

  SaveDisparityPreview(out_data, oh, ow, out_path);
  std::cout << "Wrote preview: " << out_path << "\n";

  if (!ply_path.empty()) {
    std::vector<float> depth;
    DispToRelativeDepth(out_data, oh, ow, &depth);
    WriteAsciiPlyEquirect(depth, resized_rgb_u8, oh, ow, ply_path, ply_flip_y, ply_flip_x);
    std::cout << "Wrote point cloud (ASCII PLY): " << ply_path << "\n";
  }

  return 0;
}
