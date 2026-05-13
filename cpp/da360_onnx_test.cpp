/*
 * DA360 ONNX smoke test (OpenCV + ONNX Runtime).
 *
 * Build (example, Apple Silicon + Homebrew): see README.md “C++ smoke test”.
 *   brew install onnxruntime opencv cmake && cd cpp && cmake -S . -B build && cmake --build build
 *
 * CMake: ORT_ROOT defaults to /opt/homebrew; override with -DORT_ROOT=$(brew --prefix onnxruntime).
 *
 * Run:
 *   ./build/da360_onnx_test ../DA360_small.onnx ../input/panorama.jpg ../pred_disp_vis.png
 *
 * Input ONNX tensor: normalized_rgb  NCHW float32, ImageNet norm (same as DA360 datasets/real.py).
 * Default spatial size is read from the model (typically 518 x 1036).
 */

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

int64_t DimOr(int64_t d, int64_t fallback) {
  if (d <= 0) {  // includes ONNX dynamic dim -1
    return fallback;
  }
  return d;
}

void PreprocessRgb(const cv::Mat& rgb_u8, int H, int W, std::vector<float>& out_nchw) {
  if (rgb_u8.type() != CV_8UC3) {
    throw std::runtime_error("Expected CV_8UC3 RGB image");
  }
  cv::Mat resized;
  cv::resize(rgb_u8, resized, cv::Size(W, H), 0, 0, cv::INTER_CUBIC);
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> <input_image> [output_preview.png]\n";
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string image_path = argv[2];
  const std::string out_path = (argc >= 4) ? argv[3] : "pred_disp_preview.png";

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

  Ort::AllocatorWithDefaultOptions allocator;

  // Input shape from model (batch may be dynamic).
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

  std::vector<float> input_tensor;
  PreprocessRgb(rgb, static_cast<int>(H), static_cast<int>(W), input_tensor);

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
  if (outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() < static_cast<size_t>(plane)) {
    std::cerr << "Output element count too small for inferred HxW\n";
    return 1;
  }

  SaveDisparityPreview(out_data, oh, ow, out_path);
  std::cout << "Wrote preview: " << out_path << "\n";
  return 0;
}
