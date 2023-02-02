#pragma once

#include <algorithm>
#include <iostream>
#include <libraw.h>
#include <math.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace yk {
class RawConverter {
public:
  std::vector<int> gamma_curve;

  // Conversion Matrix from Camera Native Color Spaxce to sRGB'
  xt::xtensor<float, 2> rgb_cam;

  // Gamma Curve Coefficients
  static constexpr float gmm = 2.4;
  static constexpr float linear_coeff = 12.92;
  static constexpr float linear_thresh_coeff = 0.0031308;
  static constexpr float black_offset = 0.055;

  RawConverter() : rgb_cam{xt::eye(3)}, gamma_curve(1 << 16, -1){};
  RawConverter(const float color_matrix[3][4])
      : rgb_cam{xt::zeros<ushort>({3, 3})}, gamma_curve(1 << 16, -1) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        rgb_cam(i, j) = color_matrix[i][j];
      }
    }
  };
  ~RawConverter() = default;

  template <class E, class T>
  void subtract_black(xt::xexpression<E> &e, T black_level,
                      T *black_levels) const noexcept {
    auto &image = e.derived_cast();
    if (black_level) {
      image -= black_level;
    } else {
      for (int ch = 0; ch < 3; ch++) {
        if (black_levels[ch]) {
          xt::view(image, ch, xt::all()) -= black_levels[ch];
        }
      }
    }
  }
  template <class E> void convert_to_rgb(xt::xexpression<E> &e) const noexcept {
    auto &image = e.derived_cast();
    auto rgb =
        xt::linalg::dot(rgb_cam, xt::view(image, xt::range(0, 3), xt::all()));
  }
  template <class E>
  void gamma_correction(xt::xexpression<E> &e,
                        const float maxV = USHRT_MAX) noexcept {
    auto &image = e.derived_cast();
    ushort thresh = std::max<float>(
        0, std::min<float>(USHRT_MAX, linear_thresh_coeff * maxV));

    for (int ch = 0; ch < 3; ch++) {
      for (int i = 0; i < image.shape()[1]; i++) {
        const ushort src_val = image(ch, i);
        if (0 <= gamma_curve[src_val]) {
          image(ch, i) = gamma_curve[src_val];
        } else if (image(ch, i) < thresh) {
          gamma_curve[src_val] =
              std::min<int>(USHRT_MAX, image(ch, i) * linear_coeff);
          image(ch, i) = gamma_curve[src_val];
        } else {
          float value =
              ((std::pow(src_val / maxV, 1. / gmm) * 1.055) - black_offset);
          value = std::min(1.f, std::max(0.000000001f, value)) * maxV;
          gamma_curve[src_val] = value;
          image(ch, i) = gamma_curve[src_val];
        }
      }
    }
  }
};

auto ToCvMat3b(const xt::xtensor<ushort, 2> &src, const std::size_t rows,
               const std::size_t cols) noexcept {
  cv::Mat dst(rows, cols, CV_8UC3);
  for (int r = 0, i = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++, i++) {
      auto &pixel = dst.at<cv::Vec3b>(r, c);
      // B
      pixel[0] = std::max<uchar>(0, src(2, i) >> 8);

      //  G
      pixel[1] = std::max<uchar>(0, src(1, i) >> 8);

      //  R
      pixel[2] = std::max<uchar>(0, src(0, i) >> 8);
    }
  }
  return dst;
}

template <class T>
auto VecToString(const std::vector<T> &v, const std::string del = " ") {
  std::stringstream ss;
  ss << "{";
  std::for_each(v.begin() + 1, v.end(),
                [&ss, del](auto s) { ss << " " << s << del; });
  ss << "}";
  return ss.str();
}
} // namespace yk