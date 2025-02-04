#pragma once

#include <algorithm>
#include <execution>
#include <iostream>
#include <libraw.h>
#include <math.h>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace yk {

/**
 * @class RawConverter
 * @brief Processor for ProRaw/DNG files
 * The RawConverter class implements raw image processing with xtensor.
 * Image data stored in xtensor is assumed to be basically of 3-channel type
 * ushort. Its shape should be (3, N), where N equals image-width *
 * image-height. Channels are in RGB order.
 */
class RawConverter {
public:
  // Gamma curve constants
  static constexpr float gmm = 2.4;
  static constexpr float linear_coeff = 12.92;
  static constexpr float linear_thresh_coeff = 0.0031308;
  static constexpr float black_offset = 0.055;

  RawConverter()
      : gamma_curve(1 << 16, -1), sRGB_from_xyzD65{
                                      {3.079955, -1.537139, -0.542816},
                                      {-0.921259, 1.876011, 0.045247},
                                      {0.052887, -0.204026, 1.151138}} {};
  RawConverter(const RawConverter &other) = delete;
  RawConverter &operator=(const RawConverter &other) = delete;
  RawConverter(RawConverter &&other) = default;
  RawConverter &operator=(RawConverter &&other) = delete;

  ~RawConverter() = default;

  /**
   * @brief Subtract black level from image data.
   * @tparam E The derived type of xtensor
   * @tparam T The type of black lebel values
   * @param e an image data stored in xtensor xexpression
   * @param black_level common black level. f non-zero, only this value is
   * applied and black_lebels is ignored.
   * @param black_levels black lebels for RGB
   */
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

  /**
   * @brief Convert image data in camera native color space to CIE D65 XYZ color
   * space.
   * @tparam E The derived type of xtensor
   * @param e an image data stored in xtensor xexpression
   * @param cm transformation matrix that converts XYZ values to reference
   * camera native color space. It is stored as ColorMatrix2 in DNG.
   * @param ab AnalogBalance values in DNG
   * @return  image data converted to D65 XYZ
   */
  template <class E>
  auto camera_to_xyz(const xt::xexpression<E> &e, const float cm[4][3],
                     const float ab[4]) const noexcept {
    auto &image = e.derived_cast();
    // Matrix to convert from XYZ color space to camera native color space.
    xt::xtensor<float, 2> color_matrix({3, 3});
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        color_matrix(i, j) = cm[i][j];
      }
    }
    auto &&analog_balance = xt::diag(xt::xarray<float>{ab[0], ab[1], ab[2]});
    //   Matrix to convert from camera native color space to XYZ color
    //   space.
    auto &&cam_from_xyz = xt::linalg::dot(analog_balance, color_matrix);
    auto sum = xt::sum(cam_from_xyz, {1});
    for (int i = 0; i < 3; i++) {
      if (0.0000001 < sum(i)) {
        xt::view(cam_from_xyz, i, xt::all()) /= sum(i);
      } else {
        xt::view(cam_from_xyz, i, xt::all()) = 0;
      }
    }
    auto &&xyz_from_cam = xt::linalg::inv(cam_from_xyz);
    return xt::linalg::dot(xyz_from_cam, image);
  }

  /**
   * @brief Convert image data in CIE D65 XYZ color space to sRGB'.
   * @tparam E The derived type of xtensor
   * @param e an image data stored in xtensor xexpression
   * @return image data converted to sRGB'
   */
  template <class E>
  auto xyz_to_sRGB(const xt::xexpression<E> &e) const noexcept {
    return xt::linalg::dot(sRGB_from_xyzD65, e);
  }

  /**
   * @brief Convert an image data in camera native color space to sRGB'.
   * @tparam E The derived type of xtensor
   * @param e an image data stored in xtensor xexpression
   * @param color_matrix transformation matrix that converts XYZ values to
   * reference camera native color space. It is stored as ColorMatrix2 in DNG.
   * @return image data converted to sRGB'
   */
  template <class E>
  auto camera_to_sRGB(const xt::xexpression<E> &e,
                      const float color_matrix[3][4]) const noexcept {
    auto &image = e.derived_cast();
    // Conversion Matrix from Camera Native Color Spaxce to sRGB'
    xt::xtensor<float, 2> srgb_to_cam({3, 3});
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        srgb_to_cam(i, j) = color_matrix[i][j];
      }
    }
    return xt::linalg::dot(srgb_to_cam, image);
  }

  /**
   * @brief Apply gamma correction
   * @tparam E The derived type of xtensor
   * @param e an image data stored in xtensor xexpression
   * @return
   */
  template <class E>
  auto gamma_correction(const xt::xexpression<E> &e) noexcept {
    auto &src = e.derived_cast();
    auto image = xt::eval(src);
    constexpr float max_value = USHRT_MAX;
    constexpr ushort thresh = linear_thresh_coeff * max_value;

    for (int ch = 0; ch < 3; ch++) {
      for (int i = 0; i < image.shape()[1]; i++) {
        const int src_val =
            std::min<int>(USHRT_MAX, std::max<int>(0, image(ch, i)));
        if (0 <= gamma_curve[src_val]) {
          image(ch, i) = static_cast<ushort>(gamma_curve[src_val]);
        } else if (image(ch, i) < thresh) {
          gamma_curve[src_val] = std::max<int>(
              0, std::min<int>(USHRT_MAX, image(ch, i) * linear_coeff));
          image(ch, i) = static_cast<ushort>(gamma_curve[src_val]);
        } else {
          float value = static_cast<float>(src_val) / max_value;
          value = (std::pow(value, 1. / gmm) * 1.055) - black_offset;
          value *= max_value;
          gamma_curve[src_val] =
              std::max<int>(0, std::min<int>(USHRT_MAX, value));
          image(ch, i) = static_cast<ushort>(gamma_curve[src_val]);
        }
      }
    }
    return image;
  }

  template <class E>
  void raw_adjust(xt::xexpression<E> &e,
                  const float scale = 1.) const noexcept {
    auto &src = e.derived_cast();
    int n = src.shape()[1];
    for (int i = 0; i < src.shape()[1]; i++) {
      src(0, i) = std::clamp<int>(src(0, i) << 3, 0, USHRT_MAX);
      src(1, i) = std::clamp<int>(src(1, i) << 3, 0, USHRT_MAX);
      src(2, i) = std::clamp<int>(src(2, i) << 3, 0, USHRT_MAX);
    }
  }

  template <class E>
  auto adjust_brightness_6(const xt::xexpression<E> &e,
                           const bool debug = false) const noexcept {
    auto &src = e.derived_cast();
    auto minv = xt::amin(src);
    auto maxv = xt::amax(src);
    float scale =
        1. / std::clamp<float>(maxv() - minv(), 0.00000001, USHRT_MAX);
    return (src - minv) * scale;
  }

  template <class E>
  auto adjust_brightness_5(const xt::xexpression<E> &e,
                           const float stddev_rate = 0.96,
                           const bool debug = false) noexcept {
    auto &src = e.derived_cast();
    int n = src.shape()[1];

    float stddev = xt::stddev(xt::view(src, 1, xt::all()))();
    float stddev_after = (float(USHRT_MAX) * stddev_rate) / 3.f;

    if (debug) {
      std::cout << "stddev: " << stddev << std::endl;
      debug_message << "stddev: " << stddev << "\n";
      std::cout << "stddev_after: " << stddev_after << std::endl;
      debug_message << "stddev_after: " << stddev_after << "\n";
    }
    std::vector<int> mapped(1 << 16, -1);
    auto res = src;
    auto it = res.begin();
    while (it != res.end()) {
      int v = *it;
      if (mapped[v] < 0) {
        float fv = v;
        fv /= stddev;
        fv *= stddev_after;
        *it = std::clamp<int>(int(fv), 0, USHRT_MAX);
        mapped[v] = *it;
      } else {
        *it = mapped[v];
      }
      it++;
    }

    if (debug) {
      float stddev_actual = xt::stddev(res)();
      std::cout << "stddev_actual: " << stddev_actual << std::endl;
      debug_message << "stddev_actual: " << stddev_actual << "\n";
    }

    return res;
  }

  template <class E>
  auto adjust_brightness_4(const xt::xexpression<E> &e,
                           const float mean_rate = 0.5,
                           const float stddev_rate = 0.96,
                           const bool debug = false) noexcept {
    auto &src = e.derived_cast();
    int n = src.shape()[1];

    float mean = xt::mean(xt::view(src, 1, xt::all()))();
    float stddev = xt::stddev(xt::view(src, 1, xt::all()))();
    float mean_after = float(USHRT_MAX) * mean_rate;
    float stddev_after = (float(USHRT_MAX) * stddev_rate - mean_after) / 3.f;

    if (debug) {
      std::cout << "mean: " << mean << std::endl;
      debug_message << "mean: " << mean << "\n";
      std::cout << "stddev: " << stddev << std::endl;
      debug_message << "stddev: " << stddev << "\n";
      std::cout << "mean_after: " << mean_after << std::endl;
      debug_message << "mean_after: " << mean_after << "\n";
      std::cout << "stddev_after: " << stddev_after << std::endl;
      debug_message << "stddev_after: " << stddev_after << "\n";
    }
    std::vector<int> mapped(1 << 16, -1);
    auto res = src;
    auto it = res.begin();
    while (it != res.end()) {
      int v = *it;
      if (mapped[v] < 0) {
        float fv = v;
        fv = (fv - mean) / stddev;
        fv = fv * stddev_after + mean_after;
        *it = std::clamp<int>(int(fv), 0, USHRT_MAX);
        mapped[v] = *it;
      } else {
        *it = mapped[v];
      }
      it++;
    }

    if (debug) {
      float mean_actual = xt::mean(res)();
      float stddev_actual = xt::stddev(res)();
      std::cout << "mean after actual: " << mean_actual << std::endl;
      debug_message << "mean after actual: " << mean_actual << "\n";
      std::cout << "stddev_actual: " << stddev_actual << std::endl;
      debug_message << "stddev_actual: " << stddev_actual << "\n";
    }

    return res;
  }

  template <class E>
  auto adjust_brightness_3(const xt::xexpression<E> &e,
                           const bool debug = false) noexcept {
    auto &src = e.derived_cast();

    std::vector<std::size_t> size = {1 << 16};
    xt::xtensor<float, 1> histogram = xt::zeros<float>(size);
    for (int i = 0; i < src.shape()[1]; i++) {
      histogram(std::clamp<int>(src(1, i), 0, USHRT_MAX))++;
    }

    histogram =
        xt::fma(histogram, float(USHRT_MAX) / float(src.shape()[1]), 0.);

    for (int i = 1; i < 1 << 16; i++) {
      histogram(i) += histogram(i - 1);
    }

    auto res = xt::empty_like(src);
    for (int i = 0; i < src.shape()[1]; i++) {
      res(0, i) = histogram(std::clamp<int>(src(0, i), 0, USHRT_MAX));
      res(1, i) = histogram(std::clamp<int>(src(1, i), 0, USHRT_MAX));
      res(2, i) = histogram(std::clamp<int>(src(2, i), 0, USHRT_MAX));
    }

    return res;
  }

  template <class E>
  auto adjust_brightness_2(const xt::xexpression<E> &e,
                           const float edge_acc_rate = 0.01,
                           const float edge_val_rage = 0.001,
                           const bool debug = false) noexcept {
    if (debug) {
      debug_message << "Start adjust_brightness_2()\n";
    }
    auto &src = e.derived_cast();
    auto image = xt::clip(src, 0, USHRT_MAX);
    auto res = xt::eval(image);

    int acc_thresh = image.shape()[1] * edge_acc_rate;
    if (debug) {
      debug_message << "acc_thresh: " << acc_thresh << "\n";
    }
    std::vector<long long> histogram(1 << 13, 0);
    for (int i = 0; i < image.shape()[1]; i++) {
      histogram[static_cast<ushort>(image(1, i)) >> 3]++;
    }

    // calculate the minimum value in the scope.
    ushort lower_bound = 0;
    {
      int bin = 0;
      int acc = 0;
      while (acc < acc_thresh && bin < histogram.size()) {
        acc += histogram[bin];
        bin++;
      }
      if (debug) {
        debug_message << "min bin: " << bin << "\n";
      }
      lower_bound = (bin << 3);
    }

    // calculate the maximum value in the scope.
    ushort upper_bound = USHRT_MAX;
    {
      int bin = histogram.size() - 1;
      int acc = 0;
      while (acc < acc_thresh && 0 < bin) {
        acc += histogram[bin];
        bin--;
      }
      if (debug) {
        debug_message << "max bin: " << bin << "\n";
      }
      upper_bound = (bin << 3);
    }

    ushort mapped_lower_bound = USHRT_MAX * edge_val_rage;
    ushort mapped_upper_bound = USHRT_MAX - mapped_lower_bound;

    float lower_edge_slope = float(mapped_lower_bound) / float(lower_bound);
    float upper_edge_slope =
        float(USHRT_MAX - mapped_lower_bound) / float(USHRT_MAX - upper_bound);
    float mid_slope = float(mapped_upper_bound - mapped_lower_bound) /
                      float(upper_bound - lower_bound);
    if (debug) {
      debug_message << "lower bound: " << lower_bound << "\n";
      debug_message << "mapped lower bound: " << mapped_lower_bound << "\n";
      debug_message << "upper bound: " << upper_bound << "\n";
      debug_message << "mapped bound: " << mapped_upper_bound << "\n";
      debug_message << "lower edge slope: " << lower_edge_slope << "\n";
      debug_message << "upper edge slope: " << upper_edge_slope << "\n";
      debug_message << "mid slope: " << mid_slope << "\n";
    }

    std::vector<int> mapped(1 << 16, -1);
    auto it = res.begin();
    while (it != res.end()) {
      if (mapped[*it] < 0) {
        ushort val = std::clamp<int>(*it, 0., USHRT_MAX);
        if (*it < lower_bound) {
          *it = *it * lower_edge_slope;
        } else if (upper_bound < *it) {
          *it = (*it - upper_bound) * upper_edge_slope + mapped_upper_bound;
        } else {
          *it = (*it - lower_bound) * mid_slope + mapped_lower_bound;
        }
        mapped[val] = *it;
      } else {
        *it = mapped[*it];
      }
      it++;
    }
    return res;
  }

  /**
   * @brief Emphasize the brightness and contrast (histogram stretching).
   * Clip the input data to [0, USHRT_MAX] and create a histogram with
   * interval 8. Change the range of values so that [min-point, max-point] is
   * [0, USHRT_MAX]. min_point and max-point are determined to be top
   * stretch_rate/2% and bottom stretch_rate/2%.
   * @tparam E The derived type of xtensor
   * @param e an image data stored in xtensor xexpression
   * @param stretch_rate Percentage that defines the min and max thresholds
   * (Range [0, 1])
   * @return
   */
  template <class E>
  auto adjust_brightness(const xt::xexpression<E> &e,
                         const float strech_rate = 0.4,
                         const bool debug = false) noexcept {
    if (debug) {
      debug_message << "Start adjust_brightness()\n";
    }
    auto &src = e.derived_cast();
    if (strech_rate < 0.000001f) {
      return src;
    }
    auto &&image = xt::clip(src, 0, USHRT_MAX);
    float min_value = xt::amin(image)(), max_value = xt::amax(image)();
    if (0.999999f <= strech_rate) {
      max_value = min_value;
    } else if (0 < strech_rate) {
      int acc_thresh = image.shape()[1] * strech_rate * 0.5f;
      if (debug) {
        debug_message << "acc_thresh: " << acc_thresh << "\n";
      }
      std::vector<long long> histogram(1 << 13, 0);
      for (int i = 0; i < image.shape()[1]; i++) {
        histogram[static_cast<ushort>(image(1, i)) >> 3]++;
      }
      // calculate the minimum value in the scope.
      {
        int bin = 0;
        int acc = 0;
        while (acc < acc_thresh && bin < histogram.size()) {
          acc += histogram[bin];
          bin++;
        }
        if (debug) {
          debug_message << "min bin: " << bin << "\n";
        }
        min_value = (bin << 3);
      }
      // calculate the maximum value in the scope.
      {
        int bin = histogram.size() - 1;
        int acc = 0;
        while (acc < acc_thresh && 0 < bin) {
          acc += histogram[bin];
          bin--;
        }
        if (debug) {
          debug_message << "max bin: " << bin << "\n";
        }
        max_value = (bin << 3);
      }
    }
    // scaling: min_value -> 0, max_value -> USHRT_MAX
    if (debug) {
      debug_message << "max value: " << max_value << "\n";
      debug_message << "min value: " << min_value << "\n";
    }
    const float alpha =
        (max_value - min_value) < 0.00001
            ? 0
            : static_cast<float>(USHRT_MAX) / (max_value - min_value);
    const float beta = -min_value * alpha;
    if (debug) {
      debug_message << "alpha: " << std::to_string(alpha) << "\n";
      debug_message << "beta: " << std::to_string(beta) << "\n";
    }
    // calculate image * alpha + beta
    auto &&res = xt::eval(xt::fma(image, alpha, beta));
    if (debug) {
      debug_message << "End adjust_brightness()\n";
    }
    return res;
  }

  // Cache gamma correction values
  std::vector<int> gamma_curve;

  // CIE-XYZ to sRGB'
  const xt::xtensor_fixed<float, xt::xshape<3, 3>> sRGB_from_xyzD65;

  std::stringstream debug_message;
};
} // namespace yk