#pragma once

#include <algorithm>
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
    auto image = src;
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

  /**
   * @brief Emphasize the brightness and contrast if an image.
   * Clips the input data to [0, USHRT_MAX] and creates a histogram with an
   * interval of 8. Change the range of values so that [min-point, max-point] to
   * [0, USHRT_MAX]. The min_point and max-point are determined as the value
   * at which the cumulative histogram is about stresh_thresh/2 from both ends
   * of each histogram.
   * @tparam E The derived type of xtensor
   * @param e an image data stored in xtensor xexpression
   * @param strech_thresh Thresholds of cumulative histograms that are the two
   * ends of the re-rang.
   * @return
   */
  template <class E>
  auto adjust_brightness(const xt::xexpression<E> &e,
                         const float strech_thresh = 0.4,
                         const bool debug = false) noexcept {
    if (debug) {
      debug_message << "Start adjust_brightness()\n";
    }
    auto &src = e.derived_cast();
    auto &&image = xt::clip(src, 0, USHRT_MAX);
    float min_value = xt::amin(image)(), max_value = xt::amax(image)();
    if (0.999999f <= strech_thresh) {
      max_value = min_value;
    } else if (0 < strech_thresh) {
      int acc_thresh = image.shape()[1] * strech_thresh * 0.5f;
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
    auto &&res = xt::eval(image * alpha +
                          beta); // xt::eval(xt::fma(image, alpha, beta));
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