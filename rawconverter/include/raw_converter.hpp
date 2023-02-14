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
    xt::xtensor<float, 2> analog_balance = xt::zeros<float>({3, 3});
    for (int i = 0; i < 3; i++) {
      analog_balance(i, i) = ab[i];
    }

    auto &&xyz_to_cam = xt::linalg::dot(analog_balance, color_matrix);

    // Matrix to convert from camera native color space to XYZ color
    // space.
    auto &&cam_to_xyz = xt::linalg::inv(xyz_to_cam);
    return xt::linalg::dot(cam_to_xyz, image);
  }

  /**
   * @brief Convert image data in CIE D65 XYZ color space to sRGB'.
   * @tparam E The derived type of xtensor
   * @param e an image data stored in xtensor xexpression
   * @return image data converted to sRGB'
   */
  template <class E>
  auto xyz_to_sRGB(const xt::xexpression<E> &e) const noexcept {
    auto &image = e.derived_cast();
    return xt::linalg::dot(xyzD65_to_sRGB, image);
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
  auto gamma_correction(const xt::xexpression<E> &&e) noexcept {
    auto &src = e.derived_cast();
    auto max_value = xt::amax(src)();
    xt::xtensor<ushort, 2> image =
        xt::where(src < 0, 0, xt::where(USHRT_MAX < src, USHRT_MAX, src));
    ushort thresh = static_cast<ushort>(std::max<float>(
        0.f, std::min<float>(USHRT_MAX, linear_thresh_coeff * max_value)));

    for (int ch = 0; ch < 3; ch++) {
      for (int i = 0; i < image.shape()[1]; i++) {
        const ushort src_val = image(ch, i);
        if (0 <= gamma_curve[src_val]) {
          image(ch, i) = static_cast<ushort>(gamma_curve[src_val]);
        } else if (image(ch, i) < thresh) {
          gamma_curve[src_val] = static_cast<int>(std::max<float>(
              0, std::min<float>(USHRT_MAX, image(ch, i) * linear_coeff)));
          image(ch, i) = static_cast<ushort>(gamma_curve[src_val]);
        } else {
          float value = src_val / max_value;
          value = (std::pow(value, 1. / gmm) * 1.055) - black_offset;
          value =
              std::max(0.f, std::min(1.f, std::max(0.f, value)) * max_value);
          gamma_curve[src_val] = static_cast<int>(
              std::max<int>(0, std::min<int>(USHRT_MAX, value)));
          image(ch, i) = static_cast<ushort>(gamma_curve[src_val]);
        }
      }
    }
    return image;
  }

  /**
   * @brief
   * @tparam E
   * @param e
   * @param trunc_thresh
   * @return
   */
  template <class E>
  auto ajust_brightness(const xt::xexpression<E> &e,
                        const float trunc_thresh = 0.0) const {
    auto &src = e.derived_cast();
    auto &&image =
        xt::where(src < 0, 0, xt::where(USHRT_MAX < src, USHRT_MAX, src));

    int min_value = xt::amin(image)(), max_value = xt::amax(image)();
    if (trunc_thresh < 1.f / image.shape()[1] || 0.5 < trunc_thresh) {
      std::cerr << "wrong parameter is set in RawConverter::trunc_thresh(). "
                   "trunc_thresh: "
                << std::to_string(trunc_thresh) << std::endl;
      std::cerr << "RawConverter::trunc_thresh() will not change any values."
                << std::endl;
    } else {
      int acc_thresh = image.shape()[1] * trunc_thresh;
      std::vector<int> histogram(1 << 13, 0);
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
        if (0 < bin) {
          min_value = (bin << 3) - 1;
        }
      }
      // calculate the maximum value in the scope.
      {
        int bin = histogram.size() - 1;
        int acc = 0;
        while (acc < acc_thresh && 0 < bin) {
          acc += histogram[bin];
          bin--;
        }
        if (bin < histogram.size() - 1) {
          max_value = ((bin + 1) << 3);
        }
      }
    }
    // scaling: min_value -> 0, max_value -> USHRT_MAX
    const float alpha = USHRT_MAX / (max_value - min_value);
    const float beta = -min_value * alpha;
    // calculate image * alpha + beta
    return xt::eval(xt::fma(image, alpha, beta));
  }

  // Cache gamma correction values
  std::vector<int> gamma_curve;

  // CIE-XYZ to sRGB'
  const xt::xtensor_fixed<float, xt::xshape<3, 3>> sRGB_from_xyzD65;

  std::stringstream debug_message;
};
} // namespace yk