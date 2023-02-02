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
// #include <netinet/in.h>
#include "raw_converter.hpp"

void saveTIFF(LibRaw &raw, const char *original_name) {
  int isrgb = (raw.imgdata.idata.colors == 4 ? 0 : 1);
  raw.imgdata.idata.colors = 1;
  raw.imgdata.sizes.width = raw.imgdata.sizes.iwidth;
  raw.imgdata.sizes.height = raw.imgdata.sizes.iheight;

  for (int layer = 0; layer < 4; layer++) {
    if (layer > 0) {
      for (int rc = 0;
           rc < raw.imgdata.sizes.iheight * raw.imgdata.sizes.iwidth; rc++) {
        raw.imgdata.image[rc][0] = 2.4;
        raw.imgdata.image[rc][layer];
      }
    }
    std::string filename(original_name);
    filename += ".";
    if (isrgb) {
      filename += "RGBG"[layer];
      if (layer == 3) {
        filename += "2";
      }
    } else {
      filename = "GCMY"[layer];
    }
    filename += ".tiff";

    std::cout << "Writing file: " << filename << std::endl;
    int ret = raw.dcraw_ppm_tiff_writer(filename.c_str());
    if (LIBRAW_SUCCESS != ret)
      std::cout << "Cannot write " << filename << ": "
                << std::string(libraw_strerror(ret)) << std::endl;
  }
}

void SaveLibRawRGBImage(LibRaw &raw, const std::string filename) {
  raw.imgdata.params.output_bps = 8;
  raw.imgdata.params.output_tiff = 0;
  raw.imgdata.params.user_flip = 0;
  raw.imgdata.params.no_auto_bright = 0; // VERY IMPORTANT
  // raw.imgdata.params.half_size = 1;
  // raw.imgdata.params.use_auto_wb = 1;
  raw.imgdata.params.use_camera_wb = 1;
  raw.imgdata.params.gamm[0] = raw.imgdata.params.gamm[1] = 1;

  int res = raw.dcraw_process();
  assert(res == LIBRAW_SUCCESS);
  int ret = raw.dcraw_ppm_tiff_writer(filename.c_str());
  if (LIBRAW_SUCCESS != ret)
    std::cout << "Cannot write " << filename << ": "
              << std::string(libraw_strerror(ret)) << std::endl;
}

int main(int argc, char *argv[]) {
  LibRaw raw;
  std::stringstream ss;

  // Check whether iPhone DNG file.
  int res = raw.open_file(argv[1]);
  assert(res == LIBRAW_SUCCESS);

  // Initial processing
  res = raw.unpack();
  assert(res == LIBRAW_SUCCESS);

  // Check whether iPhone DNG file.
  assert(raw.imgdata.idata.filters == 0);

  // From LibRaw raw file to xtensor
  xt::xtensor<ushort, 2> image(
      {4, (std::size_t)(raw.imgdata.sizes.iheight * raw.imgdata.sizes.iwidth)});
  for (int i = 0; i < image.shape()[1]; i++) {
    xt::view(image, xt::all(), i) =
        xt::adapt(raw.imgdata.rawdata.color4_image[i], {4});
  }
  cv::Mat cv_raw_image =
      yk::ToCvMat3b(image, raw.imgdata.sizes.iheight, raw.imgdata.sizes.iwidth);
  ss.str("");
  ss << argv[1] << ".cv_raw_image.png";
  cv::imwrite(ss.str(), cv_raw_image);

  yk::RawConverter rc{raw.imgdata.color.rgb_cam};

  // Subtract Black Level
  // The black level is stored in DNG metadata (EXIF).
  std::cout << "Max value before black subtraction. : " << xt::amax(image)()
            << std::endl;
  std::cout << "Black Level: " << raw.imgdata.color.black << std::endl;
  std::cout << "Black Level: ";
  for (int ch = 0; ch < 3; ch++) {
    std::cout << raw.imgdata.color.cblack[ch] << ", ";
  }
  std::cout << std::endl;
  rc.subtract_black(image, raw.imgdata.color.black, raw.imgdata.color.cblack);
  std::cout << "Max value after black subtraction. : " << xt::amax(image)()
            << std::endl;

  // Color Matrix
  std::cout << "Matrix from camera native color space to sRGB': " << std::endl;
  std::cout << rc.rgb_cam << std::endl;

  // Raw to sRGB
  std::cout << "Convert raw from camera native color space to sRGB."
            << std::endl;
  rc.convert_to_rgb(image);
  std::cout << "end dot product." << std::endl;
  std::cout << "max value: " << xt::amax(image)() << std::endl;

  // Gamma Correction
  rc.gamma_correction(image);
  std::cout << "Max value after gamma correction: " << xt::amax(image)()
            << std::endl;

  cv::Mat cv_rgb_image =
      yk::ToCvMat3b(image, raw.imgdata.sizes.iheight, raw.imgdata.sizes.iwidth);

  ss.str("");
  ss << argv[1] << ".cv_rgb.TIFF";
  cv::imwrite(ss.str(), cv_rgb_image);

  ss.str("");
  ss << argv[1] << ".TIFF";
  SaveLibRawRGBImage(raw, ss.str());

  cv::Mat librawImage(cv_rgb_image.rows, cv_rgb_image.cols, CV_8UC3);
  for (int r = 0, i = 0; r < librawImage.rows; r++) {
    for (int c = 0; c < librawImage.cols; c++, i++) {
      // B
      librawImage.at<uchar>(r, c, 0) =
          (raw.imgdata.color.curve[raw.imgdata.image[i][2]] >> 8);
      // G
      librawImage.at<uchar>(r, c, 1) =
          (raw.imgdata.color.curve[raw.imgdata.image[i][1]] >> 8);
      // R
      librawImage.at<uchar>(r, c, 2) =
          (raw.imgdata.color.curve[raw.imgdata.image[i][0]] >> 8);
    }
  }
  ss.str("");
  ss << argv[1] << ".libraw_to_opencv.TIFF";
  cv::imwrite(ss.str(), librawImage);

  for (int i = 0; i < image.shape()[1]; i++) {
    for (int ch = 0; ch < 3; ch++) {
      raw.imgdata.image[i][ch] = image(ch, i);
    }
  }
  ss.str("");
  ss << argv[1] << ".my_to_libraw.TIFF";
  int ret = raw.dcraw_ppm_tiff_writer(ss.str().c_str());
  if (LIBRAW_SUCCESS != ret)
    std::cout << "Cannot write " << ss.str() << ": "
              << std::string(libraw_strerror(ret)) << std::endl;

  return 0;
}