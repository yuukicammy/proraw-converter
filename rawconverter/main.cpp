#include <iostream>
#include <libraw.h>
#include <math.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
// #include <netinet/in.h>

void saveTIFF(LibRaw &raw, const char *original_name) {
  int isrgb = (raw.imgdata.idata.colors == 4 ? 0 : 1);
  raw.imgdata.idata.colors = 1;
  raw.imgdata.sizes.width = raw.imgdata.sizes.iwidth;
  raw.imgdata.sizes.height = raw.imgdata.sizes.iheight;

  for (int layer = 0; layer < 4; layer++) {
    if (layer > 0) {
      for (int rc = 0;
           rc < raw.imgdata.sizes.iheight * raw.imgdata.sizes.iwidth; rc++)
        raw.imgdata.image[rc][0] = raw.imgdata.image[rc][layer];
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

int main(int argc, char *argv[]) {
  LibRaw raw;

  raw.imgdata.params.output_bps = 8;
  raw.imgdata.params.output_tiff = 1;
  //  raw.imgdata.params.user_flip = 0;
  raw.imgdata.params.no_auto_bright = 1; // VERY IMPORTANT
  // raw.imgdata.params.half_size = 1;
  // raw.imgdata.params.gamm[0] = raw.imgdata.params.gamm[1] = 1;

  int res = raw.open_file(argv[1]);
  assert(res == LIBRAW_SUCCESS);
  res = raw.unpack();
  assert(res == LIBRAW_SUCCESS);

  assert(std::string(raw.imgdata.idata.cdesc) == "RGBG");
  assert(raw.imgdata.idata.filters == 0);

  res = raw.raw2image();
  assert(res == LIBRAW_SUCCESS);

  raw.subtract_black();

  std::stringstream ss;
  ss << argv[1] << ".TIFF";
  int ret = raw.dcraw_ppm_tiff_writer(ss.str().c_str());
  if (LIBRAW_SUCCESS != ret)
    std::cout << "Cannot write " << argv[2] << ": "
              << std::string(libraw_strerror(ret)) << std::endl;

  unsigned short maxV = raw.imgdata.color.data_maximum
                            ? raw.imgdata.color.data_maximum
                            : raw.imgdata.color.maximum;
  std::cout << maxV << std::endl;
  unsigned short scale =
      static_cast<unsigned short>(1 << raw.imgdata.params.output_bps) / maxV;
  cv::Mat color(raw.imgdata.sizes.iheight, raw.imgdata.sizes.iwidth, CV_8UC3);
  for (int r = 0; r < raw.imgdata.sizes.iheight; r++) {
    for (int c = 0; c < raw.imgdata.sizes.iwidth; c++) {
      cv::Vec3b &pixel = color.at<cv::Vec3b>(r, c);
      int i = r * raw.imgdata.sizes.iwidth + c;
      // RGBG to BGR
      // B
      pixel[0] = static_cast<uchar>(raw.imgdata.image[i][2] * scale);
      // G
      pixel[1] = static_cast<uchar>(
          (raw.imgdata.image[i][1] + raw.imgdata.image[i][3]) * 0.5f * scale);
      // R
      pixel[2] = static_cast<uchar>(raw.imgdata.image[i][0] * scale);
    }
  }

  return 0;
}