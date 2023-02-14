#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
// #include <netinet/in.h>
#include "experiment_common.hpp"
#include "raw_converter.hpp"
#include <boost/log/trivial.hpp>
#include <stdexcept>

int main(int argc, char *argv[]) {
  try {
    std::string input_filename;
    bool is_debug = false;
    for (int i = 1; i < argc; i++) {
      std::string arg{argv[i]};
      if (arg == "-D" || arg == "-d") {
        is_debug = true;
      } else if (input_filename.empty()) {
        input_filename = arg;
      }
    }
    yk::log_init(is_debug, "librawconversion-");

    LibRaw raw;

    // Open a ProRaw file.
    {
      int res = raw.open_file(input_filename.c_str());
      if (res == LIBRAW_SUCCESS) {
        BOOST_LOG_TRIVIAL(debug) << "LibRaw successfully reads the raw file."
                                 << "Filename: " << input_filename;
      } else {
        throw std::runtime_error("LibRaw failed to read file: " +
                                 input_filename);
      }

      // Initial processing
      res = raw.unpack();
      if (res != LIBRAW_SUCCESS) {
        throw std::runtime_error("LibRaw failed to unpack. file: " +
                                 input_filename);
      }
    }

    // LibRaw Conversion with the almost default parameters.
    {
      raw.imgdata.params.use_camera_wb = 1;
      BOOST_LOG_TRIVIAL(trace) << "Convert raw to sRGB with LibRaw using "
                                  "the default parameters.";
      BOOST_LOG_TRIVIAL(trace)
          << "- no_auto_bright: " << raw.imgdata.params.no_auto_bright;
      BOOST_LOG_TRIVIAL(trace)
          << "- use_auto_wb: " << raw.imgdata.params.use_auto_wb;
      BOOST_LOG_TRIVIAL(trace)
          << "- use_camera_wb: " << raw.imgdata.params.use_camera_wb;
      BOOST_LOG_TRIVIAL(trace)
          << "- use_camera_matrix: " << raw.imgdata.params.use_camera_matrix;
      BOOST_LOG_TRIVIAL(trace)
          << "- gamm[0]: " << std::to_string(raw.imgdata.params.gamm[0]);
      BOOST_LOG_TRIVIAL(trace)
          << "- gamm[1]: " << std::to_string(raw.imgdata.params.gamm[1]);
      double total_elapsed = 0;
      {
        auto &&start = std::chrono::system_clock::now();
        int ret = raw.dcraw_process();
        if (ret != LIBRAW_SUCCESS) {
          throw std::runtime_error("LibRaw failed in dcraw_process().");
        }
        auto &&end = std::chrono::system_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        total_elapsed += elapsed;
        BOOST_LOG_TRIVIAL(debug) << "Run time of LibRaw dcraw_process(): "
                                 << std::to_string(elapsed);
      }
      {
        std::stringstream ss;
        ss << argv[1] << ".libraw_rgb_default.TIFF";
        auto &&start = std::chrono::system_clock::now();
        int ret = raw.dcraw_ppm_tiff_writer(ss.str().c_str());
        if (ret != LIBRAW_SUCCESS) {
          BOOST_LOG_TRIVIAL(error) << "Cannot write " << ss.str();
        }
        auto &&end = std::chrono::system_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        total_elapsed += elapsed;
        BOOST_LOG_TRIVIAL(debug)
            << "Run time of LibRaw dcraw_ppm_tiff_writer(): "
            << std::to_string(elapsed);
        BOOST_LOG_TRIVIAL(debug)
            << "Done LibRaw conversion from raw to sRGB. "
            << "Total Run time: " << std::to_string(total_elapsed);
      }
    }

    // Save a image converted by LibRaw through OpenCV.
    {
      BOOST_LOG_TRIVIAL(trace) << "Saving a conversion result through OpenCV.";
      cv::Mat cv_libraw_image(raw.imgdata.sizes.iheight,
                              raw.imgdata.sizes.iwidth, CV_8UC3);
      for (int r = 0, i = 0; r < cv_libraw_image.rows; r++) {
        for (int c = 0; c < cv_libraw_image.cols; c++, i++) {
          auto &pixel = cv_libraw_image.at<cv::Vec3b>(r, c);
          // B
          pixel[0] = static_cast<uchar>(
              (raw.imgdata.color.curve[raw.imgdata.image[i][2]] >> 8));
          // G
          pixel[1] = static_cast<uchar>(
              (raw.imgdata.color.curve[raw.imgdata.image[i][1]] >> 8));
          // R
          pixel[2] = static_cast<uchar>(
              (raw.imgdata.color.curve[raw.imgdata.image[i][0]] >> 8));
        }
      }
      std::stringstream ss;
      ss << argv[1] << ".libraw_to_opencv.png";
      cv::imwrite(ss.str(), cv_libraw_image);
      BOOST_LOG_TRIVIAL(trace) << "Saved image: " << ss.str();
    }

    // LibRaw Conversion with almost the same parameters.
    {
      BOOST_LOG_TRIVIAL(trace) << "Convert raw to sRGB with LibRaw using "
                                  "almost the same parameters.";

      raw.imgdata.params.output_bps = 8;
      raw.imgdata.params.output_tiff = 1;
      raw.imgdata.params.user_flip = 0;
      raw.imgdata.params.no_auto_bright = 0; // VERY IMPORTANT
      raw.imgdata.params.half_size = 0;
      raw.imgdata.params.use_auto_wb = 0;
      raw.imgdata.params.use_camera_wb = 1;
      raw.imgdata.params.use_camera_matrix = 1;

      // For sRGB
      raw.imgdata.params.gamm[0] = 1 / 2.4f;
      raw.imgdata.params.gamm[1] = 12.92;

      BOOST_LOG_TRIVIAL(trace)
          << "- no_auto_bright: " << raw.imgdata.params.no_auto_bright;
      BOOST_LOG_TRIVIAL(trace)
          << "- use_auto_wb: " << raw.imgdata.params.use_auto_wb;
      BOOST_LOG_TRIVIAL(trace)
          << "- use_camera_wb: " << raw.imgdata.params.use_camera_wb;
      BOOST_LOG_TRIVIAL(trace)
          << "- use_camera_matrix: " << raw.imgdata.params.use_camera_matrix;
      BOOST_LOG_TRIVIAL(trace)
          << "- gamm[0]: " << std::to_string(raw.imgdata.params.gamm[0]);
      BOOST_LOG_TRIVIAL(trace)
          << "- gamm[1]: " << std::to_string(raw.imgdata.params.gamm[1]);
      double total_elapsed = 0;
      {
        auto &&start = std::chrono::system_clock::now();
        int ret = raw.dcraw_process();
        if (ret != LIBRAW_SUCCESS) {
          throw std::runtime_error("LibRaw failed in dcraw_process().");
        }
        auto &&end = std::chrono::system_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        total_elapsed += elapsed;
        BOOST_LOG_TRIVIAL(debug) << "Run time of LibRaw dcraw_process(): "
                                 << std::to_string(elapsed);
      }
      {
        std::stringstream ss;
        ss << argv[1] << ".libraw_rgb.TIFF";
        auto &&start = std::chrono::system_clock::now();
        int ret = raw.dcraw_ppm_tiff_writer(ss.str().c_str());
        if (ret != LIBRAW_SUCCESS) {
          BOOST_LOG_TRIVIAL(error) << "Cannot write " << ss.str();
        }
        auto &&end = std::chrono::system_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        total_elapsed += elapsed;
        BOOST_LOG_TRIVIAL(debug)
            << "Run time of LibRaw dcraw_ppm_tiff_writer(): "
            << std::to_string(elapsed);
        BOOST_LOG_TRIVIAL(debug)
            << "Done LibRaw conversion from raw to sRGB. "
            << "Total Run time: " << std::to_string(total_elapsed);
      }
    }

    return 0;
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    BOOST_LOG_TRIVIAL(fatal) << e.what();
    return 1;
  }
}