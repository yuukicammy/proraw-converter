#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
// #include <netinet/in.h>
#include "raw_converter.hpp"
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <filesystem>
#include <iostream>
#include <stdexcept>

// Set up logger
static void log_init() {
  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::trace);

  static const std::string logdir{"../logs"};
  if (!std::filesystem::exists(logdir)) {
    std::filesystem::create_directory(logdir);
  }
  boost::log::add_file_log(
      boost::log::keywords::file_name = logdir + "/%Y-%m-%d-%H-%M-%S.txt",
      boost::log::keywords::format =
          "[%TimeStamp%] [%ThreadID%] [%Severity%] %Message%");
  boost::log::add_common_attributes();
}

int main(int argc, char *argv[]) {
  try {
    log_init();

    std::string input_filename = argv[1];

    LibRaw raw;

    // Check whether iPhone DNG file.
    int res = raw.open_file(input_filename.c_str());
    if (res == LIBRAW_SUCCESS) {
      BOOST_LOG_TRIVIAL(debug) << "LibRaw successfully reads the raw file."
                               << "Filename: " << input_filename;
    } else {
      throw std::runtime_error("LibRaw failed to read file: " + input_filename);
    }

    // Initial processing
    res = raw.unpack();
    if (res != LIBRAW_SUCCESS) {
      throw std::runtime_error("LibRaw failed to unpack. file: " +
                               input_filename);
    }

    // From LibRaw raw file to xtensor
    xt::xtensor<ushort, 2> image({4, (std::size_t)(raw.imgdata.sizes.iheight *
                                                   raw.imgdata.sizes.iwidth)});
    for (int i = 0; i < image.shape()[1]; i++) {
      xt::view(image, xt::all(), i) =
          xt::adapt(raw.imgdata.rawdata.color4_image[i], {4});
    }
    image = xt::view(image, xt::range(0, 3), xt::all());
    {
      std::stringstream ss;
      ss << "Raw image shape: ";
      std::for_each(image.shape().begin(), image.shape().end(),
                    [&ss](auto v) { ss << v << " "; });
      BOOST_LOG_TRIVIAL(debug) << ss.str();
    }

    // ProRaw values are converted directly into a 8-bit image
    {
      BOOST_LOG_TRIVIAL(trace) << "Saving ProRaw values directory as 8-bit PNG "
                                  "image with OpenCV.";
      cv::Mat &&cv_raw_image = yk::ToCvMat3b(image, raw.imgdata.sizes.iheight,
                                             raw.imgdata.sizes.iwidth);
      std::stringstream ss;
      ss << argv[1] << ".cv_raw.png";
      cv::imwrite(ss.str(), cv_raw_image);
      BOOST_LOG_TRIVIAL(trace) << "Saved image: " << ss.str();
    }

    yk::RawConverter rc{};

    // Subtract Black Level
    // The black level is stored in DNG metadata.
    {
      BOOST_LOG_TRIVIAL(debug)
          << "Black Level: " << raw.imgdata.color.black << std::endl;

      std::stringstream ss;
      ss << "Black Levels: ";
      for (int ch = 0; ch < 3; ch++) {
        ss << raw.imgdata.color.cblack[ch] << ", ";
      }
      BOOST_LOG_TRIVIAL(debug) << ss.str();

      BOOST_LOG_TRIVIAL(trace) << "Subtract black level.";
      rc.subtract_black(image, raw.imgdata.color.black,
                        raw.imgdata.color.cblack);
    }

    // Convert raw image to sRGB.
    xt::xtensor<ushort, 2> sRGB;
    {
      BOOST_LOG_TRIVIAL(trace)
          << "Convert raw from camera native color space to sRGB (16-bit).";

      auto &&start = std::chrono::system_clock::now();

      // Camera native color space to sRGB'
      auto &&srgb_ = rc.camera_to_sRGB(image, raw.imgdata.color.rgb_cam);

      // Gamma Correction
      auto srgb = rc.gamma_correction(std::move(srgb_));

      auto &&end = std::chrono::system_clock::now();
      double elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      BOOST_LOG_TRIVIAL(debug)
          << "Done convertion from camera native color space to sRGB. "
          << "Run time: " << std::to_string(elapsed);
      sRGB = srgb;
      {
        BOOST_LOG_TRIVIAL(trace)
            << "Saving a conversion result through OpenCV.";
        cv::Mat &&rgb_image = yk::ToCvMat3b(srgb, raw.imgdata.sizes.iheight,
                                            raw.imgdata.sizes.iwidth);
        std::stringstream ss;
        ss << argv[1] << ".cv_rgb.png";
        cv::imwrite(ss.str(), rgb_image);
        BOOST_LOG_TRIVIAL(trace) << "Saved image: " << ss.str();
      }
    }

    // Convertion with adjustment of the brightness and the contrast.
    {
      double total_elapsed = 0;

      BOOST_LOG_TRIVIAL(trace)
          << "Convert from camera native color space to CIE-XYZ color "
             "space (16-bit).";
      auto &&start = std::chrono::system_clock::now();
      auto &&xyz =
          rc.camera_to_xyz(image, raw.imgdata.color.dng_color[1].colormatrix,
                           raw.imgdata.color.dng_levels.analogbalance);
      auto &&end = std::chrono::system_clock::now();
      double elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      BOOST_LOG_TRIVIAL(debug) << "Done convertion camera native color space "
                                  "to CIE-XYZ color space. "
                               << "Run time: " << std::to_string(elapsed);
      total_elapsed += elapsed;

      BOOST_LOG_TRIVIAL(trace) << "Adjust the brightness and contrast.";
      start = std::chrono::system_clock::now();
      auto &&xyz_adj = rc.ajust_brightness(xyz);
      end = std::chrono::system_clock::now();
      elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      BOOST_LOG_TRIVIAL(debug) << "Done adjusting the brightness and contrast. "
                               << "Run time: " << std::to_string(elapsed);
      total_elapsed += elapsed;

      BOOST_LOG_TRIVIAL(trace) << "Convert from CIE-XYZ color space to sRGB'. ";
      start = std::chrono::system_clock::now();
      auto srgb_ = rc.xyz_to_sRGB(xyz_adj);
      end = std::chrono::system_clock::now();
      elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      BOOST_LOG_TRIVIAL(debug)
          << "Done conversion from CIE-XYZ color space to sRGB'. "
          << "Run time: " << std::to_string(elapsed);
      total_elapsed += elapsed;

      // Gamma Correction
      BOOST_LOG_TRIVIAL(trace) << "Gamma Correction";
      start = std::chrono::system_clock::now();
      auto &&srgb = rc.gamma_correction(std::move(srgb_));
      end = std::chrono::system_clock::now();
      elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      BOOST_LOG_TRIVIAL(debug) << "Done gamma correction. "
                               << "Run time: " << std::to_string(elapsed);
      total_elapsed += elapsed;

      BOOST_LOG_TRIVIAL(debug)
          << "Done conversion from raw to sRGB with the brightness and "
             "contract adjustment. "
          << "Total Run time: " << std::to_string(total_elapsed);
      {
        BOOST_LOG_TRIVIAL(trace)
            << "Saving a conversion result through OpenCV.";
        cv::Mat &&rgb_image = yk::ToCvMat3b(srgb, raw.imgdata.sizes.iheight,
                                            raw.imgdata.sizes.iwidth);

        std::stringstream ss;
        ss << argv[1] << ".cv_rgb_adj.png";
        cv::imwrite(ss.str(), rgb_image);
        BOOST_LOG_TRIVIAL(trace) << "Saved image: " << ss.str();
      }
    }

    // LibRaw Conversion with almost the same parameters.
    {
      BOOST_LOG_TRIVIAL(trace) << "Convert raw to sRGB with LibRaw using "
                                  "almost the same parameters.";

      raw.imgdata.params.output_bps = 8;
      raw.imgdata.params.output_tiff = 1;
      raw.imgdata.params.user_flip = 0;
      raw.imgdata.params.no_auto_bright = 1; // VERY IMPORTANT
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
        if (raw.dcraw_process() != LIBRAW_SUCCESS) {
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
        if (raw.dcraw_ppm_tiff_writer(ss.str().c_str()) != LIBRAW_SUCCESS) {
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
          pixel[3] = static_cast<uchar>(
              (raw.imgdata.color.curve[raw.imgdata.image[i][0]] >> 8));
        }
      }
      std::stringstream ss;
      ss << argv[1] << ".libraw_to_opencv.png";
      cv::imwrite(ss.str(), cv_libraw_image);
      BOOST_LOG_TRIVIAL(trace) << "Saved image: " << ss.str();
    }

    // Save a my conversion result through LibRaw as TIFF.
    {
      BOOST_LOG_TRIVIAL(trace) << "Save a my conversion result through "
                                  "LibRaw without auto bright.";
      for (int i = 0; i < sRGB.shape()[1]; i++) {
        for (int ch = 0; ch < 3; ch++) {
          raw.imgdata.image[i][ch] = sRGB(ch, i);
        }
      }
      raw.imgdata.params.output_bps = 8;
      raw.imgdata.params.output_tiff = 1;
      raw.imgdata.params.user_flip = 0;
      raw.imgdata.params.no_auto_bright = 1; // VERY IMPORTANT
      raw.imgdata.params.half_size = 0;
      raw.imgdata.params.use_auto_wb = 0;
      raw.imgdata.params.use_camera_wb = 0;
      raw.imgdata.params.gamm[0] = 1;
      raw.imgdata.params.gamm[1] = 1;
      raw.imgdata.params.use_camera_matrix = 0;

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

      std::stringstream ss;
      ss << argv[1] << ".my_to_libraw.TIFF";
      auto &&start = std::chrono::system_clock::now();
      int ret = raw.dcraw_ppm_tiff_writer(ss.str().c_str());
      auto &&end = std::chrono::system_clock::now();
      if (LIBRAW_SUCCESS != ret) {
        BOOST_LOG_TRIVIAL(error) << "Cannot write " << ss.str() << ": "
                                 << std::string(libraw_strerror(ret));
      }
      double elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      BOOST_LOG_TRIVIAL(debug) << "Run time of LibRaw dcraw_ppm_tiff_writer(): "
                               << std::to_string(elapsed);
    }

    // Save a my conversion result through LibRaw with auto-bright as TIFF.
    {
      BOOST_LOG_TRIVIAL(trace) << "Save a my conversion result through "
                                  "LibRaw with auto bright.";
      for (int i = 0; i < sRGB.shape()[1]; i++) {
        for (int ch = 0; ch < 3; ch++) {
          raw.imgdata.image[i][ch] = sRGB(ch, i);
        }
      }
      raw.imgdata.params.output_bps = 8;
      raw.imgdata.params.output_tiff = 1;
      raw.imgdata.params.user_flip = 0;
      raw.imgdata.params.no_auto_bright = 0; // VERY IMPORTANT
      raw.imgdata.params.half_size = 0;
      raw.imgdata.params.use_auto_wb = 0;
      raw.imgdata.params.use_camera_wb = 0;
      raw.imgdata.params.gamm[0] = 1;
      raw.imgdata.params.gamm[1] = 1;
      raw.imgdata.params.use_camera_matrix = 0;

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

      std::stringstream ss;
      ss << argv[1] << ".my_to_libraw_adj.TIFF";
      auto &&start = std::chrono::system_clock::now();
      int ret = raw.dcraw_ppm_tiff_writer(ss.str().c_str());
      auto &&end = std::chrono::system_clock::now();
      if (LIBRAW_SUCCESS != ret) {
        BOOST_LOG_TRIVIAL(error) << "Cannot write " << ss.str() << ": "
                                 << std::string(libraw_strerror(ret));
      }
      double elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      BOOST_LOG_TRIVIAL(debug) << "Run time of LibRaw dcraw_ppm_tiff_writer(): "
                               << std::to_string(elapsed);
    }
    return 0;
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    BOOST_LOG_TRIVIAL(fatal) << e.what();
    return 1;
  }
}