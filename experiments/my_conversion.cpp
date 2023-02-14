#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
// #include <netinet/in.h>
#include "experiment_common.hpp"
#include "raw_converter.hpp"
#include <boost/log/trivial.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <stdexcept>

int main(int argc, char *argv[]) {
  try {
    cxxopts::Options options(
        "ProRaw Converter",
        "The program 1) converts a ProRaw image in sRGB' color space, 2) "
        "[optional] "
        "adjusts the brightness and contrasts, 3) applys gamma correction, and "
        "then 4) saves the result in PNG format. ");

    options.add_options()("raw,r", "Save the raw image",
                          cxxopts::value<bool>())(
        "f,file", "ProRaw file path", cxxopts::value<std::string>())(
        "d,debug", "Enable debugging",
        cxxopts::value<bool>())("t,thresh", "Threshold of histogram stretching",
                                cxxopts::value<float>()->default_value("0.04"))(
        "a,adjust", "Adjust the brightness and contrast",
        cxxopts::value<bool>())("h,help", "Print usage");
    options.parse_positional({"file"});
    options.positional_help("ProRawFilePath");

    auto args = options.parse(argc, argv);
    if (args.count("help")) {
      std::cout << options.help() << std::endl;
      options.show_positional_help();
      return 0;
    }

    const std::string input_filename = args["file"].as<std::string>();
    const bool is_debug = args["debug"].as<bool>();
    const bool save_raw = args["raw"].as<bool>();
    const bool apply_adjustment = args["adjust"].as<bool>();
    const float threshold = apply_adjustment ? args["thresh"].as<float>() : 0.;

    yk::log_init(is_debug, "myconversion-");
    BOOST_LOG_TRIVIAL(debug) << "Threshold: " << std::to_string(threshold);

    LibRaw raw;

    // Open a ProRaw file through LibRaw
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

    if (save_raw) {
      BOOST_LOG_TRIVIAL(trace)
          << "Saving ProRaw values directly as a 8-bit PNG "
             "image with OpenCV.";
      // ProRaw values are converted directly into a 8-bit image.
      cv::Mat &&cv_raw_image = yk::ToCvMat3b(image, raw.imgdata.sizes.iheight,
                                             raw.imgdata.sizes.iwidth);
      std::stringstream ss;
      ss << input_filename << ".cv_raw.png";
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
    {
      double total_elapsed = 0;
      BOOST_LOG_TRIVIAL(trace)
          << "Original image[:, " << image.shape()[1] / 2
          << "]: " << xt::view(image, xt::all(), image.shape()[1] / 2);

      BOOST_LOG_TRIVIAL(trace)
          << "Converting raw from camera native color space to sRGB' (16-bit).";
      auto &&start = std::chrono::system_clock::now();
      // Camera native color space to sRGB'
      auto &&srgb_ = rc.camera_to_sRGB(image, raw.imgdata.color.rgb_cam);
      auto &&end = std::chrono::system_clock::now();
      double elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      BOOST_LOG_TRIVIAL(trace)
          << "After cam-to-sRGB' image[:, " << image.shape()[1] / 2
          << "]: " << xt::view(srgb_, xt::all(), image.shape()[1] / 2);

      BOOST_LOG_TRIVIAL(debug) << "Done conversion camera native color space "
                                  "to CIE-XYZ color space. "
                               << "Run time: " << std::to_string(elapsed);
      total_elapsed += elapsed;

      BOOST_LOG_TRIVIAL(trace) << "Adjusting the brightness and contrast.";
      if (!apply_adjustment) {
        BOOST_LOG_TRIVIAL(debug) << "adjust_brightness() is called, but "
                                    "the data is not stretched.";
      }
      start = std::chrono::system_clock::now();
      auto &&srgb_adj = rc.adjust_brightness(srgb_, threshold, is_debug);
      end = std::chrono::system_clock::now();
      elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      BOOST_LOG_TRIVIAL(trace) << rc.debug_message.str();
      rc.debug_message.str("");
      BOOST_LOG_TRIVIAL(trace)
          << "After adjustment image[:, " << image.shape()[1] / 2
          << "]: " << xt::view(srgb_adj, xt::all(), image.shape()[1] / 2);
      BOOST_LOG_TRIVIAL(debug) << "Done adjusting the brightness and contrast. "
                               << "Run time: " << std::to_string(elapsed);
      total_elapsed += elapsed;

      // Gamma Correction
      start = std::chrono::system_clock::now();
      auto &&sRGB = rc.gamma_correction(srgb_adj);
      end = std::chrono::system_clock::now();
      elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      BOOST_LOG_TRIVIAL(debug) << "Done gamma correction. "
                               << "Run time: " << std::to_string(elapsed);
      total_elapsed += elapsed;
      BOOST_LOG_TRIVIAL(trace)
          << "After gamma correction image[:, " << image.shape()[1] / 2
          << "]: " << xt::view(sRGB, xt::all(), image.shape()[1] / 2);

      BOOST_LOG_TRIVIAL(debug)
          << "Done conversion from raw to sRGB with the brightness and "
             "contract adjustment. "
          << "Total Run time: " << std::to_string(total_elapsed);
      {
        BOOST_LOG_TRIVIAL(trace)
            << "Saving a conversion result through OpenCV.";
        cv::Mat &&rgb_image = yk::ToCvMat3b(sRGB, raw.imgdata.sizes.iheight,
                                            raw.imgdata.sizes.iwidth);
        std::stringstream ss;
        if (apply_adjustment) {
          ss << input_filename << ".cv_rgb_adj_" << std::to_string(threshold)
             << ".png";
        } else {
          ss << input_filename << ".cv_rgb_no_adj.png";
        }
        cv::imwrite(ss.str(), rgb_image);
        BOOST_LOG_TRIVIAL(trace) << "Saved image: " << ss.str();
      }
    }
    return 0;
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    BOOST_LOG_TRIVIAL(fatal) << e.what();
    return 1;
  }
}