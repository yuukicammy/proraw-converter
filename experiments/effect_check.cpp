
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
        "The program 1)[optional] converts a ProRaw image in sRGB' color "
        "space, 2) "
        "[optional] "
        "adjusts the brightness and contrasts, 3) [optional] applys gamma "
        "correction, and "
        "then 4) saves the result in PNG format. \n"
        "If you do not want to adjust brightness and contrast, do not specify "
        "-a option or specify -a 0.");

    options.add_options()("raw,r", "Save the raw image",
                          cxxopts::value<bool>())(
        "f,file", "ProRaw file path", cxxopts::value<std::string>())(
        "d,debug", "Enable debugging. Log file is output to ../logs/.",
        cxxopts::value<bool>())(
        "a,alpha", "", cxxopts::value<float>()->default_value("0.001"))(
        "m,measure", "Measure execution speed", cxxopts::value<bool>())(
        "c,color", "Apply color correction", cxxopts::value<bool>())(
        "g,gamma", "Apply gamma correction",
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
    const bool measure_speed = args["measure"].as<bool>();
    const bool apply_color_correction = args["color"].as<bool>();
    const bool apply_gamma_correction = args["gamma"].as<bool>();
    const float alpha = args["alpha"].as<float>();

    yk::log_init(is_debug, "effectcheck-");

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

    yk::RawConverter rc{};
    rc.raw_adjust(image);

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

      xt::xtensor<float, 2> result({3, image.shape()[1]});
      result = image;

      if (apply_color_correction) {
        BOOST_LOG_TRIVIAL(trace) << "Converting raw from camera native color "
                                    "space to sRGB' (16-bit).";
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

        BOOST_LOG_TRIVIAL(debug)
            << "Done conversion from camera native color space "
               "to sRGB'. "
            << "Run time (ms): " << std::to_string(elapsed);
        if (measure_speed) {
          std::cout << "Done conversion from camera native color space "
                       "to sRGB'. "
                    << std::endl;
          std::cout << " -- Run time (ms): " << std::to_string(elapsed)
                    << std::endl;
        }
        total_elapsed += elapsed;
        result = xt::cast<ushort>(xt::clip(srgb_, 0, USHRT_MAX));
      }

      if (0.000001 < alpha) {
        BOOST_LOG_TRIVIAL(trace) << "Adjusting the brightness and contrast.";
        auto start = std::chrono::system_clock::now();
        auto &&srgb_adj = rc.adjust_brightness(result, alpha, is_debug);
        auto end = std::chrono::system_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        BOOST_LOG_TRIVIAL(trace) << rc.debug_message.str();
        rc.debug_message.str("");
        BOOST_LOG_TRIVIAL(trace)
            << "After adjustment image[:, " << image.shape()[1] / 2
            << "]: " << xt::view(srgb_adj, xt::all(), image.shape()[1] / 2);
        BOOST_LOG_TRIVIAL(debug)
            << "Done adjusting the brightness and contrast. "
            << "Run time (ms): " << std::to_string(elapsed);
        if (measure_speed) {
          std::cout << "Done adjusting the brightness and contrast."
                    << std::endl;
          std::cout << " -- Run time (ms): " << std::to_string(elapsed)
                    << std::endl;
        }
        total_elapsed += elapsed;
        result = xt::cast<ushort>(xt::clip(srgb_adj, 0, USHRT_MAX));
      }

      if (apply_gamma_correction) {
        // Gamma Correction
        auto start = std::chrono::system_clock::now();
        auto &&sRGB = rc.gamma_correction(result);
        auto end = std::chrono::system_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        BOOST_LOG_TRIVIAL(debug)
            << "Done gamma correction. "
            << "Run time (ms): " << std::to_string(elapsed);
        if (measure_speed) {
          std::cout << "Done gamma correction." << std::endl;
          std::cout << " -- Run time (ms): " << std::to_string(elapsed)
                    << std::endl;
        }
        total_elapsed += elapsed;
        BOOST_LOG_TRIVIAL(trace)
            << "After gamma correction image[:, " << image.shape()[1] / 2
            << "]: " << xt::view(sRGB, xt::all(), image.shape()[1] / 2);
        result = sRGB;
      }

      BOOST_LOG_TRIVIAL(debug)
          << "Done conversion from ProRaw to sRGB with the brightness and "
             "contract adjustment. "
          << "Total Run time (ms): " << std::to_string(total_elapsed);
      if (measure_speed) {
        std::cout << "Done all conversion." << std::endl;
        std::cout << " -- Total run time (ms): "
                  << std::to_string(total_elapsed) << std::endl;
      }
      {
        BOOST_LOG_TRIVIAL(trace)
            << "Saving a conversion result through OpenCV.";
        cv::Mat &&rgb_image = yk::ToCvMat3b(result, raw.imgdata.sizes.iheight,
                                            raw.imgdata.sizes.iwidth);
        std::stringstream ss;
        ss << input_filename << ".cv_";
        if (apply_color_correction) {
          ss << "c_";
        }
        if (apply_gamma_correction) {
          ss << "g_";
        }
        if (0.000001 < alpha) {
          ss << "adj_" << std::to_string(alpha);
        } else {
          ss << "no_adj";
        }
        ss << ".png";
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