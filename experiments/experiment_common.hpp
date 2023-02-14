#pragma ones

#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <xtensor/xtensor.hpp>

namespace yk {
// Set up logger
static void log_init(const bool is_debug, const std::string file_prefix = "") {
  if (is_debug) {
    boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                        boost::log::trivial::trace);
  } else {
    boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                        boost::log::trivial::info);
  }
  static const std::string logdir{"../logs"};
  if (!std::filesystem::exists(logdir)) {
    std::filesystem::create_directory(logdir);
  }
  boost::log::add_file_log(
      boost::log::keywords::file_name =
          logdir + "/" + file_prefix + "%Y-%m-%d-%H-%M-%S.txt",
      boost::log::keywords::format =
          "[%TimeStamp%] [%ThreadID%] [%Severity%] %Message%");
  boost::log::add_common_attributes();
}

auto ToCvMat3b(const xt::xtensor<ushort, 2> &src, const std::size_t rows,
               const std::size_t cols) noexcept {
  cv::Mat dst(rows, cols, CV_8UC3);
  for (int r = 0, i = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++, i++) {
      auto &pixel = dst.at<cv::Vec3b>(r, c);
      // B
      pixel[0] = src(2, i) >> 8;

      //  G
      pixel[1] = src(1, i) >> 8;

      //  R
      pixel[2] = src(0, i) >> 8;
    }
  }
  return dst;
}
} // namespace yk