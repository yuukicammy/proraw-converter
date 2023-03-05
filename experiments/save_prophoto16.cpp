#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>
// #include <netinet/in.h>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <libraw.h>
#include <stdexcept>

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
  // if (!std::filesystem::exists(logdir)) {
  //   std::filesystem::create_directory(logdir);
  // }
  boost::log::add_file_log(
      boost::log::keywords::file_name =
          logdir + "/" + file_prefix + "%Y-%m-%d-%H-%M-%S.txt",
      boost::log::keywords::format =
          "[%TimeStamp%] [%ThreadID%] [%Severity%] %Message%");
  boost::log::add_common_attributes();
}
} // namespace yk

int main(int argc, char *argv[]) {
  try {

    const std::string input_filename = argv[1];

    yk::log_init(true, "save_prophoto16-");

    LibRaw raw;
    raw.imgdata.params.output_bps = 16;
    raw.imgdata.params.output_tiff = 1;
    raw.imgdata.params.no_auto_bright = 1; // VERY IMPORTANT
    raw.imgdata.params.half_size = 0;
    raw.imgdata.params.use_auto_wb = 0;
    raw.imgdata.params.no_auto_scale = 1;
    raw.imgdata.params.use_camera_wb = 1;
    raw.imgdata.params.use_camera_matrix = 1;
    raw.imgdata.params.gamm[0] = 1;
    raw.imgdata.params.gamm[1] = 1;
    raw.imgdata.params.output_color = 4; // ProPhoto

    std::ifstream input_file(input_filename);
    if (!input_file.is_open()) {
      std::cerr << "Could not open the file - '" << input_filename << "'"
                << std::endl;
      return 1;
    }

    std::vector<std::string> raw_paths;
    std::vector<std::string> expert_paths;
    {
      std::string line;
      while (std::getline(input_file, line)) {
        std::string raw_path;
        auto it = line.begin();
        while (*it != ',') {
          raw_path.push_back(*it);
          it++;
        }
        raw_paths.push_back(raw_path);
        it++;
        std::string exp_path;
        while (*it != ',') {
          exp_path.push_back(*it);
          it++;
        }
        expert_paths.push_back(exp_path);
      }
    }
    std::vector<std::size_t> indices(raw_paths.size());
    {
      std::iota(indices.begin(), indices.end(), 0);
      std::mt19937 get_rand_mt(42);
      std::shuffle(indices.begin(), indices.end(), get_rand_mt);
    }

    // Open a ProRaw file.
    auto it = indices.begin();
    while (it != indices.end()) {
      auto id = *it;
      cv::Mat target = cv::imread(expert_paths[id]);
      const int target_height = target.rows;
      const int target_width = target.cols;
      target.release();

      BOOST_LOG_TRIVIAL(info) << "Expert image size: (" << target_height << ", "
                              << target_width << ")";

      int res = raw.open_file(raw_paths[id].c_str());
      // auto target = cv::imread(expert_paths[i]);
      if (res == LIBRAW_SUCCESS) {
        BOOST_LOG_TRIVIAL(info) << "LibRaw successfully reads the raw file."
                                << "Filename: " << raw_paths[id];
      } else {
        throw std::runtime_error("LibRaw failed to read file: " +
                                 raw_paths[id]);
      }

      // Initial processing
      raw.unpack();

      BOOST_LOG_TRIVIAL(info)
          << "RAW Original image size: (" << raw.imgdata.sizes.height << ", "
          << raw.imgdata.sizes.width << ")";

      unsigned mask = 0;
      int mask_id = 0;
      if (0 < raw.imgdata.sizes.raw_inset_crops[0].cleft) {
        mask = 1;
        mask_id = 0;
      } else {
        mask = 3;
        mask_id = 1;
      }

      int expected_width = raw.imgdata.sizes.raw_inset_crops[mask_id].cwidth;
      int expected_height = raw.imgdata.sizes.raw_inset_crops[mask_id].cheight;

      if (expected_width < target_width &&
          (target_width - expected_width) < 2) {
        raw.imgdata.sizes.raw_inset_crops[mask_id].cleft--;
        raw.imgdata.sizes.raw_inset_crops[mask_id].cwidth++;
      }
      if (expected_height < target_height &&
          (target_height - expected_height) < 2) {
        raw.imgdata.sizes.raw_inset_crops[mask_id].ctop--;
        raw.imgdata.sizes.raw_inset_crops[mask_id].cheight++;
      }
      raw.adjust_to_raw_inset_crop(mask, 0.1f);

      raw.dcraw_process();
      BOOST_LOG_TRIVIAL(info) << "RAW image size: (" << raw.imgdata.sizes.height
                              << ", " << raw.imgdata.sizes.width << ")";
      std::stringstream ss;
      ss << raw_paths[id] << ".libraw.TIFF";
      // raw.dcraw_ppm_tiff_writer(ss.str().c_str());
      // cv::imwrite(ss.str(), raw_img);

      it++;
    }
    return 0;
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    BOOST_LOG_TRIVIAL(fatal) << e.what();
    return 1;
  }
}