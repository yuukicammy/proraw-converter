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

std::string basename(const std::string &path) {
  auto str = path.substr(path.find_last_of('/') + 1);
  return str.substr(0, str.find_last_of('.'));
}

} // namespace yk

int main(int argc, char *argv[]) {
  try {
    int n_first_skip = 1692 + 130 + 13;
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
    int skip_cnt = 0;
    auto it = indices.begin();
    while (0 < n_first_skip) {
      n_first_skip--;
      it++;
    }
    while (it != indices.end()) {
      auto id = *it;
      cv::Mat target = cv::imread(expert_paths[id]);
      const int target_height = target.rows;
      const int target_width = target.cols;
      target.release();
      int retry = 3;
      while (0 <= retry) {
        int res = raw.open_file(raw_paths[id].c_str());
        if (res != LIBRAW_SUCCESS) {
          if (retry == 0) {
            throw std::runtime_error("LibRaw failed to read file: " +
                                     raw_paths[id]);
          } else {
            retry--;
          }
        } else {
          break;
        }
      }
      // Initial processing
      raw.unpack();

      raw.adjust_to_raw_inset_crop(1, 0.1f);

      ushort image_width = raw.imgdata.sizes.width;
      ushort image_height = raw.imgdata.sizes.height;

      if (4 < raw.imgdata.sizes.flip) {
        std::swap(image_width, image_height);
      }

      if (1 < std::abs(image_width - target_width) ||
          1 < std::abs(image_height - target_height)) {
        BOOST_LOG_TRIVIAL(info) << skip_cnt << " Skip " << raw_paths[id];
        BOOST_LOG_TRIVIAL(info) << " raw size: (" << raw.imgdata.sizes.height
                                << ", " << raw.imgdata.sizes.width << ")";
        BOOST_LOG_TRIVIAL(info) << " target size: (" << target_height << ", "
                                << target_width << ")";
        BOOST_LOG_TRIVIAL(info) << "";
        skip_cnt++;
        it++;
        continue;
      }

      raw.dcraw_process();

      std::string fileid = yk::basename(raw_paths[id]);

      std::stringstream ss;
      ss << "/mnt/disks/data/MITAboveFiveK/processed/"
            "libraw_linear_prophoto16/"
         << fileid << ".TIFF";
      raw.dcraw_ppm_tiff_writer(ss.str().c_str());
      BOOST_LOG_TRIVIAL(info)
	<< std::distance(indices.begin(), it) << " Saved: " << ss.str();
      it++;
    }
    return 0;
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    BOOST_LOG_TRIVIAL(fatal) << e.what();
    return 1;
  }
}
