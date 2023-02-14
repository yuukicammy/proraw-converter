#include "raw_converter.hpp"
#include "test_common.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>

TEST(RawConverterTest, TestAdjustBrightnes_0) {
  yk::RawConverter rc;
  auto v = xt::arange<int>(0, (1 << 16), 1);
  auto data = xt::stack(xt::xtuple(v, v, v));
  for (int ch = 0; ch < 3; ch++) {
    EXPECT_EQ(data(ch, 255), 255);
    EXPECT_EQ(data(ch, USHRT_MAX), USHRT_MAX);
  }
  auto &&out = rc.adjust_brightness(data, 0);
  CLOSE_ALL(out, data);
}

TEST(RawConverterTest, TestAdjustBrightnes_1) {
  yk::RawConverter rc;
  auto v = xt::arange<int>(0, (1 << 16), 1);
  auto data = xt::stack(xt::xtuple(v, v, v));
  for (int ch = 0; ch < 3; ch++) {
    EXPECT_EQ(data(ch, 255), 255);
    EXPECT_EQ(data(ch, USHRT_MAX), USHRT_MAX);
  }
  auto &&out = rc.adjust_brightness(data, 1.);
  CLOSE_ALL(out, xt::zeros_like(data));
}

TEST(RawConverterTest, TestAdjustBrightnesInt) {
  yk::RawConverter rc;
  auto v = xt::arange<int>(0, (1 << 16), 1);
  auto data = xt::stack(xt::xtuple(v, v, v));
  for (int ch = 0; ch < 3; ch++) {
    EXPECT_EQ(data(ch, 255), 255);
    EXPECT_EQ(data(ch, USHRT_MAX), USHRT_MAX);
  }
  float thresh =
      static_cast<float>(8 * 10 * 2) / (1 << 16); // 20 bins (160 values)
  if (yk::DEBUG) {
    std::cout << "thresh: " << std::to_string(thresh) << std::endl;
  }
  auto &&outf = rc.adjust_brightness(data, thresh, yk::DEBUG);
  auto &&out = xt::clip(xt::floor(std::move(outf)), 0, USHRT_MAX);
  float alpha = static_cast<float>(USHRT_MAX) / (((1 << 13) - 10 - 1) * 8 - 80);
  float beta = -8 * 10 * alpha;
  if (yk::DEBUG) {
    std::cout << "alpha: " << std::to_string(alpha) << std::endl;
    std::cout << "beta: " << std::to_string(beta) << std::endl;
  }
  xt::xtensor<ushort, 2> ans({3, 1 << 16});
  for (int i = 0; i < (1 << 16); i++) {
    ans(0, i) = ans(1, i) = ans(2, i) = std::min<int>(
        USHRT_MAX, std::max<int>(0, static_cast<float>(i) * alpha + beta));
  }
  if (yk::DEBUG) {
    std::cout << rc.debug_message.str() << std::endl;
    rc.debug_message.str("");
  }
  EXPECT_EQ(xt::amax(ans)(), xt::amax(out)());
  EXPECT_EQ(xt::amin(ans)(), xt::amin(out)());

  CLOSE_ALL(out, ans);
}

TEST(RawConverterTest, TestAdjustBrightnesFloat) {
  yk::RawConverter rc;
  xt::xarray<float> v = {-1.1,    1 << 3,  1 << 4,    1 << 5,
                         1 << 6,  1 << 7,  1 << 8,    1 << 9,
                         1 << 10, 1 << 11, 1 << 12,   1 << 13,
                         1 << 14, 1 << 15, USHRT_MAX, USHRT_MAX + 1.3};
  auto data = xt::stack(xt::xtuple(v, v, v));
  float thresh = 0.5;
  if (yk::DEBUG) {
    std::cout << "thresh: " << std::to_string(thresh) << std::endl;
  }
  auto &&outf = rc.adjust_brightness(data, thresh, yk::DEBUG);
  auto &&out = xt::clip(xt::floor(std::move(outf)), 0, USHRT_MAX);
  float alpha = static_cast<float>(USHRT_MAX) / (16376 - 40);
  float beta = -40 * alpha;
  if (yk::DEBUG) {
    std::cout << "alpha: " << std::to_string(alpha) << std::endl;
    std::cout << "beta: " << std::to_string(beta) << std::endl;
  }
  xt::xtensor<ushort, 2> ans({3, data.shape()[1]});
  for (int i = 0; i < data.shape()[1]; i++) {
    ans(0, i) = ans(1, i) = ans(2, i) =
        std::min<int>(USHRT_MAX, std::max<int>(0, data(0, i) * alpha + beta));
  }
  if (yk::DEBUG) {
    std::cout << rc.debug_message.str() << std::endl;
    rc.debug_message.str("");
  }
  EXPECT_EQ(xt::amax(ans)(), xt::amax(out)());
  EXPECT_EQ(xt::amin(ans)(), xt::amin(out)());
  CLOSE_ALL(out, ans);
}