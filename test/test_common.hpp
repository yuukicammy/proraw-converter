#pragma ones
#include <gtest/gtest.h>

namespace yk {
static constexpr bool DEBUG = false;
}

template <typename T0, typename T1> void CLOSE_ALL(T0 &&x0, T1 &&x1) {
  auto itr0 = x0.cbegin();
  auto itr1 = x1.cbegin();
  for (; itr0 != x0.cend() && itr1 != x1.cend(); itr0++, itr1++) {
    EXPECT_FLOAT_EQ(*itr0, *itr1);
  }
}

template <typename T0, typename T1>
void CLOSE_ALL(T0 &&x0, T1 &&x1, double abs_error) {
  auto itr0 = x0.cbegin();
  auto itr1 = x1.cbegin();
  for (; itr0 != x0.cend() && itr1 != x1.cend(); itr0++, itr1++) {
    EXPECT_NEAR(*itr0, *itr1, abs_error);
  }
}

#define XTENSOR_EQ(tensor0, tensor1)                                           \
  ASSERT_EQ(tensor0.shape(), tensor1.shape());                                 \
  ASSERT_EQ(tensor0.storage(), tensor1.storage());

#define XTENSOR_CLOSE(tensor0, tensor1)                                        \
  ASSERT_EQ(tensor0.shape(), tensor1.shape());                                 \
  CLOSE_ALL(tensor0, tensor1);