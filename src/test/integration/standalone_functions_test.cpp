#include <gtest/gtest.h>
#include <test/test-models/good-standalone-functions/basic.hpp>

TEST(standalone_functions, int_only_multiplication) {
  EXPECT_EQ(int_only_multiplication(2, 5), 10);
  EXPECT_NEAR(my_log1p_exp(1), 1.3132, 1E-4);
  Eigen::Matrix<double, -1, 1> a(6), correct(6), res(6);
  a << 1, 2, 3, 4, 5, 6;
  res = my_vector_mul_by_5(a);
  correct << 5, 10, 15, 20, 25, 30;
  for (int i = 0; i < a.size(); i++) {
    EXPECT_EQ(res(i), correct(i));
  }
}
