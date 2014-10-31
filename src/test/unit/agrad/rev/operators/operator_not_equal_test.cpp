#include <stan/agrad/rev/operators/operator_not_equal.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,a_neq_y) {
  AVAR a = 2.0;
  double y = 3.0;
  EXPECT_TRUE(a != y);
  EXPECT_TRUE(y != a);
  double z = 2.0;
  EXPECT_FALSE(a != z);
  EXPECT_FALSE(z != a);
}

TEST(AgradRev, logical_neq_nan) {
  stan::agrad::var nan = std::numeric_limits<double>::quiet_NaN();
  stan::agrad::var a = 1.0;
  stan::agrad::var b = 2.0;
  double nan_dbl = std::numeric_limits<double>::quiet_NaN();

  EXPECT_TRUE(1.0 != nan);
  EXPECT_TRUE(nan != 2.0);
  EXPECT_TRUE(nan != nan);
  EXPECT_TRUE(a != nan);
  EXPECT_TRUE(nan != b);
  EXPECT_TRUE(a != nan_dbl);
  EXPECT_TRUE(nan_dbl != b);
}
