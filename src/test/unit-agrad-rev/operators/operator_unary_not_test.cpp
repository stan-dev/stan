#include <stan/agrad/rev/operators/operator_unary_not.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,not_a) {
  AVAR a(6.0);
  EXPECT_EQ(0, !a);
  AVAR b(0.0);
  EXPECT_EQ(1, !b);
}

TEST(AgradRev,not_nan) {
  stan::agrad::var nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(!nan);
}
