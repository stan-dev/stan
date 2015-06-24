#include <stan/math/rev/scal/fun/is_uninitialized.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,undefined) {
  stan::math::var a;
  EXPECT_TRUE(a.is_uninitialized());
  a = 5;
  EXPECT_FALSE(a.is_uninitialized());
}

TEST(AgradRev, is_uninitialized_nan) {
  stan::math::var nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_FALSE(stan::math::is_uninitialized(nan));
}
