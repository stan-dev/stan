#include "stan/math/functions/rising_factorial.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, rising_factorial) {
  using stan::math::rising_factorial;
  
  EXPECT_FLOAT_EQ(120, rising_factorial(4.0,3));
  EXPECT_FLOAT_EQ(360, rising_factorial(3.0,4));
  EXPECT_THROW(rising_factorial(-1, 4),std::domain_error);
}
