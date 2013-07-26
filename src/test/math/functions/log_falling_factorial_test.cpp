#include "stan/math/functions/log_falling_factorial.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, log_falling_factorial) {
  using stan::math::log_falling_factorial;
  
  EXPECT_FLOAT_EQ(std::log(4.0), log_falling_factorial(4.0,3));
  EXPECT_FLOAT_EQ(std::log(0.25), log_falling_factorial(3.0,4));
  EXPECT_THROW(log_falling_factorial(-1, 4), std::domain_error);
}
