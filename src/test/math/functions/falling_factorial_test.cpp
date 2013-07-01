#include "stan/math/functions/falling_factorial.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, falling_factorial) {
  using stan::math::falling_factorial;
  
  EXPECT_FLOAT_EQ(4, falling_factorial(4.0,3));
  EXPECT_FLOAT_EQ(0.25, falling_factorial(3.0,4));
  EXPECT_THROW(falling_factorial(-1, 4), std::domain_error);
}
