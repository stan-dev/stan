#include "stan/math/functions/Phi.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, Phi) {
  EXPECT_EQ(0.5 + 0.5 * boost::math::erf(0.0), stan::math::Phi(0.0));
  EXPECT_FLOAT_EQ(0.5 + 0.5 * boost::math::erf(0.9/std::sqrt(2.0)), stan::math::Phi(0.9));
  EXPECT_EQ(0.5 + 0.5 * boost::math::erf(-5.0/std::sqrt(2.0)), stan::math::Phi(-5.0));
}
