#include "stan/math/functions/gamma_q.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, gamma_q) {
  using stan::math::gamma_q;
  
  EXPECT_FLOAT_EQ(1.0 - 0.63212055, gamma_q(1.0,1.0));
  EXPECT_FLOAT_EQ(1.0 - 0.82755178, gamma_q(0.1,0.1));
  EXPECT_FLOAT_EQ(1.0 - 0.76189667, gamma_q(3.0,4.0));
  EXPECT_FLOAT_EQ(1.0 - 0.35276812, gamma_q(4.0,3.0));
  EXPECT_THROW(gamma_q(-4.0,3.0), std::domain_error);
  EXPECT_THROW(gamma_q(4.0,-3.0), std::domain_error);
}
