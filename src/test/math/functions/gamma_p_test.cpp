#include "stan/math/functions/gamma_p.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, gamma_p) {
  using stan::math::gamma_p;
  
  EXPECT_FLOAT_EQ(0.63212055, gamma_p(1.0,1.0));
  EXPECT_FLOAT_EQ(0.82755178, gamma_p(0.1,0.1));
  EXPECT_FLOAT_EQ(0.76189667, gamma_p(3.0,4.0));
  EXPECT_FLOAT_EQ(0.35276812, gamma_p(4.0,3.0));
  EXPECT_THROW(gamma_p(-4.0,3.0), std::domain_error);
  EXPECT_THROW(gamma_p(4.0,-3.0), std::domain_error);
}
