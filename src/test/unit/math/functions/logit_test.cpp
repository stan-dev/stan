#include "stan/math/functions/logit.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, logit) {
  using stan::math::logit;
  EXPECT_FLOAT_EQ(0.0, logit(0.5));
  EXPECT_FLOAT_EQ(5.0, logit(1.0/(1.0 + exp(-5.0))));
}
