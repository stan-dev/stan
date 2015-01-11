#include <stan/math/functions/logical_negation.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions,logical_negation) {
  using stan::math::logical_negation;
  EXPECT_TRUE(logical_negation(0));
  EXPECT_TRUE(logical_negation(0.0));
  EXPECT_TRUE(logical_negation(0.0f));

  EXPECT_FALSE(logical_negation(1));
  EXPECT_FALSE(logical_negation(2.0));
  EXPECT_FALSE(logical_negation(2.0f));
}

TEST(MathFunctions, logical_negation_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_FALSE(stan::math::logical_negation(nan));
}
