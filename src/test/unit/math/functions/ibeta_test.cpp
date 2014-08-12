#include <stan/math/functions/ibeta.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, ibeta) {
  using stan::math::ibeta;
  
  EXPECT_FLOAT_EQ(0.0, ibeta(0.5, 0.5, 0.0))  << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.333333333, ibeta(0.5, 0.5, 0.25))  << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.5, ibeta(0.5, 0.5, 0.5))  << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.666666667, ibeta(0.5, 0.5, 0.75))  << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(1.0, ibeta(0.5, 0.5, 1.0))  << "reasonable values for a, b, x";

  EXPECT_FLOAT_EQ(0.0, ibeta(0.1, 1.5, 0.0))  << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.9117332, ibeta(0.1, 1.5, 0.25))  << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.9645342, ibeta(0.1, 1.5, 0.5))  << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.9897264, ibeta(0.1, 1.5, 0.75))  << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(1.0, ibeta(0.1, 1.5, 1.0))  << "reasonable values for a, b, x";
}

TEST(MathFunctions, ibeta_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::ibeta(0.5, 0.5, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::ibeta(0.5, nan, 0.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::ibeta(0.5, nan, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::ibeta(nan, 0.5, 0.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::ibeta(nan, 0.5, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::ibeta(nan, nan, 0.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::ibeta(nan, nan, nan));
}
