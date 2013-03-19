#include "stan/math/functions/log1p.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, log1p) {
  double x;

  x = 0;
  EXPECT_FLOAT_EQ(0.0, stan::math::log1p(x));
  x = 0.0000001;
  EXPECT_FLOAT_EQ(0.0000001, stan::math::log1p(x));
  x = 0.001;
  EXPECT_FLOAT_EQ(0.0009995003, stan::math::log1p(x));
  x = 0.1;
  EXPECT_FLOAT_EQ(0.09531018, stan::math::log1p(x));
  x = 1;
  EXPECT_FLOAT_EQ(0.6931472, stan::math::log1p(x));
  x = 10;
  EXPECT_FLOAT_EQ(2.397895, stan::math::log1p(x));

  x = -0.0000001;
  EXPECT_FLOAT_EQ(-0.0000001, stan::math::log1p(x));
  x = -0.001;
  EXPECT_FLOAT_EQ(-0.0010005, stan::math::log1p(x));
  x = -0.1;
  EXPECT_FLOAT_EQ(-0.1053605, stan::math::log1p(x));
  x = -0.999;
  EXPECT_FLOAT_EQ(-6.907755, stan::math::log1p(x));
  x = -1;
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::math::log1p(x));
}

TEST(MathFunctions, log1p_exception) {
  double x;

  x = -2;
  EXPECT_THROW(stan::math::log1p(x), std::domain_error);
}

