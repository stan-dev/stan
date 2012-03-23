#include <gtest/gtest.h>
#include "stan/math/constants.hpp"

TEST(MathsConstants, e) {
  EXPECT_FLOAT_EQ(std::exp(1.0), stan::math::E);
}
TEST(MathsConstants, sqrt2) {
  EXPECT_FLOAT_EQ(std::sqrt(2.0), stan::math::SQRT_2);
}
TEST(MathsConstants, log2) {
  EXPECT_FLOAT_EQ(std::log(2.0), stan::math::LOG_2);
}
TEST(MathsConstants, log10) {
  EXPECT_FLOAT_EQ(std::log(10.0), stan::math::LOG_10);
}

TEST(MathsConstants, infty) {
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), stan::math::INFTY);
}
TEST(MathsConstants, neg_infty) {
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::math::NEGATIVE_INFTY);
}
TEST(MathsConstants, nan) {
  EXPECT_TRUE(std::isnan(stan::math::NOT_A_NUMBER));
}

TEST(MathsConstants, epsilon) {
  EXPECT_TRUE(stan::math::EPSILON > 0.0);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::epsilon(), stan::math::EPSILON);
}
TEST(MathsConstants, negative_epsilon) {
  EXPECT_TRUE(stan::math::NEGATIVE_EPSILON < 0.0);
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::epsilon(), stan::math::NEGATIVE_EPSILON);
}
