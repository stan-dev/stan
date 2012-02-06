#include <gtest/gtest.h>
#include "stan/maths/constants.hpp"

TEST(MathsConstants, pi) {
  EXPECT_FLOAT_EQ(4.0 * std::atan(1.0), stan::maths::PI);
}
TEST(MathsConstants, e) {
  EXPECT_FLOAT_EQ(std::exp(1.0), stan::maths::E);
}
TEST(MathsConstants, sqrt2) {
  EXPECT_FLOAT_EQ(std::sqrt(2.0), stan::maths::SQRT_2);
}
TEST(MathsConstants, log2) {
  EXPECT_FLOAT_EQ(std::log(2.0), stan::maths::LOG_2);
}
TEST(MathsConstants, log10) {
  EXPECT_FLOAT_EQ(std::log(10.0), stan::maths::LOG_10);
}

TEST(MathsConstants, infty) {
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), stan::maths::INFTY);
}
TEST(MathsConstants, neg_infty) {
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::maths::NEGATIVE_INFTY);
}
TEST(MathsConstants, nan) {
  EXPECT_TRUE(std::isnan(stan::maths::NOT_A_NUMBER));
}

TEST(MathsConstants, epsilon) {
  EXPECT_TRUE(stan::maths::EPSILON > 0.0);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::epsilon(), stan::maths::EPSILON);
}
TEST(MathsConstants, negative_epsilon) {
  EXPECT_TRUE(stan::maths::NEGATIVE_EPSILON < 0.0);
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::epsilon(), stan::maths::NEGATIVE_EPSILON);
}
