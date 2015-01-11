#include <stan/math/functions/ibeta.hpp>
#include <gtest/gtest.h>

const double inf = std::numeric_limits<double>::infinity();

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

TEST(MathFunctions, ibeta_a_boundary) {
  double b = 0.5;
  double x = 0.5;
  
  EXPECT_NO_THROW(stan::math::ibeta(0.0, b, x));
  EXPECT_NO_THROW(stan::math::ibeta(inf, b, x));
  EXPECT_THROW(stan::math::ibeta(-0.01, b, x), std::domain_error);
}

TEST(MathFunctions, ibeta_b_boundary) {
  double a = 0.5;
  double x = 0.5;
  
  EXPECT_NO_THROW(stan::math::ibeta(a, 0.0, x));
  EXPECT_NO_THROW(stan::math::ibeta(inf, 1.0, x));
  EXPECT_THROW(stan::math::ibeta(a, -0.01, x), std::domain_error);
}

TEST(MathFunctions, ibeta_x_boundary) {
  double a = 0.5;
  double b = 0.5;
  
  EXPECT_NO_THROW(stan::math::ibeta(a, b, 0.0));
  EXPECT_NO_THROW(stan::math::ibeta(a, b, 1.0));
  EXPECT_THROW(stan::math::ibeta(a, b, -0.01), std::domain_error);
  EXPECT_THROW(stan::math::ibeta(a, b, 1.01), std::domain_error);
}

TEST(MathFunctions, ibeta_a_b_boundary) {
  double x = 0.5;
  
  EXPECT_THROW(stan::math::ibeta(0.0, 0.0, x), std::domain_error);
}


TEST(MathFunctions, ibeta_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_THROW(stan::math::ibeta(0.5, 0.0, nan), std::domain_error);
  EXPECT_THROW(stan::math::ibeta(0.5, nan, 0.0), std::domain_error);
  EXPECT_THROW(stan::math::ibeta(0.5, nan, nan), std::domain_error);
  EXPECT_THROW(stan::math::ibeta(nan, 0.0, 0.0), std::domain_error);
  EXPECT_THROW(stan::math::ibeta(nan, 0.0, nan), std::domain_error);
  EXPECT_THROW(stan::math::ibeta(nan, nan, 0.0), std::domain_error);
  EXPECT_THROW(stan::math::ibeta(nan, nan, nan), std::domain_error);
}

