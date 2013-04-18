#include <stan/math/matrix/variance.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,variance) {
  using stan::math::variance;
  std::vector<double> x;
  EXPECT_THROW(variance(x),std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(0.0,variance(x));
  x.push_back(2.0);
  EXPECT_NEAR(0.5,variance(x),0.000001);
  x.push_back(3.0);
  EXPECT_FLOAT_EQ(1.0,variance(x));

  stan::math::vector_d v;
  EXPECT_THROW(variance(v),std::domain_error);
  v = stan::math::vector_d(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0,variance(v));
  v = stan::math::vector_d(2);
  v << 1.0, 2.0;
  EXPECT_NEAR(0.5,variance(v),0.000001);
  v = stan::math::vector_d(3);
  v << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(1.0,variance(v));

  stan::math::row_vector_d rv;
  EXPECT_THROW(variance(rv),std::domain_error);
  rv = stan::math::row_vector_d(1);
  rv << 1.0;
  EXPECT_FLOAT_EQ(0.0,variance(rv));
  rv = stan::math::row_vector_d(2);
  rv << 1.0, 2.0;
  EXPECT_NEAR(0.5,variance(rv),0.000001);
  rv = stan::math::row_vector_d(3);
  rv << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(1.0,variance(rv));


  stan::math::matrix_d m;
  EXPECT_THROW(variance(m),std::domain_error);
  m = stan::math::matrix_d(1,1);
  m << 1.0;
  EXPECT_FLOAT_EQ(0.0,variance(m));
  m = stan::math::matrix_d(2,3);
  m << 1.0, 2.0, 4.0, 9.0, 16.0, 25.0;
  EXPECT_NEAR(88.3,variance(m),0.000001);
}
