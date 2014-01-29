#include <stan/math/matrix/sd.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,sd) {
  using stan::math::sd;
  std::vector<double> x;
  EXPECT_THROW(sd(x),std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(0.0,sd(x));
  x.push_back(2.0);
  EXPECT_NEAR(0.7071068,sd(x),0.000001);
  x.push_back(3.0);
  EXPECT_FLOAT_EQ(1.0,sd(x));

  stan::math::vector_d v;
  EXPECT_THROW(sd(v),std::domain_error);
  v = stan::math::vector_d(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0,sd(v));
  v = stan::math::vector_d(2);
  v << 1.0, 2.0;
  EXPECT_NEAR(0.7071068,sd(v),0.000001);
  v = stan::math::vector_d(3);
  v << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(1.0,sd(v));

  stan::math::row_vector_d rv;
  EXPECT_THROW(sd(rv),std::domain_error);
  rv = stan::math::row_vector_d(1);
  rv << 1.0;
  EXPECT_FLOAT_EQ(0.0,sd(rv));
  rv = stan::math::row_vector_d(2);
  rv << 1.0, 2.0;
  EXPECT_NEAR(0.7071068,sd(rv),0.000001);
  rv = stan::math::row_vector_d(3);
  rv << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(1.0,sd(rv));


  stan::math::matrix_d m;
  EXPECT_THROW(sd(m),std::domain_error);
  m = stan::math::matrix_d(1,1);
  m << 1.0;
  EXPECT_FLOAT_EQ(0.0,sd(m));
  m = stan::math::matrix_d(2,3);
  m << 1.0, 2.0, 4.0, 9.0, 16.0, 25.0;
  EXPECT_NEAR(9.396808,sd(m),0.000001);
}
