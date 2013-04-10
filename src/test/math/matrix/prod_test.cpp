#include <stan/math/matrix/prod.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,prod_vector_int) {
  using stan::math::prod;
  std::vector<int> v;
  EXPECT_EQ(1,prod(v));
  v.push_back(2);
  EXPECT_EQ(2,prod(v));
  v.push_back(3);
  EXPECT_EQ(6,prod(v));
}
TEST(MathMatrix,prod_vector_double) {
  using stan::math::prod;
  std::vector<double> x;
  EXPECT_FLOAT_EQ(1.0,prod(x));
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(2.0,prod(x));
  x.push_back(3);
  EXPECT_FLOAT_EQ(6.0,prod(x));

  stan::math::vector_d v;
  EXPECT_FLOAT_EQ(1.0,prod(v));
  v = stan::math::vector_d(1);
  v << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(v));
  v = stan::math::vector_d(2);
  v << 2.0, 3.0;
  EXPECT_FLOAT_EQ(6.0,prod(v));

  stan::math::row_vector_d rv;
  EXPECT_FLOAT_EQ(1.0,prod(rv));
  rv = stan::math::row_vector_d(1);
  rv << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(rv));
  rv = stan::math::row_vector_d(2);
  rv << 2.0, 3.0;
  EXPECT_FLOAT_EQ(6.0,prod(rv));

  stan::math::matrix_d m;
  EXPECT_FLOAT_EQ(1.0,prod(m));
  m = stan::math::matrix_d(1,1);
  m << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(m));
  m = stan::math::matrix_d(2,3);
  m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  EXPECT_FLOAT_EQ(720.0,prod(m));
}
