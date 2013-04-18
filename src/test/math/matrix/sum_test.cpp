#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,sum_vector_int) {
  std::vector<int> x(3);
  EXPECT_EQ(0,stan::math::sum(x));
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  EXPECT_EQ(6,stan::math::sum(x));
}
TEST(MathMatrix,sum_vector_double) {
  using stan::math::sum;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  std::vector<double> x(3);
  EXPECT_FLOAT_EQ(0.0,sum(x));
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = 3.0;
  EXPECT_FLOAT_EQ(6.0,sum(x));

  stan::math::vector_d v;
  EXPECT_FLOAT_EQ(0.0,sum(v));
  v = stan::math::vector_d(1);
  v[0] = 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(v));
  v = stan::math::vector_d(3);
  v[0] = 5.0;
  v[1] = 10.0;
  v[2] = 100.0;
  EXPECT_FLOAT_EQ(115.0,sum(v));

  stan::math::row_vector_d rv;
  EXPECT_FLOAT_EQ(0.0,sum(rv));
  rv = stan::math::row_vector_d(1);
  rv[0] = 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(rv));
  rv = stan::math::row_vector_d(3);
  rv[0] = 5.0;
  rv[1] = 10.0;
  rv[2] = 100.0;
  EXPECT_FLOAT_EQ(115.0,sum(rv));

  stan::math::matrix_d m;
  EXPECT_FLOAT_EQ(0.0,sum(m));
  m = stan::math::matrix_d(1,1);
  m << 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(m));
  m = stan::math::matrix_d(3,2);
  m << 1, 2, 3, 4, 5, 6;
  EXPECT_FLOAT_EQ(21.0,sum(m));
}
