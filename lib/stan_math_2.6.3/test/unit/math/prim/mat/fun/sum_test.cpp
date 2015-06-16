#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,sumVector) {
  using stan::math::sum;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  stan::math::vector_d v;
  EXPECT_FLOAT_EQ(0.0, sum(v));

  v = stan::math::vector_d(1);
  v[0] = 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(v));

  v = stan::math::vector_d(3);
  v[0] = 5.0;
  v[1] = 10.0;
  v[2] = 100.0;
  EXPECT_FLOAT_EQ(115.0,sum(v));
}


TEST(MathMatrix,sumRowVector) {
  using stan::math::sum;
  using Eigen::Matrix;
  using Eigen::Dynamic;

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
}

TEST(MathMatrix,sumMatrix) {
  using stan::math::sum;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  stan::math::matrix_d m;
  EXPECT_FLOAT_EQ(0.0,sum(m));

  m = stan::math::matrix_d(1,1);
  m << 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(m));

  m = stan::math::matrix_d(3,2);
  m << 1, 2, 3, 4, 5, 6;
  EXPECT_FLOAT_EQ(21.0,sum(m));
}
