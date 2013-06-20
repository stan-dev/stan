#include <stan/math/matrix/to_vector.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, to_vector) {
  using stan::math::to_vector;
  using stan::math::size_type;

  Eigen::MatrixXd a(3,3);
  Eigen::VectorXd b(9);
  for(size_type i = 0; i < 3; i++) {
    for(size_type j = 0; j < 3; j++)
      a(j,i) = j + i;
  }

  b = to_vector(a);

  EXPECT_FLOAT_EQ(0,b(0));
  EXPECT_FLOAT_EQ(1,b(1));
  EXPECT_FLOAT_EQ(2,b(2));
  EXPECT_FLOAT_EQ(1,b(3));
  EXPECT_FLOAT_EQ(2,b(4));
  EXPECT_FLOAT_EQ(3,b(5));
  EXPECT_FLOAT_EQ(2,b(6));
  EXPECT_FLOAT_EQ(3,b(7));
  EXPECT_FLOAT_EQ(4,b(8));

  stan::math::row_vector_d c(4);
  c(0) = 0, c(1) = 1, c(2) = 2, c(3) = 3;
  stan::math::vector_d d = to_vector(c);

  EXPECT_FLOAT_EQ(0, d(0));
  EXPECT_FLOAT_EQ(1, d(1));
  EXPECT_FLOAT_EQ(2, d(2));
  EXPECT_FLOAT_EQ(3, d(3));

  stan::math::vector_d e(2);
  e(0) = 0, e(1) = 1;
  stan::math::vector_d f = to_vector(e);

  EXPECT_FLOAT_EQ(0, f(0));
  EXPECT_FLOAT_EQ(1, f(1));
}
