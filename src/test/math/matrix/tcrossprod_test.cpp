#include <stan/math/matrix/tcrossprod.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

void test_tcrossprod(const stan::math::matrix_d& x) {
  using stan::math::tcrossprod;
  stan::math::matrix_d y = tcrossprod(x);
  stan::math::matrix_d xxt = x * x.transpose();
  EXPECT_EQ(y.rows(),xxt.rows());
  EXPECT_EQ(y.cols(),xxt.cols());
  for (int m = 0; m < y.rows(); ++m)
    for (int n = 0; n < y.cols(); ++n)
      EXPECT_FLOAT_EQ(xxt(m,n),y(m,n));
}
TEST(MathMatrix, tcrossprod) {
  stan::math::matrix_d x;
  test_tcrossprod(x);

  x = stan::math::matrix_d(1,1);
  x << 3.0;
  test_tcrossprod(x);

  x = stan::math::matrix_d(2,2);
  x <<
    1.0, 0.0,
    2.0, 3.0;
  test_tcrossprod(x);

  x = stan::math::matrix_d(3,3);
  x <<
    1.0, 0.0, 0.0,
    2.0, 3.0, 0.0,
    4.0, 5.0, 6.0;
  test_tcrossprod(x);
}
