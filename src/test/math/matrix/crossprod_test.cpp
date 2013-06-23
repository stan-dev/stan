#include <stan/math/matrix/crossprod.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

void test_crossprod(const stan::math::matrix_d& x) {
  using stan::math::crossprod;
  stan::math::matrix_d y = crossprod(x);
  stan::math::matrix_d xtx = x.transpose() * x;
  EXPECT_EQ(y.rows(),xtx.rows());
  EXPECT_EQ(y.cols(),xtx.cols());
  for (int m = 0; m < y.rows(); ++m)
    for (int n = 0; n < y.cols(); ++n)
      EXPECT_FLOAT_EQ(xtx(m,n),y(m,n));
}
TEST(MathMatrix,crossprod) {
  stan::math::matrix_d x;
  test_crossprod(x);

  x = stan::math::matrix_d(1,1);
  x << 3.0;
  test_crossprod(x);

  x = stan::math::matrix_d(2,2);
  x <<
    1.0, 0.0,
    2.0, 3.0;
  test_crossprod(x);

  x = stan::math::matrix_d(3,3);
  x <<
    1.0, 0.0, 0.0,
    2.0, 3.0, 0.0,
    4.0, 5.0, 6.0;
  test_crossprod(x);
}
