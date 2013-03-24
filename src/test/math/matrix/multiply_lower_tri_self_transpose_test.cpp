#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

void test_multiply_lower_tri_self_transpose(const stan::math::matrix_d& x) {
  using stan::math::multiply_lower_tri_self_transpose;
  stan::math::matrix_d y = multiply_lower_tri_self_transpose(x);
  stan::math::matrix_d xp = x;
  for (int m = 0; m < xp.rows(); ++m)
    for (int n = m+1; n < xp.cols(); ++n)
      xp(m,n) = 0;

  stan::math::matrix_d xxt = xp * xp.transpose();
  EXPECT_EQ(y.rows(),xxt.rows());
  EXPECT_EQ(y.cols(),xxt.cols());
  for (int m = 0; m < y.rows(); ++m)
    for (int n = 0; n < y.cols(); ++n)
      EXPECT_FLOAT_EQ(xxt(m,n),y(m,n));
}
TEST(MathMatrix, multiply_lower_tri_self_transpose) {
  stan::math::matrix_d x;
  test_multiply_lower_tri_self_transpose(x);

  x = stan::math::matrix_d(1,1);
  x << 3.0;
  test_multiply_lower_tri_self_transpose(x);

  x = stan::math::matrix_d(2,2);
  x << 
    1.0, 0.0,
    2.0, 3.0;
  test_multiply_lower_tri_self_transpose(x);

  x = stan::math::matrix_d(3,3);
  x << 
    1.0, 0.0, 0.0,
    2.0, 3.0, 0.0,
    4.0, 5.0, 6.0;
  test_multiply_lower_tri_self_transpose(x);

  x = stan::math::matrix_d(3,3);
  x << 
    1.0, 0.0, 100000.0,
    2.0, 3.0, 0.0,
    4.0, 5.0, 6.0;
  test_multiply_lower_tri_self_transpose(x);

  x = stan::math::matrix_d(3,2);
  x << 
    1.0, 0.0,
    2.0, 3.0,
    4.0, 5.0;
  test_multiply_lower_tri_self_transpose(x);

  x = stan::math::matrix_d(2,3);
  x << 
    1.0, 0.0, 0.0,
    2.0, 3.0, 0.0;
  test_multiply_lower_tri_self_transpose(x);
}
