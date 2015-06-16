#include <stan/math/prim/mat/fun/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <stan/math/prim/mat/err/check_symmetric.hpp>

using stan::math::matrix_d;

matrix_d generate_large_L_tri_mat(){
  matrix_d x;
  double vals[10000];

  vals[0] = 0.1;
  for (int i = 1; i < 10000; ++i)
    vals[i] = vals[i- 1] + 0.1123456;
  
  x = Eigen::Map< Eigen::Matrix<double,100,100> >(vals);
  x *= 1e10;

  return x;
}

void test_multiply_lower_tri_self_transpose(const matrix_d& x) {
  using stan::math::multiply_lower_tri_self_transpose;
  matrix_d y = multiply_lower_tri_self_transpose(x);
  matrix_d xp = x;
  for (int m = 0; m < xp.rows(); ++m)
    for (int n = m+1; n < xp.cols(); ++n)
      xp(m,n) = 0;

  matrix_d xxt = xp * xp.transpose();
  EXPECT_EQ(y.rows(),xxt.rows());
  EXPECT_EQ(y.cols(),xxt.cols());
  for (int m = 0; m < y.rows(); ++m)
    for (int n = 0; n < y.cols(); ++n)
      EXPECT_FLOAT_EQ(xxt(m,n),y(m,n));
}

TEST(MathMatrix, multiply_lower_tri_self_transpose) {
  using stan::math::check_symmetric;
  using stan::math::multiply_lower_tri_self_transpose;
  static const char* function = "stan::math::multiply_lower_tri_self_transpose(%1%)";
  matrix_d x;
  test_multiply_lower_tri_self_transpose(x);

  x = matrix_d(1,1);
  x << 3.0;
  test_multiply_lower_tri_self_transpose(x);

  x = matrix_d(2,2);
  x << 
    1.0, 0.0,
    2.0, 3.0;
  test_multiply_lower_tri_self_transpose(x);

  x = matrix_d(3,3);
  x << 
    1.0, 0.0, 0.0,
    2.0, 3.0, 0.0,
    4.0, 5.0, 6.0;
  test_multiply_lower_tri_self_transpose(x);

  x = matrix_d(3,3);
  x << 
    1.0, 0.0, 100000.0,
    2.0, 3.0, 0.0,
    4.0, 5.0, 6.0;
  test_multiply_lower_tri_self_transpose(x);

  x = matrix_d(3,2);
  x << 
    1.0, 0.0,
    2.0, 3.0,
    4.0, 5.0;
  test_multiply_lower_tri_self_transpose(x);

  x = matrix_d(2,3);
  x << 
    1.0, 0.0, 0.0,
    2.0, 3.0, 0.0;
  test_multiply_lower_tri_self_transpose(x);

  x = generate_large_L_tri_mat();
  EXPECT_NO_THROW(check_symmetric(function, 
                                  "Symmetric matrix", 
                                  multiply_lower_tri_self_transpose(x)));
}
