#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <stan/error_handling/matrix/check_symmetric.hpp>

using stan::math::matrix_d;

matrix_d generate_large_L_tri_mat(){
  matrix_d x;
  srand(1);

  x = Eigen::MatrixXd::Random(100,100);
  x *= 1e10;

  return x;
}

matrix_d old_multiply_lower_tri_self_transpose(const matrix_d& x){
  using stan::error_handling::check_symmetric;

  int K = x.rows();
  static const char* function = "old_multiply_lower_tri_self_transpose(%1%)";

  if (K == 0)
    return matrix_d(0,0);
  if (K == 1) {
    matrix_d result(1,1);
    result(0,0) = x(0,0) * x(0,0);
    return result;
  }

  matrix_d Lt = x.transpose().triangularView<Eigen::Upper>();
  matrix_d LLt = x.triangularView<Eigen::Lower>() * Lt; 

  check_symmetric(function, LLt,
                  "LLt", static_cast<double*>(0));
  return LLt;
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
  using stan::error_handling::check_symmetric;
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
                                  multiply_lower_tri_self_transpose(x),
                                  "Symmetric matrix",static_cast<double*>(0)));
  EXPECT_THROW(old_multiply_lower_tri_self_transpose(x),std::domain_error);
}
