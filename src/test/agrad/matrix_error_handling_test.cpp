#include <gtest/gtest.h>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/agrad/matrix.hpp>

TEST(AgradMatrixErrorHandling,CheckCovMatrixDefaultPolicy) {
  using stan::agrad::var;
  using Eigen::Dynamic;
  using Eigen::Matrix;

  const char* function = "check_cov_matrix (%1%)";
  var result = 0;
  Matrix<var,Dynamic,Dynamic> Sigma;
  Sigma.resize(1,1);
  Sigma << 1;

  EXPECT_NO_THROW(check_cov_matrix(function, Sigma, &result, default_policy())) << "check_cov_matrix should not throw exception with Sigma: " << Sigma;
}
