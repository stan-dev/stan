#include <gtest/gtest.h>
#include <stan/error_handling/matrix.hpp>
#include <stan/agrad/rev.hpp>

TEST(AgradRevErrorHandlingMatrix,CheckCovMatrix) {
  using stan::agrad::var;
  using Eigen::Dynamic;
  using Eigen::Matrix;
  
  using stan::error_handling::check_cov_matrix;
  
  const char* function = "check_cov_matrix(%1%)";
  var result = 0;
  Matrix<var,Dynamic,Dynamic> Sigma;
  Sigma.resize(1,1);
  Sigma << 1;

  EXPECT_NO_THROW(check_cov_matrix(function, Sigma,"Sigma", &result))
    << "check_cov_matrix should not throw exception with Sigma";
}
