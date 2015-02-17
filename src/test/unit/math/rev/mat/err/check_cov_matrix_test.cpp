#include <gtest/gtest.h>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than.hpp>
#include <stan/math/rev/scal/fun/operator_less_than_or_equal.hpp>
#include <stan/math/prim/mat/err/check_cov_matrix.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>

TEST(AgradRevErrorHandlingMatrix,CheckCovMatrix) {
  using stan::agrad::var;
  using Eigen::Dynamic;
  using Eigen::Matrix;
  
  using stan::math::check_cov_matrix;
  
  const char* function = "check_cov_matrix";
  Matrix<var,Dynamic,Dynamic> Sigma;
  Sigma.resize(1,1);
  Sigma << 1;

  EXPECT_NO_THROW(check_cov_matrix(function, "Sigma", Sigma))
    << "check_cov_matrix should not throw exception with Sigma";
}
