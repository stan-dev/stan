#include <gtest/gtest.h>

#include <stan/prob/distributions/multivariate/continuous/multi_normal_cholesky.hpp>

#include <vector>
#include <test/agrad/distributions/multivariate/continuous/test_gradients.hpp>
#include <test/agrad/distributions/expect_eq_diffs.hpp>


using Eigen::Dynamic;
using Eigen::Matrix;

using stan::agrad::var;
using stan::agrad::to_var;



struct multi_normal_cholesky_fun {
  const int K_;

  multi_normal_cholesky_fun(int K) : K_(K) { }

  template <typename T>
  T operator()(const std::vector<T>& x) const {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using stan::agrad::var;
    Matrix<T,Dynamic,1> y(K_);
    Matrix<T,Dynamic,1> mu(K_);
    Matrix<T,Dynamic,Dynamic> L(K_,K_);
    int pos = 0;
    for (int i = 0; i < K_; ++i)
      y(i) = x[pos++];
    for (int i = 0; i < K_; ++i)
      mu(i) = x[pos++];
    // fill lower triangular by row
    for (int i = 0; i < K_; ++i) {
      for (int j = 0; j <= i; ++j)
        L(i,j) = x[pos++];
      for (int j = i + 1; j < K_; ++j)
        L(i,j) = 0;
    }
    return stan::prob::multi_normal_cholesky_log<false>(y,mu,L);
    // can't test propto=true because finite diffs are
    // all 0 by design for double inputs
  }
};

TEST(MultiNormalPrec, TestGradFunctional) {
  std::vector<double> x(3 + 3 + 3 * 3);
  // y
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = -3.0;
  // mu
  x[3] = 0.0;
  x[4] = -2.0;
  x[5] = -3.0;
  // L
  x[6] = 1;
  x[7] = -1;
  x[8] = 2;
  x[9] = -3;
  x[10] = 7;
  x[11] = 8;

  test_grad(multi_normal_cholesky_fun(3), x);

  std::vector<double> u(3);
  u[0] = 1.9;
  u[1] = -2.7;
  u[2] = 0.48;
  
  test_grad(multi_normal_cholesky_fun(1), u);
}
