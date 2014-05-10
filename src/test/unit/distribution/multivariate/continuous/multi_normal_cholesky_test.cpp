#include <gtest/gtest.h>

#include <stan/prob/distributions/multivariate/continuous/multi_normal_cholesky.hpp>

#include <vector>
#include <test/unit/distribution/multivariate/continuous/test_gradients.hpp>
#include <test/unit/distribution/expect_eq_diffs.hpp>


using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;

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

TEST(MultiNormalCholesky, TestGradFunctional) {
  std::vector<double> x(3 + 3 + 3 * 2);
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

struct vectorized_multi_normal_cholesky_fun {
  const int K_; //size of each vector and order of square matrix sigma
  const int L_; //size of the array of eigen vectors

  vectorized_multi_normal_cholesky_fun(int K, int L) : K_(K), L_(L) { }

  template <typename T>
  T operator()(const std::vector<T>& x) const {
    vector<Matrix<T,Dynamic,1> > y(L_, Matrix<T,Dynamic,1> (K_));
    vector<Matrix<T,Dynamic,1> > mu(L_, Matrix<T,Dynamic,1> (K_));
    Matrix<T,Dynamic,Dynamic> L(K_,K_);
    int pos = 0;
    for (int i = 0; i < L_; ++i) 
      for (int j = 0; j < K_; ++j)
        y[i](j) = x[pos++];
        
    for (int i = 0; i < L_; ++i)         
      for (int j = 0; j < K_; ++j)
        mu[i](j) = x[pos++];
    
    for (int i = 0; i < K_; ++i) {
      for (int j = 0; j <= i; ++j)
        L(i,j) = x[pos++];
      for (int j = i + 1; j < K_; ++j)
        L(i,j) = 0;
    }
    return stan::prob::multi_normal_cholesky_log<false>(y,mu,L);
  }
};

TEST(MultiNormalCholesky, TestGradFunctionalVectorized) {
  
  {
  vector<double> x(3 + 3 + 3 * 3);
  // y
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = -3.0;
  // mu
  x[3] = 0.0;
  x[4] = -2.0;
  x[5] = -3.0;
  // Sigma
  x[6] = 1;
  x[7] = -1;
  x[8] = 10;
  x[9] = -2;
  x[10] = 20;
  x[11] = 56;

  test_grad(vectorized_multi_normal_cholesky_fun(3, 1), x);
  }
  
  vector<double> x(18);
  // y[1]
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = -3.0;
  // y[2]
  x[3] = 0.0;
  x[4] = -2.0;
  x[5] = -3.0;
  
  // mu[1]
  x[6] = 0.0;
  x[7] = 1.0;
  x[8] = 3.0;
  // mu[2]
  x[9] = 0.0;
  x[10] = -1.0;
  x[11] = -2.0;
  
  // Sigma
  x[12] = 1;
  x[13] = -1;
  x[14] = 1;
  x[15] = -2;
  x[16] = 1;
  x[17] = 6;

  test_grad(vectorized_multi_normal_cholesky_fun(3, 2), x);
  
  vector<double> u(3);
  u[0] = 1.9;
  u[1] = -2.7;
  u[2] = 0.48;
  
  test_grad(vectorized_multi_normal_cholesky_fun(1, 1), u);
}
