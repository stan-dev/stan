#include <gtest/gtest.h>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/fwd/matrix.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal_sufficient.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/agrad/autodiff.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;
TEST(ProbDistributionsMultiNormalSufficient,Fit) {
  int sampleSize = 15;
  Matrix<double,Dynamic,1> mu(2,1);
  mu.setZero();
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma.setIdentity();
  Matrix<double,Dynamic,1> sampleMu(2,1);
  sampleMu << .5114703, -.5020808;
  Matrix<double,Dynamic,Dynamic> sampleSigma(2,2);
  sampleSigma << .72174728, -.01822455,
    -.01822455, 1.06501495;
  EXPECT_FLOAT_EQ(-16.35999, stan::prob::multi_normal_sufficient_log(sampleSize, sampleMu, sampleSigma, mu, Sigma));
}

template <typename T>
struct test_model {
  int sampleSize;
  //  typedef double valueType;
  typedef T valueType;
  Matrix<valueType,Dynamic,1> sampleMu;
  Matrix<valueType,Dynamic,Dynamic> sampleSigma;

  test_model() {
    sampleSize = 15;
    sampleMu.resize(2);
    sampleMu << .5114703, -.5020808;
    sampleSigma.resize(2,2);
    sampleSigma << .72174728, -.01822455,
      -.01822455, 1.06501495;
  }

  T operator()(Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
    Matrix<T,Dynamic,1> mu(2,1);
    mu[0] = x[0];
    mu[1] = x[1];
    Matrix<T,Dynamic,Dynamic> Sigma(2,2);
    Sigma(0,0) = x[2];
    Sigma(1,0) = x[3];
    Sigma(0,1) = x[3];
    Sigma(1,1) = x[4];
    return stan::prob::multi_normal_sufficient_log(sampleSize, sampleMu, sampleSigma, mu, Sigma);
  }
};

TEST(ProbDistributionsMultiNormalSufficient,var) {
  using stan::agrad::var;

  Matrix<double, Dynamic, 1> cont_params(5);
  cont_params.setZero();
  cont_params[2] = 1;
  cont_params[4] = 1;
  test_model<var> tm;
  double fx = 0.0;
  Matrix<double, Dynamic, 1> grad(5);
  gradient(tm, cont_params, fx, grad);
  EXPECT_FLOAT_EQ( 7.67205461, grad[0]);
  EXPECT_FLOAT_EQ(-7.53121146, grad[1]);
  EXPECT_FLOAT_EQ( 0.01424497, grad[2]);
  EXPECT_FLOAT_EQ(-4.10713480, grad[3]);
  EXPECT_FLOAT_EQ( 2.34574285, grad[4]);
}
