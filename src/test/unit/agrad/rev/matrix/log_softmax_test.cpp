#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/matrix/log_softmax.hpp>
#include <stan/agrad/rev/matrix/log_softmax.hpp>

TEST(AgradRevMatrix,logSoftmaxLeak) {
  // FIXME: very brittle test depending on unrelated constants of 
  //        block sizes/growth in stan::memory::stack_alloc
  using stan::math::log_softmax;
  using stan::agrad::log_softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_v;
  using stan::agrad::var;

  int NUM = 500;
  int SIZE = 500;
  Matrix<var,Dynamic,1> x(SIZE);
  for (int i = 0; i < NUM; ++i) {
    for (int n = 0; n < x.size(); ++n) {
      x(n) = 0.1 * n;
    }
    Matrix<var,Dynamic,1> theta = log_softmax(x);
  }
  EXPECT_TRUE(stan::agrad::memalloc_.bytes_allocated() > 4000000);
}

TEST(AgradRevMatrix,log_softmax) {
  using stan::math::log_softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_v;

  EXPECT_THROW(log_softmax(vector_v()),std::domain_error);
  
  Matrix<AVAR,Dynamic,1> x(1);
  x << 0.0;
  
  Matrix<AVAR,Dynamic,1> theta = log_softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(log(1.0),theta[0].val());

  Matrix<AVAR,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
  Matrix<AVAR,Dynamic,1> theta2 = log_softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(log(exp(-1)/(exp(-1) + exp(1))), theta2[0].val());
  EXPECT_FLOAT_EQ(log(exp(1)/(exp(-1) + exp(1))), theta2[1].val());

  Matrix<AVAR,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
  Matrix<AVAR,Dynamic,1> theta3 = log_softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(log(exp(-1)/(exp(-1) + exp(1) + exp(10.0))),
                      theta3[0].val());
  EXPECT_FLOAT_EQ(log(exp(1)/(exp(-1) + exp(1) + exp(10.0))), 
                      theta3[1].val());
  EXPECT_FLOAT_EQ(log(exp(10)/(exp(-1) + exp(1) + exp(10.0))), 
                  theta3[2].val());
}

// compute grad using templated definition in math
// to check custom derivatives
std::vector<double>
log_softmax_grad(Eigen::Matrix<double,Eigen::Dynamic,1>& alpha_dbl,
                 int k) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::var;
  Matrix<var,Dynamic,1> alpha(alpha_dbl.size());
  for (int i = 0; i < alpha.size(); ++i)
    alpha(i) = alpha_dbl(i);

  std::vector<var> x(alpha.size());
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = alpha(i);
  
  // var fx_k = stan::math::log_softmax(alpha)[k];
  var fx_k = log(stan::math::softmax(alpha)[k]);
  std::vector<double> grad(alpha.size());
  fx_k.grad(x,grad);
  return grad;
}
TEST(AgradRevLogSoftmax, Grad) {
  using stan::math::log_softmax;
  using stan::agrad::var;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  for (int k = 0; k < 3; ++k) {
    Matrix<AVAR,Dynamic,1> alpha(3);
    alpha << 0.0, 3.0, -1.0;
    Matrix<double,Dynamic,1> alpha_dbl(3);
    alpha_dbl << 0.0, 3.0, -1.0;

    AVEC x(3);
    for (int i = 0; i < 3; ++i)
      x[i] = alpha(i);
    Matrix<AVAR,Dynamic,1> theta = log_softmax(alpha);
    AVAR fx_k = theta(k);
    std::vector<double> grad;
    fx_k.grad(x,grad);

    std::vector<double> grad_expected = log_softmax_grad(alpha_dbl,k);
    EXPECT_EQ(grad_expected.size(), grad.size());
    for (size_t i = 0; i < grad_expected.size(); ++i)
      EXPECT_FLOAT_EQ(grad_expected[i], grad[i]);
  }
}
