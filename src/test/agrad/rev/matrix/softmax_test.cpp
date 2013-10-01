#include <gtest/gtest.h>
#include <test/agrad/util.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/matrix.hpp>
#include <stan/math/matrix/softmax.hpp>

TEST(AgradRevMatrix,softmax) {
  using stan::math::softmax;
  using stan::agrad::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_v;

  EXPECT_THROW(softmax(vector_v()),std::domain_error);
  
  Matrix<AVAR,Dynamic,1> x(1);
  x << 0.0;
  
  Matrix<AVAR,Dynamic,1> theta = softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(1.0,theta[0].val());

  Matrix<AVAR,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
  Matrix<AVAR,Dynamic,1> theta2 = softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1)), theta2[0].val());
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1)), theta2[1].val());

  Matrix<AVAR,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
  Matrix<AVAR,Dynamic,1> theta3 = softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1) + exp(10.0)), theta3[0].val());
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1) + exp(10.0)), theta3[1].val());
  EXPECT_FLOAT_EQ(exp(10)/(exp(-1) + exp(1) + exp(10.0)), theta3[2].val());
}

// compute grad using templated definition in math
// to check custom derivatives
std::vector<double>
softmax_grad(Eigen::Matrix<double,Eigen::Dynamic,1>& alpha_dbl,
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
  
  var fx_k = stan::math::softmax(alpha)[k];
  std::vector<double> grad(alpha.size());
  fx_k.grad(x,grad);
  return grad;
}
TEST(AgradRevSoftmax, Grad) {
  using stan::math::softmax;
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
    Matrix<AVAR,Dynamic,1> theta = softmax(alpha);
    AVAR fx_k = theta(k);
    std::vector<double> grad;
    fx_k.grad(x,grad);

    std::vector<double> grad_expected = softmax_grad(alpha_dbl,k);
    EXPECT_EQ(grad_expected.size(), grad.size());
    for (size_t i = 0; i < grad_expected.size(); ++i)
      EXPECT_FLOAT_EQ(grad_expected[i], grad[i]);
  }
}
