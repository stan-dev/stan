#include <gtest/gtest.h>

#include <stan/prob/distributions/multivariate/continuous/multi_normal_prec.hpp>

#include <vector>
#include <test/agrad/distributions/multivariate/continuous/test_gradients.hpp>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <test/agrad/distributions/multivariate/continuous/agrad_distributions_multi_normal_multi_row.hpp>
#include <test/agrad/distributions/multivariate/continuous/agrad_distributions_multi_normal.hpp>


using Eigen::Dynamic;
using Eigen::Matrix;

using stan::agrad::var;
using stan::agrad::to_var;

template <typename T_y, typename T_loc, typename T_scale>
void expect_propto(T_y y1, T_loc mu1, T_scale sigma1,
                   T_y y2, T_loc mu2, T_scale sigma2,
                   std::string message = "") {
  expect_eq_diffs(stan::prob::multi_normal_prec_log<false>(y1,mu1,sigma1),
                  stan::prob::multi_normal_prec_log<false>(y2,mu2,sigma2),
                  stan::prob::multi_normal_prec_log<true>(y1,mu1,sigma1),
                  stan::prob::multi_normal_prec_log<true>(y2,mu2,sigma2),
                  message);
}


TEST_F(agrad_distributions_multi_normal,Propto) {
  expect_propto(to_var(y),to_var(mu),to_var(Sigma),
                to_var(y2),to_var(mu2),to_var(Sigma2),
                "All vars: y, mu, sigma");
}
TEST_F(agrad_distributions_multi_normal,ProptoY) {
  expect_propto(to_var(y),mu,Sigma,
                to_var(y2),mu,Sigma,
                "var: y");

}
TEST_F(agrad_distributions_multi_normal,ProptoYMu) {
  expect_propto(to_var(y),to_var(mu),Sigma,
                to_var(y2),to_var(mu2),Sigma,
                "var: y and mu");
}
TEST_F(agrad_distributions_multi_normal,ProptoYSigma) {
  expect_propto(to_var(y),mu,to_var(Sigma),
                to_var(y2),mu,to_var(Sigma2),
                "var: y and sigma");
}
TEST_F(agrad_distributions_multi_normal,ProptoMu) {
  expect_propto(y,to_var(mu),Sigma,
                y,to_var(mu2),Sigma,
                "var: mu");
}
TEST_F(agrad_distributions_multi_normal,ProptoMuSigma) {
  expect_propto(y,to_var(mu),to_var(Sigma),
                y,to_var(mu2),to_var(Sigma2),
                "var: mu and sigma");
}
TEST_F(agrad_distributions_multi_normal,ProptoSigma) {
  expect_propto(y,mu,to_var(Sigma),
                y,mu,to_var(Sigma2),
                "var: sigma");
}

TEST_F(agrad_distributions_multi_normal_multi_row,Propto) {
  expect_propto(to_var(y),to_var(mu),to_var(Sigma),
                to_var(y2),to_var(mu2),to_var(Sigma2),
                "All vars: y, mu, sigma");
}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoY) {
  expect_propto(to_var(y),mu,Sigma,
                to_var(y2),mu,Sigma,
                "var: y");

}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoYMu) {
  expect_propto(to_var(y),to_var(mu),Sigma,
                to_var(y2),to_var(mu2),Sigma,
                "var: y and mu");
}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoYSigma) {
  expect_propto(to_var(y),mu,to_var(Sigma),
                to_var(y2),mu,to_var(Sigma2),
                "var: y and sigma");
}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoMu) {
  expect_propto(y,to_var(mu),Sigma,
                y,to_var(mu2),Sigma,
                "var: mu");
}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoMuSigma) {
  expect_propto(y,to_var(mu),to_var(Sigma),
                y,to_var(mu2),to_var(Sigma2),
                "var: mu and sigma");
}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoSigma) {
  expect_propto(y,mu,to_var(Sigma),
                y,mu,to_var(Sigma2),
                "var: sigma");
}



struct multi_normal_prec_fun {
  const int K_;

  multi_normal_prec_fun(int K) : K_(K) { }

  template <typename T>
  T operator()(const std::vector<T>& x) const {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using stan::agrad::var;
    Matrix<T,Dynamic,1> y(K_);
    Matrix<T,Dynamic,1> mu(K_);
    Matrix<T,Dynamic,Dynamic> Sigma(K_,K_);
    int pos = 0;
    for (int i = 0; i < K_; ++i)
      y(i) = x[pos++];
    for (int i = 0; i < K_; ++i)
      mu(i) = x[pos++];
    for (int j = 0; j < K_; ++j) {
      for (int i = 0; i <= j; ++i) {
        Sigma(i,j) = x[pos++];
        Sigma(j,i) = Sigma(i,j);
      }
    }
    return stan::prob::multi_normal_prec_log<false>(y,mu,Sigma);
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
  // Sigma
  x[6] = 1;
  x[7] = -1;
  x[8] = 10;
  x[9] = -2;
  x[10] = 20;
  x[11] = 56;

  test_grad(multi_normal_prec_fun(3), x);

  std::vector<double> u(3);
  u[0] = 1.9;
  u[1] = -2.7;
  u[2] = 0.48;
  
  test_grad(multi_normal_prec_fun(1), u);

}
