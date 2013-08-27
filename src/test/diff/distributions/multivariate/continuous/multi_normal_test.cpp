#include <gtest/gtest.h>
#include <test/diff/distributions/expect_eq_diffs.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

template <typename T_y, typename T_loc, typename T_scale>
void expect_propto(T_y y1, T_loc mu1, T_scale sigma1,
                   T_y y2, T_loc mu2, T_scale sigma2,
                   std::string message = "") {
  expect_eq_diffs(stan::prob::multi_normal_log<false>(y1,mu1,sigma1),
                  stan::prob::multi_normal_log<false>(y2,mu2,sigma2),
                  stan::prob::multi_normal_log<true>(y1,mu1,sigma1),
                  stan::prob::multi_normal_log<true>(y2,mu2,sigma2),
                  message);
  expect_eq_diffs(stan::prob::multi_normal_prec_log<false>(y1,mu1,sigma1),
                  stan::prob::multi_normal_prec_log<false>(y2,mu2,sigma2),
                  stan::prob::multi_normal_prec_log<true>(y1,mu1,sigma1),
                  stan::prob::multi_normal_prec_log<true>(y2,mu2,sigma2),
                  message);
}

using stan::diff::var;
using stan::diff::to_var;

class DiffDistributionsMultiNormal : public ::testing::Test {
protected:
  virtual void SetUp() {
    y.resize(3,1);
    y << 2.0, -2.0, 11.0;
    y2.resize(3,1);
    y2 << 15.0, 1.0, -5.0;

    mu.resize(3,1);
    mu << 1.0, -1.0, 3.0;
    mu2.resize(3,1);
    mu2 << 6.0, 2.0, -6.0;

    Sigma.resize(3,3);
    Sigma << 9.0, -3.0, 0.0,
      -3.0,  4.0, 0.0,
      0.0, 0.0, 5.0;
    Sigma2.resize(3,3);
    Sigma2 << 3.0, 1.0, 0.0,
      1.0,  5.0, -2.0,
      0.0, -2.0, 9.0;
  }
  
  Matrix<double,Dynamic,1> y;
  Matrix<double,Dynamic,1> y2;
  Matrix<double,Dynamic,1> mu;
  Matrix<double,Dynamic,1> mu2;
  Matrix<double,Dynamic,Dynamic> Sigma;
  Matrix<double,Dynamic,Dynamic> Sigma2;
};

TEST_F(DiffDistributionsMultiNormal,Propto) {
  expect_propto(to_var(y),to_var(mu),to_var(Sigma),
                to_var(y2),to_var(mu2),to_var(Sigma2),
                "All vars: y, mu, sigma");
}
TEST_F(DiffDistributionsMultiNormal,ProptoY) {
  expect_propto(to_var(y),mu,Sigma,
                to_var(y2),mu,Sigma,
                "var: y");

}
TEST_F(DiffDistributionsMultiNormal,ProptoYMu) {
  expect_propto(to_var(y),to_var(mu),Sigma,
                to_var(y2),to_var(mu2),Sigma,
                "var: y and mu");
}
TEST_F(DiffDistributionsMultiNormal,ProptoYSigma) {
  expect_propto(to_var(y),mu,to_var(Sigma),
                to_var(y2),mu,to_var(Sigma2),
                "var: y and sigma");
}
TEST_F(DiffDistributionsMultiNormal,ProptoMu) {
  expect_propto(y,to_var(mu),Sigma,
                y,to_var(mu2),Sigma,
                "var: mu");
}
TEST_F(DiffDistributionsMultiNormal,ProptoMuSigma) {
  expect_propto(y,to_var(mu),to_var(Sigma),
                y,to_var(mu2),to_var(Sigma2),
                "var: mu and sigma");
}
TEST_F(DiffDistributionsMultiNormal,ProptoSigma) {
  expect_propto(y,mu,to_var(Sigma),
                y,mu,to_var(Sigma2),
                "var: sigma");
}
class DiffDistributionsMultiNormalMultiRow : public ::testing::Test {
protected:
  virtual void SetUp() {
    y.resize(1,3);
    y << 2.0, -2.0, 11.0;
    y2.resize(1,3);
    y2 << 15.0, 1.0, -5.0;

    mu.resize(3,1);
    mu << 1.0, -1.0, 3.0;
    mu2.resize(3,1);
    mu2 << 6.0, 2.0, -6.0;

    Sigma.resize(3,3);
    Sigma << 9.0, -3.0, 0.0,
      -3.0,  4.0, 0.0,
      0.0, 0.0, 5.0;
    Sigma2.resize(3,3);
    Sigma2 << 3.0, 1.0, 0.0,
      1.0,  5.0, -2.0,
      0.0, -2.0, 9.0;
  }
  
  Matrix<double,Dynamic,Dynamic> y;
  Matrix<double,Dynamic,Dynamic> y2;
  Matrix<double,Dynamic,1> mu;
  Matrix<double,Dynamic,1> mu2;
  Matrix<double,Dynamic,Dynamic> Sigma;
  Matrix<double,Dynamic,Dynamic> Sigma2;
};

TEST_F(DiffDistributionsMultiNormalMultiRow,Propto) {
  expect_propto(to_var(y),to_var(mu),to_var(Sigma),
                to_var(y2),to_var(mu2),to_var(Sigma2),
                "All vars: y, mu, sigma");
}
TEST_F(DiffDistributionsMultiNormalMultiRow,ProptoY) {
  expect_propto(to_var(y),mu,Sigma,
                to_var(y2),mu,Sigma,
                "var: y");

}
TEST_F(DiffDistributionsMultiNormalMultiRow,ProptoYMu) {
  expect_propto(to_var(y),to_var(mu),Sigma,
                to_var(y2),to_var(mu2),Sigma,
                "var: y and mu");
}
TEST_F(DiffDistributionsMultiNormalMultiRow,ProptoYSigma) {
  expect_propto(to_var(y),mu,to_var(Sigma),
                to_var(y2),mu,to_var(Sigma2),
                "var: y and sigma");
}
TEST_F(DiffDistributionsMultiNormalMultiRow,ProptoMu) {
  expect_propto(y,to_var(mu),Sigma,
                y,to_var(mu2),Sigma,
                "var: mu");
}
TEST_F(DiffDistributionsMultiNormalMultiRow,ProptoMuSigma) {
  expect_propto(y,to_var(mu),to_var(Sigma),
                y,to_var(mu2),to_var(Sigma2),
                "var: mu and sigma");
}
TEST_F(DiffDistributionsMultiNormalMultiRow,ProptoSigma) {
  expect_propto(y,mu,to_var(Sigma),
                y,mu,to_var(Sigma2),
                "var: sigma");
}
