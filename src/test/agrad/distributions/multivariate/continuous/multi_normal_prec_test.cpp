#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal_prec.hpp>
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



