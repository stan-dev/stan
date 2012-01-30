#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>

template <typename T_n, typename T_N, typename T_prob>
void expect_propto(T_n n1, T_N N1, T_prob theta1, 
                   T_n n2, T_N N2, T_prob theta2, 
                   std::string message) {
  expect_eq_diffs(stan::prob::binomial_log<false>(n1, N1, theta1),
                  stan::prob::binomial_log<false>(n2, N2, theta2),
                  stan::prob::binomial_log<true>(n1, N1, theta1),
                  stan::prob::binomial_log<true>(n2, N2, theta2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsBinomial,Propto) {
  expect_propto<var, var, var>(2, 10, 0.3,
                               3, 11, 0.75,
                               "all var: n, N, and prob");
}
TEST(AgradDistributionsBinomial,ProptoN1) {
  double N;
  double theta;
  
  N = 12;
  theta = 0.4;
  expect_propto<var,double,double>(2, N, theta,
                                   3, N, theta,
                                   "var: n");
}
TEST(AgradDistributionsBinomial,ProptoN1N2) {
  double theta;
  
  theta = 0.4;
  expect_propto<var,var,double>(2, 4, theta,
                                3, 15, theta,
                                "var: n and N");
}
TEST(AgradDistributionsBinomial,ProptoN1Theta) {
  double N;
  
  N = 15;
  expect_propto<var,double,var>(2, N, 0.14,
                                3, N, 0.65,
                                "var: n and theta");
  
}
TEST(AgradDistributionsBinomial,ProptoN2) {
  double n;
  double theta;
  
  n = 15;
  theta = 0.345;
  expect_propto<var,double,var>(n, 43, theta,
                                n, 112, theta,
                                "var: N");
  
}
TEST(AgradDistributionsBinomial,ProptoN2Theta) {
  double n;
  
  n = 15;
  expect_propto<var,double,var>(n, 43, 0.14,
                                n, 112, 0.65,
                                "var: n and theta");
}
TEST(AgradDistributionsBinomial,ProptoTheta) {
  double theta;
  
  theta = 0.95;
  expect_propto<var,double,var>(40, 43, theta,
                                2, 112, theta,
                                "var: theta");
}

