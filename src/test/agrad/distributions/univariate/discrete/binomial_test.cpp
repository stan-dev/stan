#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>

template <typename T_prob>
void expect_propto(int n1, int N1, T_prob theta1, 
                   int n2, int N2, T_prob theta2, 
                   std::string message) {
  expect_eq_diffs(stan::prob::binomial_log<false>(n1, N1, theta1),
                  stan::prob::binomial_log<false>(n2, N2, theta2),
                  stan::prob::binomial_log<true>(n1, N1, theta1),
                  stan::prob::binomial_log<true>(n2, N2, theta2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsBinomial,Propto) {
  expect_propto<var>(2, 10, 0.3,
                     2, 10, 0.75,
                     "all var: n, N, and prob");
}
TEST(AgradDistributionsBinomial,ProptoN1) {
  int N;
  double theta;
  
  N = 12;
  theta = 0.4;
  expect_propto<double>(3, N, theta,
                        3, N, theta,
                        "var: n");
}


