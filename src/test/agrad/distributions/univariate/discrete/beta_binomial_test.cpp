#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/univariate/discrete/beta_binomial.hpp>


template <typename T_size>
void expect_propto(int n1, int N1, T_size alpha1, T_size beta1,
                   int n2, int N2, T_size alpha2, T_size beta2,
                   std::string message) {
  expect_eq_diffs(stan::prob::beta_binomial_log<false>(n1,N1,alpha1,beta1),
                  stan::prob::beta_binomial_log<false>(n2,N2,alpha2,beta2),
                  stan::prob::beta_binomial_log<true>(n1,N1,alpha1,beta1),
                  stan::prob::beta_binomial_log<true>(n2,N2,alpha2,beta2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsBetaBinomial,Propto) {
  int n = 10;
  int N = 35;
  expect_propto<var>(n,N,2.5,2.0,
                     n,N,5.0,3.0,
                     "var: alpha and beta");
}
