#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/univariate/discrete/neg_binomial.hpp>


template <typename T_shape, typename T_inv_scale>
void expect_propto(int n1, T_shape alpha1, T_inv_scale beta1,
                   int n2, T_shape alpha2, T_inv_scale beta2,
                   std::string message) {
  expect_eq_diffs(stan::prob::neg_binomial_log<false>(n1,alpha1,beta1),
                  stan::prob::neg_binomial_log<false>(n2,alpha2,beta2),
                  stan::prob::neg_binomial_log<true>(n1,alpha1,beta1),
                  stan::prob::neg_binomial_log<true>(n2,alpha2,beta2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsNegBinomial,Propto) {
  int n = 10;
  expect_propto<var,var>(n,2.5,2.0,
                         n,5.0,3.0,
                         "var: alpha and beta");
}
TEST(AgradDistributionsNegBinomial,ProptoAlpha) {
  int n = 10;
  double beta = 3.0;
  
  expect_propto<var,double>(n, 0.2, beta,
                            n, 6.0, beta,
                            "var: alpha");
}
TEST(AgradDistributionsNegBinomial,ProptoBeta) {
  int n = 10;
  double alpha = 6.0;
  
  expect_propto<double,var>(n, alpha, 1.0,
                            n, alpha, 1.5,
                            "var: beta");
}
