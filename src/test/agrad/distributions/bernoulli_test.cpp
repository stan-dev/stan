#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/bernoulli.hpp>

template <typename T_prob>
void expect_propto(unsigned int n1, T_prob alpha1, 
		   unsigned int n2, T_prob alpha2, 
		   std::string message) {
  expect_eq_diffs(stan::prob::bernoulli_log<false>(n1, alpha1),
		  stan::prob::bernoulli_log<false>(n2, alpha2),
		  stan::prob::bernoulli_log<true>(n1, alpha1),
		  stan::prob::bernoulli_log<true>(n2, alpha2),
		  message);
}

using stan::agrad::var;

TEST(AgradDistributionsBernoulli,Propto) {
  unsigned int n;
  n = 0;
  expect_propto<var>(n, 0.3,
		     n, 0.75,
		     "var: prob, n=0");
  n = 1;
  expect_propto<var>(n, 0.3,
		     n, 0.75,
		     "var: prob, n=1");

}
