#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/poisson.hpp>

template <typename T_prob>
void expect_propto(unsigned int n1, T_prob lambda1, 
		   unsigned int n2, T_prob lambda2, 
		   std::string message) {
  expect_eq_diffs(stan::prob::poisson_log<false>(n1, lambda1),
		  stan::prob::poisson_log<false>(n2, lambda2),
		  stan::prob::poisson_log<true>(n1, lambda1),
		  stan::prob::poisson_log<true>(n2, lambda2),
		  message);
}

using stan::agrad::var;

TEST(AgradDistributionsPoisson,Propto) {
  unsigned int n;

  n = 0;
  expect_propto<var>(n, 10,
		     n, 15,
		     "var: lambda");
  
  n = 1;
  expect_propto<var>(n, 11,
		     n, 20,
		     "var: lambda");

  n = 123;
  expect_propto<var>(n, 30,
		     n, 15,
		     "var: lambda");
}
