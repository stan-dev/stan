#include <gtest/gtest.h>
#include <test/unit-distribution/expect_eq_diffs.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/multivariate/discrete/multinomial.hpp>

template <typename T_prob>
void expect_propto(std::vector<int>& ns1, T_prob theta1, 
                   std::vector<int>& ns2, T_prob theta2, 
                   std::string message) {
  expect_eq_diffs(stan::prob::multinomial_log<false>(ns1, theta1),
                  stan::prob::multinomial_log<false>(ns2, theta2),
                  stan::prob::multinomial_log<true>(ns1, theta1),
                  stan::prob::multinomial_log<true>(ns2, theta2),
                  message);
}

using stan::agrad::var;
using Eigen::Dynamic;
using Eigen::Matrix;


TEST(AgradDistributionsMultinomial,Propto) {
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<var,Dynamic,1> theta1(3,1);
  theta1 << 0.3, 0.5, 0.2;
  Matrix<var,Dynamic,1> theta2(3,1);
  theta2 << 0.1, 0.2, 0.7;
  
  expect_propto(ns, theta1,
                ns, theta2,
                "var: theta");
}
