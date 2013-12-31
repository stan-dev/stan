#include <gtest/gtest.h>
#include <test/unit-distribution/expect_eq_diffs.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/multivariate/discrete/categorical.hpp>

template <typename T_prob>
void expect_propto(unsigned int n1, T_prob theta1, 
                   unsigned int n2, T_prob theta2, 
                   std::string message) {
  expect_eq_diffs(stan::prob::categorical_log<false>(n1, theta1),
                  stan::prob::categorical_log<false>(n2, theta2),
                  stan::prob::categorical_log<true>(n1, theta1),
                  stan::prob::categorical_log<true>(n2, theta2),
                  message);
}

using stan::agrad::var;
using Eigen::Dynamic;
using Eigen::Matrix;


TEST(AgradDistributionsCategorical,Propto) {
  unsigned int n;
  Matrix<var,Dynamic,1> theta1(3,1);
  theta1 << 0.3, 0.5, 0.2;
  Matrix<var,Dynamic,1> theta2(3,1);
  theta2 << 0.1, 0.2, 0.7;
  
  n = 1;
  expect_propto(n, theta1,
                n, theta2,
                "var: theta");
  
  n = 2;
  expect_propto(n, theta1,
                n, theta2,
                "var: theta");

  n = 3;
  expect_propto(n, theta1,
                n, theta2,
                "var: theta");
}
