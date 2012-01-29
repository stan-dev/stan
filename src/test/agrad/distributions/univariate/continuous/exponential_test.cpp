#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/prob/distributions/univariate/continuous/exponential.hpp>

template <typename T_y, typename T_inv_scale>
void expect_propto(T_y y1, T_inv_scale beta1,
                   T_y y2, T_inv_scale beta2,
                   std::string message) {
  expect_eq_diffs(stan::prob::exponential_log<false>(y1,beta1),
                  stan::prob::exponential_log<false>(y2,beta2),
                  stan::prob::exponential_log<true>(y1,beta1),
                  stan::prob::exponential_log<true>(y2,beta2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsExponential,Propto) {
  expect_propto<var,var>(2.0, 1.5,
                         15.0, 3.9,
                         "var: y and beta");
}
TEST(AgradDistributionsExponential,ProptoY) {
  double beta = 1.5;
  expect_propto<var,double>(2.0, beta,
                            15.0, beta,
                            "var: y");
}
TEST(AgradDistributionsExponential,ProptoBeta) {
  double y = 15.0;
  expect_propto<double,var>(y, 1.5,
                            y, 3.9,
                            "var: beta");
}

