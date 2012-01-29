#include <gtest/gtest.h>
#include "test/agrad/distributions/expect_eq_diffs.hpp"
#include "stan/agrad/agrad.hpp"
#include "stan/agrad/special_functions.hpp"
#include "stan/meta/traits.hpp"
#include "stan/prob/distributions/univariate/continuous/inv_gamma.hpp"


template <typename T_y, typename T_shape, typename T_scale>
void expect_propto(T_y y1, T_shape alpha1, T_scale beta1,
                   T_y y2, T_shape alpha2, T_scale beta2,
                   std::string message) {
  expect_eq_diffs(stan::prob::inv_gamma_log<false>(y1,alpha1,beta1),
                  stan::prob::inv_gamma_log<false>(y2,alpha2,beta2),
                  stan::prob::inv_gamma_log<true>(y1,alpha1,beta1),
                  stan::prob::inv_gamma_log<true>(y2,alpha2,beta2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsInvGamma,Propto) {
  expect_propto<var,var,var>(5.0,2.5,2.0,
                             6.0,5.0,3.0,
                             "var: y, alpha, beta");
}
TEST(AgradDistributionsInvGamma,ProptoY) {
  double alpha = 3.0;
  double beta = 2.0;
  
  expect_propto<var,double,double>(3.0, alpha, beta,
                                   7.0, alpha, beta,
                                   "var: y");
}
TEST(AgradDistributionsInvGamma,ProptoYAlpha) {
  double beta = 2.0;
  
  expect_propto<var,var,double>(3.0, 1.0, beta,
                                  7.0, 4.0, beta,
                                "var: y and alpha");
}
TEST(AgradDistributionsInvGamma,ProptoYBeta) {
  double alpha = 2.0;
  
  expect_propto<var,double,var>(3.0, alpha, 0.5,
                                7.0, alpha, 3.0,
                                "var: y and beta");
}
TEST(AgradDistributionsInvGamma,ProptoAlpha) {
  double y = 2.0;
  double beta = 3.0;
  
  expect_propto<double,var,double>(y, 0.2, beta,
                                   y, 6.0, beta,
                                   "var: alpha");
}
TEST(AgradDistributionsInvGamma,ProptoAlphaBeta) {
  double y = 1.1;
  
  expect_propto<double,var,var>(y, 0.6, 3.0,
                                y, 5.0, 1.4,
                                "var: alpha and beta");
}
TEST(AgradDistributionsInvGamma,ProptoBeta) {
  double y = 3.0;
  double alpha = 6.0;
  
  expect_propto<double,double,var>(y, alpha, 1.0,
                                   y, alpha, 1.5,
                                   "var: beta");
}
