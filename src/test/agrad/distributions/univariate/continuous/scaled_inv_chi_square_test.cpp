#include <gtest/gtest.h>
#include "test/agrad/distributions/expect_eq_diffs.hpp"
#include "stan/agrad/agrad.hpp"
#include "stan/agrad/special_functions.hpp"
#include "stan/prob/distributions/univariate/continuous/scaled_inv_chi_square.hpp"


template <typename T_y, typename T_dof, typename T_scale>
void expect_propto(T_y y1, T_dof nu1, T_scale s1,
                   T_y y2, T_dof nu2, T_scale s2,
                   std::string message) {
  expect_eq_diffs(stan::prob::scaled_inv_chi_square_log<false>(y1,nu1,s1),
                  stan::prob::scaled_inv_chi_square_log<false>(y2,nu2,s2),
                  stan::prob::scaled_inv_chi_square_log<true>(y1,nu1,s1),
                  stan::prob::scaled_inv_chi_square_log<true>(y2,nu2,s2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsScaledInvChiSquare,Propto) {
  expect_propto<var,var,var>(12.7, 6.1, 3.0,
                             1.0, 1.0, 0.5,
                             "var: y, nu, and s");
}
TEST(AgradDistributionsScaledInvChiSquare,ProptoY) {
  double nu = 6.1;
  double s = 3.0;
  
  expect_propto<var,double,double>(3.0, nu, s,
                                   7.0, nu, s,
                                   "var: y");
}
TEST(AgradDistributionsScaledInvChiSquare,ProptoYNu) {
  double s = 3.0;
  
  expect_propto<var,var,double>(3.0, 6.1, s,
                                7.0, 1.0, s,
                                "var: y and nu");
}
TEST(AgradDistributionsScaledInvChiSquare,ProptoYS) {
  double nu = 6.1;
  
  expect_propto<var,double,var>(3.0, nu, 3.0,
                                7.0, nu, 0.5,
                                "var: y and s");
}
TEST(AgradDistributionsScaledInvChiSquare,ProptoNu) {
  double y = 12.7;
  double s = 0.5;
  
  expect_propto<double,var,double>(y, 0.2, s,
                                   y, 6.0, s,
                                   "var: nu");
}
TEST(AgradDistributionsScaledInvChiSquare,ProptoNuS) {
  double y = 12.7;
  
  expect_propto<double,var,var>(y, 0.2, 3.0,
                                y, 6.0, 0.5,
                                "var: nu and s");
}
TEST(AgradDistributionsScaledInvChiSquare,ProptoS) {
  double y = 12.7;
  double nu = 6.1;
  
  expect_propto<var,double,var>(y, nu, 3.0,
                                y, nu, 0.5,
                                "var: s");
}


