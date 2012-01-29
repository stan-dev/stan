#include <gtest/gtest.h>
#include "test/agrad/distributions/expect_eq_diffs.hpp"
#include "stan/agrad/agrad.hpp"
#include "stan/agrad/special_functions.hpp"
#include "stan/prob/distributions/univariate/continuous/chi_square.hpp"


template <typename T_y, typename T_dof>
void expect_propto(T_y y1, T_dof nu1,
                   T_y y2, T_dof nu2,
                   std::string message) {
  expect_eq_diffs(stan::prob::chi_square_log<false>(y1,nu1),
                  stan::prob::chi_square_log<false>(y2,nu2),
                  stan::prob::chi_square_log<true>(y1,nu1),
                  stan::prob::chi_square_log<true>(y2,nu2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsChiSquare,Propto) {
  expect_propto<var,var>(3.0,6.9,
                         3.4,9.0,
                         "var: y and nu");
}
TEST(AgradDistributionsChiSquare,ProptoY) {
  double nu = 4.0;
  
  expect_propto<var,double>(3.0, nu,
                            7.0, nu,
                            "var: y");
}
TEST(AgradDistributionsChiSquare,ProptoNu) {
  double y = 2.0;
  
  expect_propto<double,var>(y, 0.2,
                            y, 6.0,
                            "var: nu");
}
