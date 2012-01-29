#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/univariate/continuous/pareto.hpp>

template <typename T_y, typename T_scale, typename T_shape>
void expect_propto(T_y y1, T_scale y_min1, T_shape alpha1,
                   T_y y2, T_scale y_min2, T_shape alpha2,
                   std::string message) {
  expect_eq_diffs(stan::prob::pareto_log<false>(y1,y_min1,alpha1),
                  stan::prob::pareto_log<false>(y2,y_min2,alpha2),
                  stan::prob::pareto_log<true>(y1,y_min1,alpha1),
                  stan::prob::pareto_log<true>(y2,y_min2,alpha2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsPareto,Propto) {
  expect_propto<var,var,var>(0.2,2.5,2.0,
                             0.9,5.0,3.0,
                             "var: y, y_min, alpha");
}
TEST(AgradDistributionsPareto,ProptoY) {
  double y_min = 3.0;
  double alpha = 2.0;
  
  expect_propto<var,double,double>(6.0, y_min, alpha,
                                   10.0, y_min, alpha,
                                   "var: y");
}
TEST(AgradDistributionsPareto,ProptoYYMin) {
  double alpha = 2.0;
  
  expect_propto<var,var,double>(5.0, 1.0, alpha,
                                4.1, 4.0, alpha,
                                "var: y and y_min");
}
TEST(AgradDistributionsPareto,ProptoYAlpha) {
  double y_min = 2.0;
  
  expect_propto<var,double,var>(2.1, y_min, 0.5,
                                3.1, y_min, 3.0,
                                "var: y and alpha");
}
TEST(AgradDistributionsPareto,ProptoYMin) {
  double y = 15.0;
  double alpha = 3.0;
  
  expect_propto<double,var,double>(y, 0.2, alpha,
                                   y, 6.0, alpha,
                                   "var: y_min");
}
TEST(AgradDistributionsPareto,ProptoYMinAlpha) {
  double y = 0.4;
  
  expect_propto<double,var,var>(y, 0.6, 3.0,
                                y, 5.0, 1.4,
                                "var: y_min and alpha");
}
TEST(AgradDistributionsPareto,ProptoAlpha) {
  double y = 6.0;
  double y_min = 0.4;
  
  expect_propto<double,double,var>(y, y_min, 1.0,
                                   y, y_min, 1.5,
                                   "var: alpha");
}
