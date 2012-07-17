#include <gtest/gtest.h>
#include "test/agrad/distributions/expect_eq_diffs.hpp"
#include "stan/prob/distributions/univariate/continuous/cauchy.hpp"
#include "stan/agrad/agrad.hpp"
#include "stan/meta/traits.hpp"


template <typename T_y, typename T_loc, typename T_scale>
void expect_propto(T_y y1, T_loc mu1, T_scale sigma1,
                   T_y y2, T_loc mu2, T_scale sigma2,
                   std::string message = "") {
  expect_eq_diffs(stan::prob::cauchy_log<false>(y1,mu1,sigma1),
                  stan::prob::cauchy_log<false>(y2,mu2,sigma2),
                  stan::prob::cauchy_log<true>(y1,mu1,sigma1),
                  stan::prob::cauchy_log<true>(y2,mu2,sigma2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsCauchy,Propto) {
  expect_propto<var,var,var>(1.0,2.0,10.0, 
                             0.1,0.0,1.0,
                             "All vars: y, mu, sigma");
}
TEST(AgradDistributionsCauchy,ProptoY) {
  double mu;
  double sigma;
  
  mu = 10.0;
  sigma = 4.0;
  expect_propto<var,double,double>(20.0,mu,sigma,
                                   15.0,mu,sigma,
                                   "var: y");

}
TEST(AgradDistributionsCauchy,ProptoYMu) {
  double sigma;
  sigma = 5.0;

  expect_propto<var,var,double>(20.0,15.0,sigma,
                                15.0,14.0,sigma,
                                "var: y and mu");
  
}
TEST(AgradDistributionsCauchy,ProptoYSigma) {
  double mu;
  mu = -5.0;

  expect_propto<var,double,var>(-3.0,mu,4.0,
                                -6.0,mu,10.0,
                                "var: y and sigma");
}
TEST(AgradDistributionsCauchy,ProptoMu) {
  double y;
  double sigma;
  
  y = 2.0;
  sigma = 10.0;
  expect_propto<double,var,double>(y,1.0,sigma,
                                   y,-1.0,sigma,
                                   "var: mu");
}
TEST(AgradDistributionsCauchy,ProptoMuSigma) {
  double y;
  
  y = 2.0;
  expect_propto<double,var,var>(y,1.0,3.0,
                                y,-1.0,4.0,
                                "var: mu and sigma");

}
TEST(AgradDistributionsCauchy,ProptoSigma) {
  double y;
  double mu;
  
  y = 2.0;
  mu = -1.0;
  expect_propto<double,double,var>(y,mu,10.0,
                                   y,mu,5.0,
                                   "var: sigma");
}

template <bool Prop, typename T_y, typename T_loc, typename T_scale>
var test_cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
  using stan::math::log1p;
  using stan::math::square;
  using stan::prob::include_summand;

  var lp = 0.0;
  if (include_summand<Prop>::value)
    lp += stan::prob::NEG_LOG_PI;
  if (include_summand<Prop,T_scale>::value)
    lp -= log(sigma);
  if (include_summand<Prop,T_y,T_loc,T_scale>::value)
    lp -= log1p(square((y - mu) / sigma));
  return lp;
}


template <typename T_y, typename T_loc, typename T_scale>
void gradient_test(double y, double mu, double sigma) {
  using stan::math::value_of;
  using stan::is_constant;
  using stan::prob::cauchy_log;

  T_y y1(y);
  T_loc mu1(mu);
  T_scale sigma1(sigma);
  var logp1 = cauchy_log<true>(y1, mu1, sigma1);
  stan::agrad::grad(logp1.vi_);
  double dy1 = var(y1).adj();
  double dmu1 = var(mu1).adj();
  double dsigma1 = var(sigma1).adj();
  
  T_y y2(y);
  T_loc mu2(mu);
  T_scale sigma2(sigma);
  var logp2 = test_cauchy_log<true>(y2, mu2, sigma2);
  stan::agrad::grad(logp2.vi_);
  double dy2 = var(y2).adj();
  double dmu2 = var(mu2).adj();
  double dsigma2 = var(sigma2).adj();

  EXPECT_FLOAT_EQ(logp2.val(), logp1.val());
  if (!is_constant<T_y>::value)
    EXPECT_FLOAT_EQ(dy2, dy1);
  if (!is_constant<T_loc>::value)
    EXPECT_FLOAT_EQ(dmu2, dmu1);
  if (!is_constant<T_scale>::value)
    EXPECT_FLOAT_EQ(dsigma2, dsigma1);
}

TEST(AgradDistributionsCauchy,GradientTest) {
  gradient_test<var,var,var>(0.3, 2.0, 3.0);
  gradient_test<var,var,double>(0.3, 2.0, 3.0);
  gradient_test<var,double,var>(0.3, 2.0, 3.0);
  gradient_test<var,double,double>(0.3, 2.0, 3.0);
  gradient_test<double,var,var>(0.3, 2.0, 3.0);
  gradient_test<double,var,double>(0.3, 2.0, 3.0);
  gradient_test<double,double,var>(0.3, 2.0, 3.0);
}
