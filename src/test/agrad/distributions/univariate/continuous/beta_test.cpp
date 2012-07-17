#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/univariate/continuous/beta.hpp>

template <typename T_y, typename T_alpha, typename T_beta>
void expect_propto(T_y y1, T_alpha alpha1, T_beta beta1,
                   T_y y2, T_alpha alpha2, T_beta beta2,
                   std::string message) {
  expect_eq_diffs(stan::prob::beta_log<false>(y1,alpha1,beta1),
                  stan::prob::beta_log<false>(y2,alpha2,beta2),
                  stan::prob::beta_log<true>(y1,alpha1,beta1),
                  stan::prob::beta_log<true>(y2,alpha2,beta2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsBeta,Propto) {
  expect_propto<var,var,var>(0.2,2.5,2.0,
                             0.9,5.0,3.0,
                             "var: y, alpha, beta");
}
TEST(AgradDistributionsBeta,ProptoY) {
  double alpha = 3.0;
  double beta = 2.0;
  
  expect_propto<var,double,double>(0.2, alpha, beta,
                                   0.9, alpha, beta,
                                   "var: y");
}
TEST(AgradDistributionsBeta,ProptoYAlpha) {
  double beta = 2.0;
  
  expect_propto<var,var,double>(0.2, 1.0, beta,
                                0.9, 4.0, beta,
                                "var: y and alpha");
}
TEST(AgradDistributionsBeta,ProptoYBeta) {
  double alpha = 2.0;
  
  expect_propto<var,double,var>(0.2, alpha, 0.5,
                                0.9, alpha, 3.0,
                                "var: y and beta");
}
TEST(AgradDistributionsBeta,ProptoAlpha) {
  double y = 0.4;
  double beta = 3.0;
  
  expect_propto<double,var,double>(y, 0.2, beta,
                                   y, 6.0, beta,
                                   "var: alpha");
}
TEST(AgradDistributionsBeta,ProptoAlphaBeta) {
  double y = 0.4;
  
  expect_propto<double,var,var>(y, 0.6, 3.0,
                                y, 5.0, 1.4,
                                "var: alpha and beta");
}
TEST(AgradDistributionsBeta,ProptoBeta) {
  double y = 0.4;
  double alpha = 6.0;
  
  expect_propto<double,double,var>(y, alpha, 1.0,
                                   y, alpha, 1.5,
                                   "var: beta");
}

template <bool Prop, typename T_y, typename T_alpha, typename T_beta>
var test_beta_log(const T_y& y, const T_alpha& alpha, const T_beta& beta) {
  using stan::prob::include_summand;
  using stan::math::log1m;
  using stan::math::multiply_log;
  var lp = 0.0;
  if (include_summand<Prop,T_alpha,T_beta>::value)
    lp += lgamma(alpha + beta);
  if (include_summand<Prop,T_alpha>::value)
    lp -= lgamma(alpha);
  if (include_summand<Prop,T_beta>::value)
    lp -= lgamma(beta);
  if (include_summand<Prop,T_y,T_alpha>::value)
    lp += multiply_log(alpha-1.0, y);
  if (include_summand<Prop,T_y,T_beta>::value)
    lp += (beta - 1.0) * log1m(y);
  return lp;
}


template <typename T_y, typename T_alpha, typename T_beta>
void gradient_test(double y, double alpha, double beta) {
  using stan::math::value_of;
  using stan::is_constant;
  using stan::prob::beta_log;

  T_y y1(y);
  T_alpha alpha1(alpha);
  T_beta beta1(beta);
  var logp1 = beta_log<true>(y1, alpha1, beta1);
  stan::agrad::grad(logp1.vi_);
  double dy1 = var(y1).adj();
  double dalpha1 = var(alpha1).adj();
  double dbeta1 = var(beta1).adj();
  

  T_y y2(y);
  T_alpha alpha2(alpha);
  T_beta beta2(beta);
  var logp2 = test_beta_log<true>(y2, alpha2, beta2);
  stan::agrad::grad(logp2.vi_);
  double dy2 = var(y2).adj();
  double dalpha2 = var(alpha2).adj();
  double dbeta2 = var(beta2).adj();


  EXPECT_FLOAT_EQ(logp2.val(), logp1.val());
  if (!is_constant<T_y>::value)
    EXPECT_FLOAT_EQ(dy2, dy1);
  if (!is_constant<T_alpha>::value)
    EXPECT_FLOAT_EQ(dalpha2, dalpha1);
  if (!is_constant<T_beta>::value)
    EXPECT_FLOAT_EQ(dbeta2, dbeta1);
}

TEST(AgradDistributionsBeta,GradientTest) {
  gradient_test<var,var,var>(0.3, 2.0, 3.0);
  gradient_test<var,var,double>(0.3, 2.0, 3.0);
  gradient_test<var,double,var>(0.3, 2.0, 3.0);
  gradient_test<var,double,double>(0.3, 2.0, 3.0);
  gradient_test<double,var,var>(0.3, 2.0, 3.0);
  gradient_test<double,var,double>(0.3, 2.0, 3.0);
  gradient_test<double,double,var>(0.3, 2.0, 3.0);
}
