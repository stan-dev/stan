#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/weibull.hpp>


template <typename T_y, typename T_shape, typename T_inv_scale>
void expect_propto(T_y y1, T_shape alpha1, T_inv_scale beta1,
		   T_y y2, T_shape alpha2, T_inv_scale beta2,
		   std::string message) {
  expect_eq_diffs(stan::prob::weibull_log<false>(y1,alpha1,beta1),
		  stan::prob::weibull_log<false>(y2,alpha2,beta2),
		  stan::prob::weibull_log<true>(y1,alpha1,beta1),
		  stan::prob::weibull_log<true>(y2,alpha2,beta2),
		  message);
}

using stan::agrad::var;

TEST(AgradDistributionsWeibull,Boundary) {
  var y;
  var alpha;
  var sigma;

  y = 0;
  alpha = 1;
  sigma = 1;
  EXPECT_FLOAT_EQ(0.0, stan::prob::weibull_log(y,alpha,sigma).val());
}

TEST(AgradDistributionsWeibull,Propto) {
  expect_propto<var,var,var>(5.0,2.5,2.0,
			     6.0,5.0,3.0,
			     "var: y, alpha, sigma");
}
TEST(AgradDistributionsWeibull,ProptoY) {
  double alpha = 3.0;
  double sigma = 2.0;
  
  expect_propto<var,double,double>(3.0, alpha, sigma,
				   7.0, alpha, sigma,
				   "var: y");
}
TEST(AgradDistributionsWeibull,ProptoYAlpha) {
  double sigma = 2.0;
  
  expect_propto<var,var,double>(3.0, 1.0, sigma,
				  7.0, 4.0, sigma,
				"var: y and alpha");
}
TEST(AgradDistributionsWeibull,ProptoYSigma) {
  double alpha = 2.0;
  
  expect_propto<var,double,var>(3.0, alpha, 0.5,
				7.0, alpha, 3.0,
				"var: y and sigma");
}
TEST(AgradDistributionsWeibull,ProptoAlpha) {
  double y = 2.0;
  double sigma = 3.0;
  
  expect_propto<double,var,double>(y, 0.2, sigma,
				   y, 6.0, sigma,
				   "var: alpha");
}
TEST(AgradDistributionsWeibull,ProptoAlphaSigma) {
  double y = 1.1;
  
  expect_propto<double,var,var>(y, 0.6, 3.0,
				y, 5.0, 1.4,
				"var: alpha and sigma");
}
TEST(AgradDistributionsWeibull,ProptoSigma) {
  double y = 3.0;
  double alpha = 6.0;
  
  expect_propto<double,double,var>(y, alpha, 1.0,
				   y, alpha, 1.5,
				   "var: sigma");
}
