#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/beta.hpp>

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
