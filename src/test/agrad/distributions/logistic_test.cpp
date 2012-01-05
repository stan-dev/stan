#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/prob/distributions/logistic.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/meta/traits.hpp>


template <typename T_y, typename T_loc, typename T_scale>
void expect_propto(T_y y1, T_loc mu1, T_scale sigma1,
		   T_y y2, T_loc mu2, T_scale sigma2,
		   std::string message = "") {
  expect_eq_diffs(stan::prob::logistic_log<false>(y1,mu1,sigma1),
		  stan::prob::logistic_log<false>(y2,mu2,sigma2),
		  stan::prob::logistic_log<true>(y1,mu1,sigma1),
		  stan::prob::logistic_log<true>(y2,mu2,sigma2),
		  message);
}

using stan::agrad::var;

TEST(AgradDistributionsLogistic,Propto) {
  expect_propto<var,var,var>(1.0,2.0,10.0, 
			     0.1,0.0,1.0,
			     "All vars: y, mu, sigma");
}
TEST(AgradDistributionsLogistic,ProptoY) {
  double mu;
  double sigma;
  
  mu = 10.0;
  sigma = 4.0;
  expect_propto<var,double,double>(20.0,mu,sigma,
				   15.0,mu,sigma,
				   "var: y");

}
TEST(AgradDistributionsLogistic,ProptoYMu) {
  double sigma;
  sigma = 5.0;

  expect_propto<var,var,double>(20.0,15.0,sigma,
				15.0,14.0,sigma,
				"var: y and mu");
  
}
TEST(AgradDistributionsLogistic,ProptoYSigma) {
  double mu;
  mu = -5.0;

  expect_propto<var,double,var>(-3.0,mu,4.0,
				-6.0,mu,10.0,
				"var: y and sigma");
}
TEST(AgradDistributionsLogistic,ProptoMu) {
  double y;
  double sigma;
  
  y = 2.0;
  sigma = 10.0;
  expect_propto<double,var,double>(y,1.0,sigma,
				   y,-1.0,sigma,
				   "var: mu");
}
TEST(AgradDistributionsLogistic,ProptoMuSigma) {
  double y;
  
  y = 2.0;
  expect_propto<double,var,var>(y,1.0,3.0,
				   y,-1.0,4.0,
				   "var: mu and sigma");

}
TEST(AgradDistributionsLogistic,ProptoSigma) {
  double y;
  double mu;
  
  y = 2.0;
  mu = -1.0;
  expect_propto<double,double,var>(y,mu,10.0,
				   y,mu,5.0,
				   "var: sigma");
}
