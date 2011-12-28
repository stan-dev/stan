#include <gtest/gtest.h>
#include "test/agrad/distributions/expect_eq_diffs.hpp"
#include "stan/prob/distributions/uniform.hpp"
#include "stan/agrad/agrad.hpp"
#include "stan/meta/traits.hpp"

template <typename T_y, typename T_loc, typename T_scale>
void expect_propto(T_y y1, T_loc mu1, T_scale sigma1,
		     T_y y2, T_loc mu2, T_scale sigma2,
		   std::string message) {
  expect_eq_diffs(stan::prob::uniform_log<false>(y1,mu1,sigma1),
		  stan::prob::uniform_log<false>(y2,mu2,sigma2),
		  stan::prob::uniform_log<true>(y1,mu1,sigma1),
		  stan::prob::uniform_log<true>(y2,mu2,sigma2),
		  message);
}

using stan::agrad::var;

TEST(AgradDistributionsUniform,Propto) {
  expect_propto<var,var,var>(-2.0,-2.5,2.0,
				   3.0,-5.0,5.0,
				   "y is within both lower and upper bounds.");

  expect_propto<var,var,var>(2.0,-2.0,4.0,
				   10.0,-4.0,2.0,
				   "y is within one set of bounds and not the other");

  expect_propto<var,var,var>(-20.0,-10.0,-5.0,
				   20.0,1.0,5.0,
				   "y is outside both bounds");
}

TEST(AgradDistributionsUniform,ProptoY) {
  double lb;
  double ub;
  
  lb = -10.0;
  ub = 5.0;
  
  expect_propto<double,double,var>(-2.0,lb,ub,
				   3.0,lb,ub,
				   "y is within both lower and upper bounds.");

  expect_propto<double,double,var>(2.0,lb,ub,
				   10.0,lb,ub,
				   "y is within one set of bounds and not the other");

  expect_propto<double,double,var>(-20.0,lb,ub,
				   20.0,lb,ub,
				   "y is outside both bounds");
}

TEST(AgradDistributionsUniform,ProptoYLower) {
  double ub;
  
  ub = 5.0;
  
  expect_propto<double,double,var>(-2.0,-5.0,ub,
				   3.0,2.0,ub,
				   "y is within both lower and upper bounds.");

  expect_propto<double,double,var>(2.0,-5.0,ub,
				   10.0,3.0,ub,
				   "y is within one set of bounds and not the other");

  expect_propto<double,double,var>(-20.0,0.5,ub,
				   20.0,-5.0,ub,
				   "y is outside both bounds");
}
TEST(AgradDistributionsUniform,ProptoYUpper) {
  double lb;
  
  lb = -10.0;
  
  expect_propto<double,double,var>(-2.0,lb,2.0,
				   3.0,lb,5.0,
				   "y is within both lower and upper bounds.");

  expect_propto<double,double,var>(2.0,lb,4.0,
				   10.0,lb,2.0,
				   "y is within one set of bounds and not the other");

  expect_propto<double,double,var>(-20.0,lb,-5.0,
				   20.0,lb,5.0,
				   "y is outside both bounds");
}
TEST(AgradDistributionsUniform,ProptoLower) {
  double y;
  double ub;

  y = 2.5;
  ub = 20.0;
  expect_propto<double,double,var>(y,0.0,ub,
				   y,-5.0,ub,
				   "y is within both lower and upper bounds.");

  y = 5.0;
  ub = 20.0;
  expect_propto<double,double,var>(y,7.0,ub,
				   y,0.0,ub,
				   "y is within one set of bounds and not the other");

  y = -5.0;
  ub = 20.0;
  expect_propto<double,double,var>(y,1.0,ub,
				   y,10.0,ub,
				   "y is outside both bounds");
}
TEST(AgradDistributionsUniform,ProptoLowerUpper) {
  double y;

  y = 2.5;
  expect_propto<double,double,var>(y,0.0,5.0,
				   y,-5.0,3.0,
				   "y is within both lower and upper bounds.");

  expect_propto<double,double,var>(y,0.0,5.0,
				   y,-4.0,1.0,
				   "y is within one set of bounds and not the other");

  expect_propto<double,double,var>(y,3.0,10.0,
				   y,-10.0,-3.0,
				   "y is outside both bounds");
}
TEST(AgradDistributionsUniform,ProptoUpper) {
  double y;
  double lb;

  y = 2.5;
  lb = -5.0;
  expect_propto<double,double,var>(y,lb,3.0, 
				   y,lb,10.0,
				   "y is within both lower and upper bounds.");

  y = 5.0;
  lb = -5.0;
  expect_propto<double,double,var>(y,lb,3.0, 
				   y,lb,10.0,
				   "y is within one set of bounds and not the other");

  y = 100.0;
  lb = -5.0;
  expect_propto<double,double,var>(y,lb,3.0, 
				   y,lb,10.0,
				   "y is outside both bounds");
}



/*TEST(AgradDistributionsUniform,propto) {
  // d,d,d
  expect_propto<double,double,double>(2.0,2.0,3.0, 
				      2.0,2.0,3.0,
				      "d,d,d");
  // d,v,v
  expect_propto<double,var,var>(1.0,2.0,3.0, 
				1.0,-1.0,1.0,
				"d,v,v");
  // v,d,d
  expect_propto<var,double,double>(1.0,2.0,3.0, 
				   3.0,2.0,3.0,
				   "v,d,d");
  // v,d,v
  expect_propto<var,double,var>(1.0,2.0,3.0, 
				4.0,2.0,1.0,
				"v,d,v");
  // v,v,d
  expect_propto<var,var,double>(1.0,2.0,3.0, 
				2.0,-4.0,3.0,
				"v,v,d");
  // v,v,v
  expect_propto<var,var,var>(1.0,2.0,3.0, 
			     5.0,-7.0,1.5,
			     "v,v,v");
}
*/
