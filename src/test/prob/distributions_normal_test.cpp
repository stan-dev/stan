#include <cmath>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <boost/math/policies/policy.hpp>
#include "stan/prob/distributions_normal.hpp"

using boost::math::policies::policy;
using boost::math::policies::evaluation_error;
using boost::math::policies::domain_error;
using boost::math::policies::overflow_error;
using boost::math::policies::domain_error;
using boost::math::policies::pole_error;
using boost::math::policies::errno_on_error;

typedef policy<
  domain_error<errno_on_error>, 
  pole_error<errno_on_error>,
  overflow_error<errno_on_error>,
  evaluation_error<errno_on_error> 
  > my_policy;


TEST(ProbDistributions,Normal) {
  // values from R dnorm()
  EXPECT_FLOAT_EQ(-0.9189385, stan::prob::normal_log(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-1.418939,  stan::prob::normal_log(1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-2.918939,  stan::prob::normal_log(-2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-3.174270,  stan::prob::normal_log(-3.5,1.9,7.2));
}
TEST(ProbDistributions,NormalDefaultPolicy) {
  double sigma_d = 0.0;
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_d), std::domain_error);
  sigma_d = -1.0;
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_d), std::domain_error);
}
TEST(ProbDistributions,NormalErrnoPolicy) {

  double sigma_d = 0.0;
  double result = 0;
  
  result = stan::prob::normal_log(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan (result));
  
  sigma_d = -1.0;
  result = stan::prob::normal_log(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan (result));
}
TEST(ProbDistributions,NormalPropTo) {
  EXPECT_FLOAT_EQ(0, stan::prob::normal_log<true>(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(0, stan::prob::normal_log<true>(1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(0, stan::prob::normal_log<true>(-2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(0, stan::prob::normal_log<true>(-3.5,1.9,7.2));
}
TEST(ProbDistributions,NormalPropToDefaultPolicy) {
  double sigma_d = 0.0;
  EXPECT_THROW(stan::prob::normal_log<true>(0.0,0.0,sigma_d), std::domain_error);
  sigma_d = -1.0;
  EXPECT_THROW(stan::prob::normal_log<true>(0.0,0.0,sigma_d), std::domain_error);
}


TEST(ProbDistributions,NormalPropToErronoPolicy) {
  double sigma_d = 0.0;
  double result = 0;
  
  result = stan::prob::normal_log<true>(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan(result));
  
  sigma_d = -1.0;
  result = stan::prob::normal_log<true>(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan(result));
}

TEST(ProbDistributionsCumulative,Normal) {
  EXPECT_FLOAT_EQ(0.5000000, stan::prob::normal_p(0.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.8413447, stan::prob::normal_p(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.4012937, stan::prob::normal_p(1.0, 2.0, 4.0));
}
TEST(ProbDistributionsCumulative,NormalDefaultPolicySigma) {
  // sigma = 0
  EXPECT_THROW(stan::prob::normal_p(0.0, 0.0, 0.0), std::domain_error);
  // sigma = -1
  EXPECT_THROW(stan::prob::normal_p(0.0, 0.0, -1.0), std::domain_error);  
}

TEST(ProbDistributions,NormalVec) {
  double mu = -2.9;
  double sigma = 1.7;

  std::vector<double> x;
  EXPECT_FLOAT_EQ(0.0, stan::prob::normal_log(x,mu,sigma));

  x.push_back(-2.0);
  x.push_back(-1.5);
  x.push_back(0.0);
  x.push_back(12.0);
  
  double lp = 0.0;
  for (unsigned int i = 0; i < x.size(); ++i)
    lp += stan::prob::normal_log(x[i],mu,sigma);
		  
  EXPECT_FLOAT_EQ(lp, stan::prob::normal_log(x,mu,sigma));
}

