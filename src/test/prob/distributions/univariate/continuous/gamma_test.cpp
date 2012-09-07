#include <gtest/gtest.h>
#include "stan/prob/distributions/univariate/continuous/gamma.hpp"


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
  > errno_policy;


TEST(ProbDistributionsGamma,Gamma) {
  EXPECT_FLOAT_EQ(-0.6137056, stan::prob::gamma_log(1.0,2.0,2.0));
  EXPECT_FLOAT_EQ(-3.379803, stan::prob::gamma_log(2.0,0.25,0.75));
  EXPECT_FLOAT_EQ(-1, stan::prob::gamma_log(1,1,1));
  EXPECT_FLOAT_EQ(log(2.0), stan::prob::gamma_log(0.0,1.0,2.0));
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::gamma_log(-10.0,1.0,2.0));
}
TEST(ProbDistributionsGamma,Boundary) {
  double y;
  double alpha;
  double gamma;

  y = 0;
  alpha = 1;
  gamma = 1;
  EXPECT_FLOAT_EQ(0.0, stan::prob::gamma_log(y,alpha,gamma));
}
TEST(ProbDistributionsGamma,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::gamma_log<true>(1.0,2.0,2.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::gamma_log<true>(2.0,0.25,0.75));
  EXPECT_FLOAT_EQ(0.0, stan::prob::gamma_log<true>(1,1,1));
}
TEST(ProbDistributionsGamma,DefaultPolicy) {
  double y = 0.5;
  double alpha = 1.5;
  double beta = 2.0;
  
  EXPECT_NO_THROW(stan::prob::gamma_log(y, alpha, beta));
  EXPECT_NO_THROW(stan::prob::gamma_log(-1.0, alpha, beta));
  EXPECT_THROW (stan::prob::gamma_log(y, 0.0, beta), std::domain_error) <<
    "alpha = 0.0 should throw an exception";
  EXPECT_THROW (stan::prob::gamma_log(y, -1.0, beta), std::domain_error) <<
    "alpha < 0 should throw an exception. alpha = " << alpha;
  EXPECT_THROW (stan::prob::gamma_log(y, alpha, 0.0), std::domain_error) <<
    "beta = 0.0 should throw an exception";
  EXPECT_THROW (stan::prob::gamma_log(y, alpha, -1.0), std::domain_error) <<
    "beta < 0 should throw an exception. beta = " << beta;
}
TEST(ProbDistributionsGamma,ErrnoPolicy) {
  double result;
  double y = 0.5;
  double alpha = 1.5;
  double beta = 2.0;
  
  EXPECT_NO_THROW(result=stan::prob::gamma_log(y, alpha, beta, errno_policy()));
  EXPECT_FALSE(std::isnan(result));
  
  EXPECT_NO_THROW(result=stan::prob::gamma_log(-1.0, alpha, beta, errno_policy()));
  EXPECT_FALSE(std::isnan(result));
  
  EXPECT_NO_THROW(result=stan::prob::gamma_log(y, 0.0, beta, errno_policy()));
  EXPECT_TRUE(std::isnan(result));
  
  EXPECT_NO_THROW(result=stan::prob::gamma_log(y, -1.0, beta, errno_policy()));
  EXPECT_TRUE(std::isnan(result));

  EXPECT_NO_THROW(result=stan::prob::gamma_log(y, alpha, 0.0, errno_policy()));
  EXPECT_TRUE(std::isnan(result));

  EXPECT_NO_THROW(result=stan::prob::gamma_log(y, alpha, -1.0, errno_policy()));
  EXPECT_TRUE(std::isnan(result));
}


TEST(ProbDistributionsGamma,Cumulative) {
  // values from R
  EXPECT_FLOAT_EQ(0.59399415, stan::prob::gamma_cdf(1.0,2.0,2.0));
  EXPECT_FLOAT_EQ(0.96658356, stan::prob::gamma_cdf(2.0,0.25,0.75));
  EXPECT_FLOAT_EQ(0.63212056, stan::prob::gamma_cdf(1,1,1));
  EXPECT_FLOAT_EQ(0.0, stan::prob::gamma_cdf(0,1,1));
}
