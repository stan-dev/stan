#include <cmath>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <boost/math/policies/policy.hpp>
#include "stan/prob/distributions_normal.hpp"

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

  double sigma_d = 0.0;
  double result = 0;
  
  result = stan::prob::normal_log(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan (result));
  
  sigma_d = -1.0;
  result = stan::prob::normal_log(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan (result));
}

TEST(ProbDistributions,NormalPropTo) {
  double diff = stan::prob::normal_propto_log (0.0, 0.0, 1.0) - (-0.9189385);
  
  EXPECT_FLOAT_EQ(-0.9189385 + diff, stan::prob::normal_propto_log(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-1.418939 + diff,  stan::prob::normal_propto_log(1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-2.918939 + diff,  stan::prob::normal_propto_log(-2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-3.174270 + diff,  stan::prob::normal_propto_log(-3.5,1.9,7.2));
}
TEST(ProbDistributions,NormalPropToDefaultPolicy) {
  double sigma_d = 0.0;
  EXPECT_THROW(stan::prob::normal_propto_log(0.0,0.0,sigma_d), std::domain_error);
  sigma_d = -1.0;
  EXPECT_THROW(stan::prob::normal_propto_log(0.0,0.0,sigma_d), std::domain_error);
}
TEST(ProbDistributions,NormalPropToErronoPolicy) {
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

  double sigma_d = 0.0;
  double result = 0;
  
  result = stan::prob::normal_propto_log(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan (result));
  
  sigma_d = -1.0;
  result = stan::prob::normal_propto_log(0.0,0.0,sigma_d, my_policy());
  EXPECT_TRUE (std::isnan (result));
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

TEST(ProbDistributionsCumulative,Normal) {
  // values from R pnorm()
  EXPECT_FLOAT_EQ(0.5000000, stan::prob::normal_p (0.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.8413447, stan::prob::normal_p (1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.4012937, stan::prob::normal_p (1.0, 2.0, 4.0));
}
TEST(ProbDistributionsCumulative,NormalDefaultPolicySigma) {
  double sigma; 
  
  // exception when sigma <= 0
  sigma = 0.0;
  EXPECT_THROW (stan::prob::normal_p (0.0, 0.0, sigma), std::domain_error);

  sigma = -1.0;
  EXPECT_THROW (stan::prob::normal_p (0.0, 0.0, sigma), std::domain_error);  
}

TEST(ProbDistributionsTruncated,NormalLowHigh) {
  // values from R dnorm()
  double mu;
  double sigma;
  double low;
  double high;
  
  mu = 0;
  sigma = 1.0;
  low = -2.0;
  high = 1.0;
  // mu <- 0; sigma <- 1.0; low <- -2.0; high <- 1.0;
  // R: log ( 0.0 )
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::prob::normal_trunc_lh_log(-5.0, mu, sigma, low, high));
  // R: log ( dnorm(-2.0, mu, sigma) / (pnorm (high, mu, sigma) - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-2.718772, stan::prob::normal_trunc_lh_log(-2.0, mu, sigma, low, high));
  // R: log ( dnorm(1.0, mu, sigma) / (pnorm (high, mu, sigma) - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-1.218772, stan::prob::normal_trunc_lh_log( 1.0, mu, sigma, low, high));
  // R: log ( 0.0 )
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::prob::normal_trunc_lh_log(10.0, mu, sigma, low, high));

  // R: log ( dnorm(0.0, mu, sigma) / (pnorm (high, mu, sigma) - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-0.7187722, stan::prob::normal_trunc_lh_log(0.0, mu, sigma, low, high));
  // R: log ( dnorm(0.5, mu, sigma) / (pnorm (high, mu, sigma) - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-0.8437722, stan::prob::normal_trunc_lh_log(0.5, mu, sigma, low, high));
  // R: log ( dnorm(-0.5, mu, sigma) / (pnorm (high, mu, sigma) - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-0.8437722, stan::prob::normal_trunc_lh_log(-0.5, mu, sigma, low, high));
}
TEST(ProbDistributionsTruncated,NormalLowHighDefaultPolicy) {
  double y = 0;
  double mu = 0;
  double sigma = 1;
  double low = -5;
  double high = 5;
  EXPECT_THROW(stan::prob::normal_trunc_lh_log(y, mu, 0.0, low, high), std::domain_error);
  EXPECT_THROW(stan::prob::normal_trunc_lh_log(y, mu, -1.0, low, high), std::domain_error);
  EXPECT_THROW(stan::prob::normal_trunc_lh_log(y, mu, sigma, high, low), std::domain_error);
  EXPECT_THROW(stan::prob::normal_trunc_lh_log(y, mu, sigma, low, low), std::domain_error);
}
TEST(ProbDistributionsTruncated,NormalLow) {
  // values from R dnorm()
  double mu;
  double sigma;
  double low;
  
  mu = 0;
  sigma = 1.0;
  low = -2.0;
  // mu <- 0; sigma <- 1.0; low <- -2.0; 
  // R: log ( 0.0 )
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::prob::normal_trunc_l_log(-5.0, mu, sigma, low));
  // R: log ( dnorm(-2.0, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-2.895926, stan::prob::normal_trunc_l_log(-2.0, mu, sigma, low));
  // R: log ( dnorm(1.0, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-1.395926, stan::prob::normal_trunc_l_log( 1.0, mu, sigma, low));
  // R: log ( dnorm(10.0, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-50.89593, stan::prob::normal_trunc_l_log(10.0, mu, sigma, low));

  // R: log ( dnorm(0.0, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-0.8959256, stan::prob::normal_trunc_l_log(0.0, mu, sigma, low));
  // R: log ( dnorm(0.5, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-1.020926, stan::prob::normal_trunc_l_log(0.5, mu, sigma, low));
  // R: log ( dnorm(-0.5, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-1.020926, stan::prob::normal_trunc_l_log(-0.5, mu, sigma, low));
}
TEST(ProbDistributionsTruncated,NormalLowDefaultPolicySigma) {
  double y = 0;
  double mu = 0;
  double sigma = 1;
  double low = -5;
  EXPECT_NO_THROW(stan::prob::normal_trunc_l_log(y, mu, sigma, low));
  EXPECT_THROW(stan::prob::normal_trunc_l_log(y, mu, 0.0, low), std::domain_error);
  EXPECT_THROW(stan::prob::normal_trunc_l_log(y, mu, -1.0, low), std::domain_error);
}
TEST(ProbDistributionsTruncated,NormalHigh) {
  // values from R dnorm()
  double mu;
  double sigma;
  double high;
  
  mu = 0;
  sigma = 1.0;
  high = 1.0;
  // mu <- 0; sigma <- 1.0; high <- 1.0
  // R: log ( 0.0 )
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::prob::normal_trunc_h_log(5.0, mu, sigma, high));
  // R: log ( dnorm(-2.0, mu, sigma) / pnorm(high, mu, sigma) )
  EXPECT_FLOAT_EQ(-2.746185, stan::prob::normal_trunc_h_log(-2.0, mu, sigma, high));
  // R: log ( dnorm(1.0, mu, sigma) / pnorm(high, mu, sigma) )
  EXPECT_FLOAT_EQ(-1.246185, stan::prob::normal_trunc_h_log( 1.0, mu, sigma, high));

  // R: log ( dnorm(0.0, mu, sigma) / pnorm(high, mu, sigma) )
  EXPECT_FLOAT_EQ(-0.7461848, stan::prob::normal_trunc_h_log(0.0, mu, sigma, high));
  // R: log ( dnorm(0.5, mu, sigma) / pnorm(high, mu, sigma) )
  EXPECT_FLOAT_EQ(-0.8711848, stan::prob::normal_trunc_h_log(0.5, mu, sigma, high));
  // R: log ( dnorm(-0.5, mu, sigma) / pnorm(high, mu, sigma) )
  EXPECT_FLOAT_EQ(-0.8711848, stan::prob::normal_trunc_h_log(-0.5, mu, sigma, high));
}
TEST(ProbDistributionsTruncated,NormalHighDefaultPolicySigma) {
  double y = 0;
  double mu = 0;
  double sigma = 1;
  double high = -5;
  EXPECT_NO_THROW(stan::prob::normal_trunc_h_log(y, mu, sigma, high));
  EXPECT_THROW(stan::prob::normal_trunc_h_log(y, mu, 0.0, high), std::domain_error);
  EXPECT_THROW(stan::prob::normal_trunc_h_log(y, mu, -1.0, high), std::domain_error);
}


