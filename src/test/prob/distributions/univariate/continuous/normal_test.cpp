#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

using boost::math::policies::policy;
using boost::math::policies::domain_error;
using boost::math::policies::errno_on_error;

typedef policy<
  domain_error<errno_on_error>
  > errno_policy;


TEST(ProbDistributionsNormal,Normal) {
  // values from R dnorm()
  EXPECT_FLOAT_EQ(-0.9189385, stan::prob::normal_log(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-1.418939,  stan::prob::normal_log(1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-2.918939,  stan::prob::normal_log(-2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-3.174270,  stan::prob::normal_log(-3.5,1.9,7.2));
}
TEST(ProbDistributionsNormal,DefaultPolicyScale) {
  double sigma_d = 0.0;
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_d), std::domain_error);
  sigma_d = -1.0;
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_d), std::domain_error);

  sigma_d = 1.0;
  EXPECT_NO_THROW(stan::prob::normal_log(0.0,0.0,sigma_d));
  
  sigma_d = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(stan::prob::normal_log(0.0,0.0,sigma_d));
}
TEST(ProbDistributionsNormal,ErrnoPolicyScale) {
  double sigma_d = 0.0;
  double result = 0;
  
  result = stan::prob::normal_log(0.0,0.0,sigma_d, errno_policy());
  EXPECT_TRUE (std::isnan (result));
  
  sigma_d = -1.0;
  result = stan::prob::normal_log(0.0,0.0,sigma_d, errno_policy());
  EXPECT_TRUE (std::isnan (result));

  sigma_d = std::numeric_limits<double>::infinity();
  result = stan::prob::normal_log(0.0,0.0,sigma_d, errno_policy());
  EXPECT_FALSE (std::isnan (result));
}
TEST(ProbDistributionsNormal,DefaultPolicyY) {
  double y = 0.0;
  double result = 0;
  EXPECT_NO_THROW (stan::prob::normal_log(y,0.0,1.0));
  
  y = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::normal_log(y,0.0,1.0), std::domain_error);
  y = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(result = stan::prob::normal_log(y,0.0,1.0));
  EXPECT_FLOAT_EQ(log(0.0), result);
  y = -std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(result = stan::prob::normal_log(y,0.0,1.0));
  EXPECT_FLOAT_EQ(log(0.0), result);
}
TEST(ProbDistributionsNormal,ErrnoPolicyY) {
  double result = 0;
  double y = 0.0;
  EXPECT_NO_THROW (result = stan::prob::normal_log(y,0.0,1.0,errno_policy()));
  
  y = std::numeric_limits<double>::quiet_NaN();
  result = stan::prob::normal_log(y,0.0,1.0,errno_policy());
  EXPECT_TRUE (std::isnan(result));
  
  y = std::numeric_limits<double>::infinity();
  result = stan::prob::normal_log(y,0.0,1.0,errno_policy());
  EXPECT_FLOAT_EQ(log(0.0), result);

  y = -std::numeric_limits<double>::infinity();
  result = stan::prob::normal_log(y,0.0,1.0,errno_policy());
  EXPECT_FLOAT_EQ(log(0.0), result);
}
TEST(ProbDistributionsNormal,DefaultPolicyLocation) {
  double mu = 0.0;
  EXPECT_NO_THROW (stan::prob::normal_log(0.0,mu,1.0));
  
  mu = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::normal_log(0.0,mu,1.0), std::domain_error);
  mu = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::normal_log(0.0,mu,1.0), std::domain_error);
  mu = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::normal_log(0.0,mu,1.0), std::domain_error);
}
TEST(ProbDistributionsNormal,ErrnoPolicyLocation) {
  double result = 0;
  double mu = 0.0;
  EXPECT_NO_THROW (result = stan::prob::normal_log(0.0,mu,1.0,errno_policy()));
  
  mu = std::numeric_limits<double>::quiet_NaN();
  result = stan::prob::normal_log(0.0,mu,1.0,errno_policy());
  EXPECT_TRUE (std::isnan(result));
  
  mu = std::numeric_limits<double>::infinity();
  result = stan::prob::normal_log(0.0,mu,1.0,errno_policy());
  EXPECT_TRUE (std::isnan(result));

  mu = -std::numeric_limits<double>::infinity();
  result = stan::prob::normal_log(0.0,mu,1.0,errno_policy());
  EXPECT_TRUE (std::isnan(result));
}
TEST(ProbDistributionsNormal,Propto) {
  EXPECT_FLOAT_EQ(0, stan::prob::normal_log<true>(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(0, stan::prob::normal_log<true>(1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(0, stan::prob::normal_log<true>(-2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(0, stan::prob::normal_log<true>(-3.5,1.9,7.2));
}
TEST(ProbDistributionsNormal,Cumulative) {
  EXPECT_FLOAT_EQ(0.5000000, stan::prob::normal_p(0.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.8413447, stan::prob::normal_p(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.4012937, stan::prob::normal_p(1.0, 2.0, 4.0));
}
TEST(ProbDistributionsNormal,CumulativeDefaultPolicySigma) {
  // sigma = 0
  EXPECT_THROW(stan::prob::normal_p(0.0, 0.0, 0.0), std::domain_error);
  // sigma = -1
  EXPECT_THROW(stan::prob::normal_p(0.0, 0.0, -1.0), std::domain_error);  
}
TEST(ProbDistributionsNormal,NormalVector) {
  double mu = -2.9;
  double sigma = 1.7;

  std::vector<double> x;


  // EXPECT_FLOAT_EQ(0.0, stan::prob::normal_log(x,mu,sigma));

  x.push_back(-2.0);
  x.push_back(-1.5);
  x.push_back(0.0);
  x.push_back(12.0);
  
  EXPECT_FLOAT_EQ(-46.14256, stan::prob::normal_log(x,mu,sigma));
}
TEST(ProbDistributionsNormal,ProptoVector) {
  std::vector<double> x;
  EXPECT_FLOAT_EQ(0.0, stan::prob::normal_log<true>(x,0.0,1.0));

  x.push_back(-2.0);
  x.push_back(-1.5);
  x.push_back(0.0);
  x.push_back(12.0);

  EXPECT_FLOAT_EQ(0.0, stan::prob::normal_log<true>(x,0.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::normal_log<true>(x,0.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::normal_log<true>(x,0.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::normal_log<true>(x,1.9,7.2));
}
TEST(ProbDistributionsNormal,DefaultPolicyVectorScale) {
  std::vector<double> x;
  x.push_back(-2.0);
  x.push_back(-1.5);
  x.push_back(0.0);
  x.push_back(12.0);

  double sigma_d = 0.0;
  EXPECT_THROW(stan::prob::normal_log(x,0.0,sigma_d), std::domain_error);
  sigma_d = -1.0;
  EXPECT_THROW(stan::prob::normal_log(x,0.0,sigma_d), std::domain_error);
}
TEST(ProbDistributionsNormal,ErrnoPolicyNormalVectorScale) {
  std::vector<double> x;
  x.push_back(-2.0);
  x.push_back(-1.5);
  x.push_back(0.0);
  x.push_back(12.0);
  
  double result = 0;
  
  double sigma_d = 0.0;
  result = stan::prob::normal_log(x, 0.0, sigma_d, errno_policy());
  EXPECT_TRUE (std::isnan(result));
  
  sigma_d = -1.0;
  result = stan::prob::normal_log(x, 0.0, sigma_d, errno_policy());
  EXPECT_TRUE (std::isnan(result));
}
TEST(ProbDistributionsNormal,DefaultPolicyNormalVectorY) {
  std::vector<double> y;
  y.push_back(-2.0);
  y.push_back(-1.5);
  y.push_back(0.0);
  y.push_back(12.0);

  EXPECT_NO_THROW (stan::prob::normal_log(y,0.0,1.0));
  
  y[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::normal_log(y,0.0,1.0), std::domain_error);
  y[1] = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(stan::prob::normal_log(y,0.0,1.0));
  y[1] = -std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(stan::prob::normal_log(y,0.0,1.0));
}
TEST(ProbDistributionsNormal,ErrnoPolicyNormalVectorY) {
  double result = 0;
  std::vector<double> y;
  y.push_back(-2.0);
  y.push_back(-1.5);
  y.push_back(0.0);
  y.push_back(12.0);
  EXPECT_NO_THROW (result = stan::prob::normal_log(y,0.0,1.0,errno_policy()));
  
  y[2] = std::numeric_limits<double>::quiet_NaN();
  result = stan::prob::normal_log(y,0.0,1.0,errno_policy());
  EXPECT_TRUE (std::isnan(result));
  
  y[2] = std::numeric_limits<double>::infinity();
  result = stan::prob::normal_log(y,0.0,1.0,errno_policy());
  EXPECT_FALSE (std::isnan(result));

  y[2] = -std::numeric_limits<double>::infinity();
  result = stan::prob::normal_log(y,0.0,1.0,errno_policy());
  EXPECT_FALSE (std::isnan(result));
}
TEST(ProbDistributionsNormal,DefaultPolicyNormalVectorLocation) {
  std::vector<double> y;
  y.push_back(-2.0);
  y.push_back(-1.5);
  y.push_back(0.0);
  y.push_back(12.0);

  double mu = 0.0;
  EXPECT_NO_THROW (stan::prob::normal_log(y,mu,1.0));
  
  mu = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::normal_log(y,mu,1.0), std::domain_error);
  mu = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::normal_log(y,mu,1.0), std::domain_error);
  mu = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::normal_log(y,mu,1.0), std::domain_error);
}
TEST(ProbDistributionsNormal,ErrnoPolicyNormalVectorLocation) {
  double result = 0;
  double mu = 0.0;
  std::vector<double> y;
  y.push_back(-2.0);
  y.push_back(-1.5);
  y.push_back(0.0);
  y.push_back(12.0);
  EXPECT_NO_THROW (result = stan::prob::normal_log(y,mu,1.0,errno_policy()));
  
  mu = std::numeric_limits<double>::quiet_NaN();
  result = stan::prob::normal_log(y,mu,1.0,errno_policy());
  EXPECT_TRUE (std::isnan(result));
  
  mu = std::numeric_limits<double>::infinity();
  result = stan::prob::normal_log(y,mu,1.0,errno_policy());
  EXPECT_TRUE (std::isnan(result));

  mu = -std::numeric_limits<double>::infinity();
  result = stan::prob::normal_log(y,mu,1.0,errno_policy());
  EXPECT_TRUE (std::isnan(result));
}

