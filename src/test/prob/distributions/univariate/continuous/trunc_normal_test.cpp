#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/trunc_normal.hpp>

using boost::math::policies::policy;
using boost::math::policies::domain_error;
using boost::math::policies::errno_on_error;

typedef policy<
  domain_error<errno_on_error>
  > errno_policy;


TEST(ProbDistributionsTruncNormal,Normal) {
  // values from R dnorm()
  EXPECT_FLOAT_EQ(-0.9189385, stan::prob::trunc_normal_log(0.0,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(-1.418939,  stan::prob::trunc_normal_log(1.0,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(-2.918939,  stan::prob::trunc_normal_log(-2.0,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(-3.174270,  stan::prob::trunc_normal_log(-3.5,1.9,7.2,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
}
TEST(ProbDistributionsTruncNormal,DefaultPolicyScale) {
  double sigma_d = 0.0;
  EXPECT_THROW(stan::prob::trunc_normal_log(0.0,0.0,sigma_d,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()), std::domain_error);
  sigma_d = -1.0;
  EXPECT_THROW(stan::prob::trunc_normal_log(0.0,0.0,sigma_d,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()), std::domain_error);

  sigma_d = 1.0;
  EXPECT_NO_THROW(stan::prob::trunc_normal_log(0.0,0.0,sigma_d,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  
  sigma_d = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(stan::prob::trunc_normal_log(0.0,0.0,sigma_d,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
}
TEST(ProbDistributionsTruncNormal,ErrnoPolicyScale) {
  double sigma_d = 0.0;
  double result = 0;
  
  result = stan::prob::trunc_normal_log(0.0,0.0,sigma_d,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(), errno_policy());
  EXPECT_TRUE (std::isnan (result));
  
  sigma_d = -1.0;
  result = stan::prob::trunc_normal_log(0.0,0.0,sigma_d,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(), errno_policy());
  EXPECT_TRUE (std::isnan (result));

  sigma_d = std::numeric_limits<double>::infinity();
  result = stan::prob::trunc_normal_log(0.0,0.0,sigma_d,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(), errno_policy());
  EXPECT_FALSE (std::isnan (result));
}
TEST(ProbDistributionsTruncNormal,DefaultPolicyY) {
  double y = 0.0;
  double result = 0;
  EXPECT_NO_THROW (stan::prob::trunc_normal_log(y,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  
  y = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::trunc_normal_log(y,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()), std::domain_error);

  y = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(result = stan::prob::trunc_normal_log(y,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(log(0.0), result);

  y = -std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(result = stan::prob::trunc_normal_log(y,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(log(0.0), result);

  y = -2.0;
  EXPECT_NO_THROW(result = stan::prob::trunc_normal_log(y,0.0,1.0,-1.0,1.0));
  EXPECT_FLOAT_EQ(log(0.0), result);
  EXPECT_NO_THROW(result = stan::prob::trunc_normal_log(y,0.0,1.0,-1.0,std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(log(0.0), result);
  
  y = 2.0;
  EXPECT_NO_THROW(result = stan::prob::trunc_normal_log(y,0.0,1.0,-1.0,1.0));
  EXPECT_FLOAT_EQ(log(0.0), result);
  EXPECT_NO_THROW(result = stan::prob::trunc_normal_log(y,0.0,1.0,-std::numeric_limits<double>::infinity(),1.0));
  EXPECT_FLOAT_EQ(log(0.0), result);
}
TEST(ProbDistributionsTruncNormal,ErrnoPolicyY) {
  double result = 0;
  double y = 0.0;
  EXPECT_NO_THROW (result = stan::prob::trunc_normal_log(y,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),errno_policy()));
  
  y = std::numeric_limits<double>::quiet_NaN();
  result = stan::prob::trunc_normal_log(y,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),errno_policy());
  EXPECT_TRUE (std::isnan(result));
  
  y = std::numeric_limits<double>::infinity();
  result = stan::prob::trunc_normal_log(y,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),errno_policy());
  EXPECT_FLOAT_EQ(log(0.0), result);

  y = -std::numeric_limits<double>::infinity();
  result = stan::prob::trunc_normal_log(y,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),errno_policy());
  EXPECT_FLOAT_EQ(log(0.0), result);

  y = -2.0;
  result = stan::prob::trunc_normal_log(y,0.0,1.0,-1.0,1.0,errno_policy());
  EXPECT_FLOAT_EQ(log(0.0), result);
  result = stan::prob::trunc_normal_log(y,0.0,1.0,-1.0,std::numeric_limits<double>::infinity(),errno_policy());
  EXPECT_FLOAT_EQ(log(0.0), result);
  
  y = 2.0;
  result = stan::prob::trunc_normal_log(y,0.0,1.0,-1.0,1.0,errno_policy());
  EXPECT_FLOAT_EQ(log(0.0), result);
  result = stan::prob::trunc_normal_log(y,0.0,1.0,-std::numeric_limits<double>::infinity(),1.0,errno_policy());
  EXPECT_FLOAT_EQ(log(0.0), result);
}
TEST(ProbDistributionsTruncNormal,DefaultPolicyLocation) {
  double mu = 0.0;
  EXPECT_NO_THROW (stan::prob::trunc_normal_log(0.0,mu,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  
  mu = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::trunc_normal_log(0.0,mu,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()), std::domain_error);
  mu = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::trunc_normal_log(0.0,mu,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()), std::domain_error);
  mu = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::trunc_normal_log(0.0,mu,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()), std::domain_error);
}
TEST(ProbDistributionsTruncNormal,ErrnoPolicyLocation) {
  double result = 0;
  double mu = 0.0;
  EXPECT_NO_THROW (result = stan::prob::trunc_normal_log(0.0,mu,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),errno_policy()));
  
  mu = std::numeric_limits<double>::quiet_NaN();
  result = stan::prob::trunc_normal_log(0.0,mu,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),errno_policy());
  EXPECT_TRUE (std::isnan(result));
  
  mu = std::numeric_limits<double>::infinity();
  result = stan::prob::trunc_normal_log(0.0,mu,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),errno_policy());
  EXPECT_TRUE (std::isnan(result));

  mu = -std::numeric_limits<double>::infinity();
  result = stan::prob::trunc_normal_log(0.0,mu,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),errno_policy());
  EXPECT_TRUE (std::isnan(result));
}
TEST(ProbDistributionsTruncNormal,DefaultPolicyBnds) {
  double lb = -1.0;
  double ub = 1.0;
  EXPECT_NO_THROW (stan::prob::trunc_normal_log(0.0,0.0,1.0,lb,ub));

  lb = -std::numeric_limits<double>::infinity();
  ub = 1.0;
  EXPECT_NO_THROW (stan::prob::trunc_normal_log(0.0,0.0,1.0,lb,ub));

  lb = -1.0;
  ub = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW (stan::prob::trunc_normal_log(0.0,0.0,1.0,lb,ub));
  
  lb = std::numeric_limits<double>::quiet_NaN();
  ub = 1.0;
  EXPECT_THROW(stan::prob::trunc_normal_log(0.0,0.0,1.0,lb,ub),std::domain_error);
  lb = -1.0;
  ub = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::trunc_normal_log(0.0,0.0,1.0,lb,ub),std::domain_error);

  lb = std::numeric_limits<double>::infinity();
  ub = 1.0;
  EXPECT_THROW(stan::prob::trunc_normal_log(0.0,0.0,1.0,lb,ub),std::domain_error);

  ub = -std::numeric_limits<double>::infinity();
  lb = -1.0;
  EXPECT_THROW(stan::prob::trunc_normal_log(0.0,0.0,1.0,lb,ub),std::domain_error);

  ub = -1.0;
  lb = 1.0;
  EXPECT_THROW(stan::prob::trunc_normal_log(0.0,0.0,1.0,lb,ub),std::domain_error);
}
TEST(ProbDistributionsTruncNormal,Propto) {
  EXPECT_FLOAT_EQ(0, stan::prob::trunc_normal_log<true>(0.0,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(0, stan::prob::trunc_normal_log<true>(1.0,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(0, stan::prob::trunc_normal_log<true>(-2.0,0.0,1.0,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(0, stan::prob::trunc_normal_log<true>(-3.5,1.9,7.2,-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()));
}
