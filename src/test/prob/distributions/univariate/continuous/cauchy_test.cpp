#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/cauchy.hpp>

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

using stan::prob::cauchy_log;

TEST(ProbDistributionsCauchy,Cauchy) {
  EXPECT_FLOAT_EQ(-1.837877, stan::prob::cauchy_log(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(-2.323385, stan::prob::cauchy_log(-1.5, 0.0, 1.0));
  EXPECT_FLOAT_EQ(-2.323385, stan::prob::cauchy_log(-2.5, -1.0, 1.0));
  // need test with scale != 1
}
TEST(ProbDistributionsCauchy,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::cauchy_log<true>(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::cauchy_log<true>(-1.5, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::cauchy_log<true>(-2.5, -1.0, 1.0));
  // need test with scale != 1
}
TEST(ProbDistributionsCauchy,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  double y = 0.5;
  double mu = 2.0;
  double sigma = 3.0;
  
  EXPECT_NO_THROW(cauchy_log(y, mu, sigma));
  EXPECT_NO_THROW(cauchy_log(inf, mu, sigma));
  EXPECT_NO_THROW(cauchy_log(-inf, mu, sigma));
  EXPECT_NO_THROW(cauchy_log(y, 0.0, sigma));
  EXPECT_NO_THROW(cauchy_log(y, -1.0, sigma));
  
  EXPECT_THROW(cauchy_log(nan, mu, sigma), std::domain_error);

  EXPECT_THROW(cauchy_log(y, -inf, sigma), std::domain_error);
  EXPECT_THROW(cauchy_log(y, nan, sigma), std::domain_error);
  EXPECT_THROW(cauchy_log(y, inf, sigma), std::domain_error);

  EXPECT_THROW(cauchy_log(y, mu, 0.0), std::domain_error);
  EXPECT_THROW(cauchy_log(y, mu, -1.0), std::domain_error);
  EXPECT_THROW(cauchy_log(y, mu, -inf), std::domain_error);
  EXPECT_THROW(cauchy_log(y, mu, nan), std::domain_error);
  EXPECT_THROW(cauchy_log(y, mu, inf), std::domain_error);
}
TEST(ProbDistributionsCauchy,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double result;
  double y = 0.5;
  double mu = 2.0;
  double sigma = 3.0;
  
  result = cauchy_log(y, mu, sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = cauchy_log(inf, mu, sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = cauchy_log(-inf, mu, sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = cauchy_log(y, 0.0, sigma, errno_policy()); 
  EXPECT_FALSE(std::isnan(result));
  result = cauchy_log(y, -1.0, sigma, errno_policy()); 
  EXPECT_FALSE(std::isnan(result));
    

  result = cauchy_log(nan, mu, sigma, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));

  result = cauchy_log(y, -inf, sigma, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = cauchy_log(y, nan, sigma, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = cauchy_log(y, inf, sigma, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));

  result = cauchy_log(y, mu, 0.0, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = cauchy_log(y, mu, -1.0, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = cauchy_log(y, mu, -inf, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = cauchy_log(y, mu, nan, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
  result = cauchy_log(y, mu, inf, errno_policy()); 
  EXPECT_TRUE(std::isnan(result));
}
TEST(ProbDistributionsCauchy,Cumulative) {
  using stan::prob::cauchy_p;
  EXPECT_FLOAT_EQ(0.75, cauchy_p(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.187167, cauchy_p(-1.5, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.187167, cauchy_p(-2.5, -1.0, 1.0));
}
