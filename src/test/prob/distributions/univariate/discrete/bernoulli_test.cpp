#include <gtest/gtest.h>
#include "stan/prob/distributions/univariate/discrete/bernoulli.hpp"

TEST(ProbDistributionsBernoulli,Bernoulli) {
  EXPECT_FLOAT_EQ(std::log(0.25), stan::prob::bernoulli_log(1,0.25));
  EXPECT_FLOAT_EQ(std::log(1.0 - 0.25), stan::prob::bernoulli_log(0,0.25));
}
TEST(ProbDistributionsBernoulli,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::bernoulli_log<true>(1,0.25));
  EXPECT_FLOAT_EQ(0.0, stan::prob::bernoulli_log<true>(0,0.25));
}

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

using stan::prob::bernoulli_log;

TEST(ProbDistributionsBernoulli,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  unsigned int k = 1;
  double theta = 0.75;

  EXPECT_NO_THROW(bernoulli_log(k, theta));
  EXPECT_NO_THROW(bernoulli_log(k, 0.0));
  EXPECT_NO_THROW(bernoulli_log(k, 1.0));
    
  EXPECT_THROW(bernoulli_log(2U, theta), std::domain_error);
  EXPECT_THROW(bernoulli_log(k, nan), std::domain_error);
  EXPECT_THROW(bernoulli_log(k, inf), std::domain_error);
  EXPECT_THROW(bernoulli_log(k, -inf), std::domain_error);
  EXPECT_THROW(bernoulli_log(k, -1.0), std::domain_error);
  EXPECT_THROW(bernoulli_log(k, 2.0), std::domain_error);
}
TEST(ProbDistributionsBernoulli,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double result;
  unsigned int k = 1;
  double theta = 0.75;

  result = bernoulli_log(k, theta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = bernoulli_log(k, 0.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = bernoulli_log(k, 1.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
    
  result = bernoulli_log(2U, theta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_log(k, nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_log(k, inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_log(k, -inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_log(k, -1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = bernoulli_log(k, 2.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
