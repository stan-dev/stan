#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/logistic.hpp>

TEST(ProbDistributionsLogistic,Logistic) {
  EXPECT_FLOAT_EQ(-2.129645, stan::prob::logistic_log(1.2,0.3,2.0));
  EXPECT_FLOAT_EQ(-3.430098, stan::prob::logistic_log(-1.0,0.2,0.25));
}
TEST(ProbDistributionsLogistic,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::logistic_log<true>(1.2,0.3,2.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::logistic_log<true>(-1.0,0.2,0.25));
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

using stan::prob::logistic_log;

TEST(ProbDistributionsLogistic,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double y = 1.2;
  double mu = 0.4;
  double sigma = 2.0;
  
  EXPECT_NO_THROW(logistic_log(y, mu, sigma));

  EXPECT_THROW(logistic_log(nan, mu, sigma), std::domain_error);
  EXPECT_THROW(logistic_log(-inf, mu, sigma), std::domain_error);
  EXPECT_THROW(logistic_log(inf, mu, sigma), std::domain_error);
  
  EXPECT_THROW(logistic_log(y, nan, sigma), std::domain_error);
  EXPECT_THROW(logistic_log(y, -inf, sigma), std::domain_error);
  EXPECT_THROW(logistic_log(y, inf, sigma), std::domain_error);
  
  EXPECT_THROW(logistic_log(y, mu, nan), std::domain_error);
  EXPECT_THROW(logistic_log(y, mu, 0.0), std::domain_error);
  EXPECT_THROW(logistic_log(y, mu, -1.0), std::domain_error);
  EXPECT_THROW(logistic_log(y, mu, -inf), std::domain_error);
  EXPECT_THROW(logistic_log(y, mu, inf), std::domain_error);
}
TEST(ProbDistributionsLogistic,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double result;
  double y = 1.2;
  double mu = 0.4;
  double sigma = 2.0;
  
  result = logistic_log(y, mu, sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));

  result = logistic_log(nan, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = logistic_log(-inf, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = logistic_log(inf, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = logistic_log(y, nan, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = logistic_log(y, -inf, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = logistic_log(y, inf, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = logistic_log(y, mu, nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = logistic_log(y, mu, 0.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = logistic_log(y, mu, -1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = logistic_log(y, mu, -inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = logistic_log(y, mu, inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
