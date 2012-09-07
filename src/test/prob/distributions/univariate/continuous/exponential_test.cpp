#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/exponential.hpp>

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

using stan::prob::exponential_log;

TEST(ProbDistributionsExponential,Exponential) {
  EXPECT_FLOAT_EQ(-2.594535, exponential_log(2.0,1.5));
  EXPECT_FLOAT_EQ(-57.13902, exponential_log(15.0,3.9));
}
TEST(ProbDistributionsExponential,Propto) {
  EXPECT_FLOAT_EQ(0.0, exponential_log<true>(2.0,1.5));
  EXPECT_FLOAT_EQ(0.0, exponential_log<true>(15.0,3.9));
}
TEST(ProbDistributionsExponential,DefaultPolicy){
  double inf = std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  double y_valid = 5.0;
  double beta_valid = 1.0;

  EXPECT_NO_THROW(exponential_log(y_valid, beta_valid));
  
  EXPECT_NO_THROW(exponential_log(0.0, beta_valid));
  EXPECT_NO_THROW(exponential_log(-1.0, beta_valid));
  EXPECT_NO_THROW(exponential_log(-inf, beta_valid));
  EXPECT_NO_THROW(exponential_log(inf, beta_valid));
  
  EXPECT_THROW(exponential_log(nan, beta_valid), std::domain_error);
  EXPECT_THROW(exponential_log(y_valid, 0.0), std::domain_error);
  EXPECT_THROW(exponential_log(y_valid, -1.0), std::domain_error);
  EXPECT_THROW(exponential_log(y_valid, inf), std::domain_error);
  EXPECT_THROW(exponential_log(y_valid, -inf), std::domain_error);
  EXPECT_THROW(exponential_log(y_valid, nan), std::domain_error);
}
TEST(ProbDistributionsExponential,ErrnoPolicy){
  double inf = std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  double result;
  double y_valid = 5.0;
  double beta_valid = 1.0;

  result = exponential_log(y_valid, beta_valid, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  result = exponential_log(0.0, beta_valid, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = exponential_log(-1.0, beta_valid, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = exponential_log(-inf, beta_valid, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = exponential_log(inf, beta_valid, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  result = exponential_log(nan, beta_valid, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = exponential_log(y_valid, 0.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = exponential_log(y_valid, -1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = exponential_log(y_valid, inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = exponential_log(y_valid, -inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = exponential_log(y_valid, nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}

TEST(ProbDistributionsExponential,Cumulative) {
  EXPECT_FLOAT_EQ(0.95021293, stan::prob::exponential_cdf(2.0,1.5));
  EXPECT_FLOAT_EQ(0.0, stan::prob::exponential_cdf(0,1.5));
  EXPECT_FLOAT_EQ(1.0, stan::prob::exponential_cdf(15.0,3.9));
  EXPECT_FLOAT_EQ(0.62280765, stan::prob::exponential_cdf(0.25,3.9));
}
