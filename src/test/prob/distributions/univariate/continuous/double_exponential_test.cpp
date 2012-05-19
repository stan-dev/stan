#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/double_exponential.hpp>

TEST(ProbDistributionsDoubleExponential,DoubleExponential) {
  EXPECT_FLOAT_EQ(-0.6931472, stan::prob::double_exponential_log(1.0,1.0,1.0));
  EXPECT_FLOAT_EQ(-1.693147, stan::prob::double_exponential_log(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(-5.693147, stan::prob::double_exponential_log(-3.0,2.0,1.0));
  EXPECT_FLOAT_EQ(-1.886294, stan::prob::double_exponential_log(1.0,0.0,2.0));
  EXPECT_FLOAT_EQ(-0.8, stan::prob::double_exponential_log(1.9,2.3,0.5));
  EXPECT_FLOAT_EQ(-0.9068528, stan::prob::double_exponential_log(1.9,2.3,0.25));
}
TEST(ProbDistributionsDoubleExponential,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::double_exponential_log<true>(1.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::double_exponential_log<true>(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::double_exponential_log<true>(-3.0,2.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::double_exponential_log<true>(1.0,0.0,2.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::double_exponential_log<true>(1.9,2.3,0.5));
  EXPECT_FLOAT_EQ(0.0, stan::prob::double_exponential_log<true>(1.9,2.3,0.25));
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

using stan::prob::double_exponential_log;

TEST(ProbDistributionsDoubleExponential,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  double y = 5.5;
  double mu = 5.0;
  double sigma = 3.0;
  
  EXPECT_NO_THROW(double_exponential_log(y, mu, sigma));
  EXPECT_NO_THROW(double_exponential_log(-y, mu, sigma));
  EXPECT_NO_THROW(double_exponential_log(y, -mu, sigma));
  
  EXPECT_THROW(double_exponential_log(nan, mu, sigma), std::domain_error);
  EXPECT_THROW(double_exponential_log(-inf, mu, sigma), std::domain_error);
  EXPECT_THROW(double_exponential_log(inf, mu, sigma), std::domain_error);
  
  EXPECT_THROW(double_exponential_log(y, nan, sigma), std::domain_error);
  EXPECT_THROW(double_exponential_log(y, -inf, sigma), std::domain_error);
  EXPECT_THROW(double_exponential_log(y, inf, sigma), std::domain_error);
  
  EXPECT_THROW(double_exponential_log(y, mu, nan), std::domain_error);
  EXPECT_THROW(double_exponential_log(y, mu, 0.0), std::domain_error);
  EXPECT_THROW(double_exponential_log(y, mu, -1.0), std::domain_error);
  EXPECT_THROW(double_exponential_log(y, mu, -inf), std::domain_error);
  EXPECT_THROW(double_exponential_log(y, mu, inf), std::domain_error);
}
TEST(ProbDistributionsDoubleExponential,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  double result;
  double y = 5.5;
  double mu = 5.0;
  double sigma = 3.0;
  
  result = double_exponential_log(y, mu, sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = double_exponential_log(-y, mu, sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = double_exponential_log(y, -mu, sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
    
  result = double_exponential_log(nan, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = double_exponential_log(-inf, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = double_exponential_log(inf, mu, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = double_exponential_log(y, nan, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = double_exponential_log(y, -inf, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = double_exponential_log(y, inf, sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = double_exponential_log(y, mu, nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = double_exponential_log(y, mu, 0.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = double_exponential_log(y, mu, -1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = double_exponential_log(y, mu, -inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = double_exponential_log(y, mu, inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
TEST(ProbDistributionsDoubleExponential,Cumulative) {
  EXPECT_FLOAT_EQ(0.5, stan::prob::double_exponential_p(1.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.8160603, stan::prob::double_exponential_p(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.003368973, stan::prob::double_exponential_p(-3.0,2.0,1.0));
  EXPECT_FLOAT_EQ(0.6967347, stan::prob::double_exponential_p(1.0,0.0,2.0));
  EXPECT_FLOAT_EQ(0.2246645, stan::prob::double_exponential_p(1.9,2.3,0.5));
  EXPECT_FLOAT_EQ(0.10094826, stan::prob::double_exponential_p(1.9,2.3,0.25));
}
