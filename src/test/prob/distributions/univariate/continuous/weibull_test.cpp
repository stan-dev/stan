#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/weibull.hpp>

TEST(ProbDistributionsWeibull,Weibull) {
  EXPECT_FLOAT_EQ(-2.0, stan::prob::weibull_log(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(-3.277094, stan::prob::weibull_log(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(-102.8962, stan::prob::weibull_log(3.9,1.7,0.25));
}
TEST(ProbDistributionsWeibull,WeibullPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::weibull_log<true>(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::weibull_log<true>(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(0.0, stan::prob::weibull_log<true>(3.9,1.7,0.25));
}

TEST(ProbDistributionsWeibull,Cumulative) {
  using stan::prob::weibull_cdf;
  using std::numeric_limits;
  EXPECT_FLOAT_EQ(0.86466472, weibull_cdf(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.0032585711, weibull_cdf(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(1.0, weibull_cdf(3.9,1.7,0.25));

  // ??
  EXPECT_FLOAT_EQ(0.0,
                  weibull_cdf(-numeric_limits<double>::infinity(),
                              1.0,1.0));
  EXPECT_FLOAT_EQ(0.0, weibull_cdf(0.0,1.0,1.0));
  EXPECT_FLOAT_EQ(1.0, weibull_cdf(numeric_limits<double>::infinity(),
                                   1.0,1.0));
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

using stan::prob::weibull_log;

TEST(ProbDistributionsWeibull,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  double y = 1.0;
  double alpha = 2.0;
  double sigma = 3.0;

  EXPECT_NO_THROW(weibull_log(y,alpha,sigma));
  
  EXPECT_THROW(weibull_log(nan,alpha,sigma), std::domain_error);
  EXPECT_THROW(weibull_log(inf,alpha,sigma), std::domain_error);
  EXPECT_THROW(weibull_log(-inf,alpha,sigma), std::domain_error);

  EXPECT_THROW(weibull_log(y,nan,sigma), std::domain_error);
  EXPECT_THROW(weibull_log(y,0.0,sigma), std::domain_error);
  EXPECT_THROW(weibull_log(y,-inf,sigma), std::domain_error);
  EXPECT_THROW(weibull_log(y,inf,sigma), std::domain_error);
  
  EXPECT_THROW(weibull_log(y,alpha,nan), std::domain_error);
  EXPECT_THROW(weibull_log(y,alpha,0.0), std::domain_error);
  EXPECT_THROW(weibull_log(y,alpha,-inf), std::domain_error);
  EXPECT_NO_THROW(weibull_log(y,alpha,inf));
}
TEST(ProbDistributionsWeibull,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  double result;
  double y = 1.0;
  double alpha = 2.0;
  double sigma = 3.0;

  result = weibull_log(y,alpha,sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  result = weibull_log(nan,alpha,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = weibull_log(inf,alpha,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = weibull_log(-inf,alpha,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));

  result = weibull_log(y,nan,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = weibull_log(y,0.0,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = weibull_log(y,-inf,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = weibull_log(y,inf,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = weibull_log(y,alpha,nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = weibull_log(y,alpha,0.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = weibull_log(y,alpha,-inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = weibull_log(y,alpha,inf, errno_policy());
  EXPECT_FALSE(std::isnan(result));
}
