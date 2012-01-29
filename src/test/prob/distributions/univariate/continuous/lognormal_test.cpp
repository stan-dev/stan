#include <gtest/gtest.h>
#include "stan/prob/distributions/univariate/continuous/lognormal.hpp"

TEST(ProbDistributionsLognormal,Lognormal) {
  EXPECT_FLOAT_EQ(-1.509803, stan::prob::lognormal_log(1.2,0.3,1.5));
  EXPECT_FLOAT_EQ(-3.462263, stan::prob::lognormal_log(12.0,3.0,0.9));
}
TEST(ProbDistributionsLognormal,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::lognormal_log<true>(1.2,0.3,1.5));
  EXPECT_FLOAT_EQ(0.0, stan::prob::lognormal_log<true>(12.0,3.0,0.9));
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

using stan::prob::lognormal_log;

TEST(ProbDistributionsLognormal,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double y = 1.0;
  double mu = 2.0;
  double sigma = 3.0;

  EXPECT_NO_THROW(lognormal_log(y,mu,sigma));
  EXPECT_NO_THROW(lognormal_log(inf,mu,sigma));
  EXPECT_NO_THROW(lognormal_log(-inf,mu,sigma));

  EXPECT_THROW(lognormal_log(nan,mu,sigma), std::domain_error);
  
  EXPECT_THROW(lognormal_log(y,nan,sigma), std::domain_error);
  EXPECT_THROW(lognormal_log(y,-inf,sigma), std::domain_error);
  EXPECT_THROW(lognormal_log(y,inf,sigma), std::domain_error);

  EXPECT_THROW(lognormal_log(y,mu,nan), std::domain_error);
  EXPECT_THROW(lognormal_log(y,mu,0.0), std::domain_error);
  EXPECT_THROW(lognormal_log(y,mu,-1.0), std::domain_error);
  EXPECT_THROW(lognormal_log(y,mu,-inf), std::domain_error);
  EXPECT_THROW(lognormal_log(y,mu,inf), std::domain_error);
}
TEST(ProbDistributionsLognormal,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  double result;
  double y = 1.0;
  double mu = 2.0;
  double sigma = 3.0;
  
  result = lognormal_log(y,mu,sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = lognormal_log(inf,mu,sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = lognormal_log(-inf,mu,sigma, errno_policy());
  EXPECT_FALSE(std::isnan(result));

  result = lognormal_log(nan,mu,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = lognormal_log(y,nan,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = lognormal_log(y,-inf,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = lognormal_log(y,inf,sigma, errno_policy());
  EXPECT_TRUE(std::isnan(result));

  result = lognormal_log(y,mu,nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = lognormal_log(y,mu,0.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = lognormal_log(y,mu,-1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = lognormal_log(y,mu,-inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = lognormal_log(y,mu,inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
