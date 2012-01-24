#include <gtest/gtest.h>
#include <stan/prob/distributions/poisson.hpp>

TEST(ProbDistributionsPoisson,Poisson) {
  EXPECT_FLOAT_EQ(-2.900934, stan::prob::poisson_log(17,13.0));
  EXPECT_FLOAT_EQ(-145.3547, stan::prob::poisson_log(192,42.0));
  EXPECT_FLOAT_EQ(-3.0, stan::prob::poisson_log(0, 3.0));
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::poisson_log (0, 0.0));
}
TEST(ProbDistributionsPoisson,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::poisson_log<true>(17,13.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::poisson_log<true>(192,42.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::poisson_log<true>(0, 3.0));
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::poisson_log<true>(0, 0.0));
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

using stan::prob::poisson_log;

TEST(ProbDistributionsPossion,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  unsigned int n = 4;
  double lambda = 3.0;
  
  EXPECT_NO_THROW(poisson_log(n, lambda));
  EXPECT_NO_THROW(poisson_log(n, 0.0));
  
  EXPECT_THROW(poisson_log(n, nan), std::domain_error);
  EXPECT_THROW(poisson_log(n, -1.0), std::domain_error);
  EXPECT_THROW(poisson_log(n, -inf), std::domain_error);
  EXPECT_THROW(poisson_log(n, inf), std::domain_error);
}
TEST(ProbDistributionsPossion,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  double result;
  unsigned int n = 4;
  double lambda = 3.0;
  
  result = poisson_log(n, lambda, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = poisson_log(n, 0.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  result = poisson_log(n, nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = poisson_log(n, -1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = poisson_log(n, -inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = poisson_log(n, inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
