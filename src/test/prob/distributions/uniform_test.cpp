#include <gtest/gtest.h>
#include <stan/prob/distributions/uniform.hpp>

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


TEST(ProbDistributionsUniform,Uniform) {
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log(0.2,0.0,1.0)));
  EXPECT_FLOAT_EQ(2.0, exp(stan::prob::uniform_log(0.2,-0.25,0.25)));
  EXPECT_FLOAT_EQ(0.1, exp(stan::prob::uniform_log(101.0,100.0,110.0)));
  // lower boundary
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log(0.0,0.0,1.0)));
  EXPECT_FLOAT_EQ(0.0, exp(stan::prob::uniform_log(-1.0,0.0,1.0)));
  // upper boundary
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log(1.0,0.0,1.0)));
  EXPECT_FLOAT_EQ(0.0, exp(stan::prob::uniform_log(2.0,0.0,1.0)));  
}
TEST(ProbDistributionsUniform,Propto) {
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log<true>(0.2,0.0,1.0)));
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log<true>(0.2,-0.25,0.25)));
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log<true>(101.0,100.0,110.0)));
  // lower boundary
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log<true>(0.0,0.0,1.0)));
  EXPECT_FLOAT_EQ(0.0, exp(stan::prob::uniform_log<true>(-1.0,0.0,1.0)));
  // upper boundary
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log<true>(1.0,0.0,1.0)));
  EXPECT_FLOAT_EQ(0.0, exp(stan::prob::uniform_log<true>(2.0,0.0,1.0)));  
}
TEST(ProbDistributionsUniform,DefaultPolicyY) {
  double y = 0.0;
  EXPECT_NO_THROW(stan::prob::uniform_log(y,0.0,1.0));

  y = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::uniform_log(y,0.0,0.0), std::domain_error);
  y = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::uniform_log(y,0.0,0.0), std::domain_error);
  y = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::uniform_log(y,0.0,0.0), std::domain_error);
}
TEST(ProbDistributionsUniform,DefaultPolicyLower) {
  double lb = 0.0;
  EXPECT_NO_THROW(stan::prob::uniform_log(0.0,lb,1.0));

  lb = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::uniform_log(0.0,lb,0.0), std::domain_error);
  lb = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::uniform_log(0.0,lb,0.0), std::domain_error);
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::uniform_log(0.0,lb,0.0), std::domain_error);
}
TEST(ProbDistributionsUniform,DefaultPolicyUpper) {
  double ub = 10.0;
  EXPECT_NO_THROW(stan::prob::uniform_log(0.0,0.0,ub));

  ub = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::prob::uniform_log(0.0,0.0,ub), std::domain_error);
  ub = std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::uniform_log(0.0,0.0,ub), std::domain_error);
  ub = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(stan::prob::uniform_log(0.0,0.0,ub), std::domain_error);
}
TEST(ProbDistributionsUniform,DefaultPolicyBounds) {
  // lower bound higher than the upper bound
  EXPECT_THROW(stan::prob::uniform_log(0.0,1.0,0.0), std::domain_error);
  // lower and upper boundary the same 
  EXPECT_THROW(stan::prob::uniform_log(0.0, 0.0, 0.0), std::domain_error);
}
TEST(ProbDistributionsUniform,ErrnoPolicyBounds) {
  double y = 0;
  double result;
  // lower and uppper bounds same
  EXPECT_NO_THROW(result = stan::prob::uniform_log(y, 0.0, 0.0, errno_policy()));
  EXPECT_EQ(33, errno);
}
TEST(ProbDistributionsUniform,ErrnoPolicyY) {
  double y = 0.0;
  double result;
  result = stan::prob::uniform_log(y,0.0,1.0,errno_policy());
  EXPECT_FALSE(std::isnan(result));

  y = std::numeric_limits<double>::quiet_NaN();
  result = stan::prob::uniform_log(y,0.0,1.0,errno_policy());
  EXPECT_TRUE(std::isnan(result));

  y = std::numeric_limits<double>::infinity();
  result = stan::prob::uniform_log(y,0.0,1.0,errno_policy());
  EXPECT_TRUE(std::isnan(result));

  y = -std::numeric_limits<double>::infinity();
  result = stan::prob::uniform_log(y,0.0,1.0,errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
TEST(ProbDistributionsUniform,ErrnoPolicyLower) {
  double lb = 0.0;
  double result;
  result = stan::prob::uniform_log(0.0,lb,1.0,errno_policy());
  EXPECT_FALSE(std::isnan(result));
 
  lb = std::numeric_limits<double>::quiet_NaN();
  result = stan::prob::uniform_log(0.0,lb,0.0,errno_policy());
  EXPECT_TRUE(std::isnan(result));
  lb = std::numeric_limits<double>::infinity();
  result = stan::prob::uniform_log(0.0,lb,0.0,errno_policy());
  EXPECT_TRUE(std::isnan(result));
  lb = -std::numeric_limits<double>::infinity();
  result = stan::prob::uniform_log(0.0,lb,0.0,errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
TEST(ProbDistributionsUniform,ErrnoPolicyUpper) {
  double ub = 10.0;
  double result;
  result = stan::prob::uniform_log(0.0,0.0,ub,errno_policy());
  EXPECT_FALSE(std::isnan(result));
 
  ub = std::numeric_limits<double>::quiet_NaN();
  result = stan::prob::uniform_log(0.0,0.0,ub,errno_policy());
  EXPECT_TRUE(std::isnan(result));
  ub = std::numeric_limits<double>::infinity();
  result = stan::prob::uniform_log(0.0,0.0,ub,errno_policy());
  EXPECT_TRUE(std::isnan(result));
  ub = -std::numeric_limits<double>::infinity();
  result = stan::prob::uniform_log(0.0,0.0,ub,errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
