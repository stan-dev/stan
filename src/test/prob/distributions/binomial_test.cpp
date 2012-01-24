#include <gtest/gtest.h>
#include <stan/prob/distributions/binomial.hpp>

TEST(ProbDistributionsBinomial,Binomial) {
  EXPECT_FLOAT_EQ(-2.144372, stan::prob::binomial_log(10,20,0.4));
  EXPECT_FLOAT_EQ(-16.09438, stan::prob::binomial_log(0,10,0.8));
}
TEST(ProbDistributionsBinomial,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::binomial_log<true>(10,20,0.4));
  EXPECT_FLOAT_EQ(0.0, stan::prob::binomial_log<true>(0,10,0.8));
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

using stan::prob::binomial_log;

TEST(ProbDistributionsBinomial,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  int n = 4;
  int N = 10;
  double prob = 0.5;

  EXPECT_NO_THROW(binomial_log(n,N,prob));
  EXPECT_NO_THROW(binomial_log(0,N,prob));
  EXPECT_NO_THROW(binomial_log(n,N,0.0));
  EXPECT_NO_THROW(binomial_log(n,N,1.0));
  EXPECT_NO_THROW(binomial_log(N,N,prob));

  EXPECT_THROW(binomial_log(nan,N,prob), std::domain_error);
  EXPECT_THROW(binomial_log(inf,N,prob), std::domain_error);
  EXPECT_THROW(binomial_log(-inf,N,prob), std::domain_error);
  EXPECT_THROW(binomial_log(-1,N,prob), std::domain_error);
  
  EXPECT_THROW(binomial_log(n,nan,prob), std::domain_error);
  EXPECT_THROW(binomial_log(n,inf,prob), std::domain_error);
  EXPECT_THROW(binomial_log(n,-inf,prob), std::domain_error);
  EXPECT_THROW(binomial_log(n,3,prob), std::domain_error);
  
  EXPECT_THROW(binomial_log(n,N,nan), std::domain_error);
  EXPECT_THROW(binomial_log(n,N,-inf), std::domain_error);
  EXPECT_THROW(binomial_log(n,N,inf), std::domain_error);
  EXPECT_THROW(binomial_log(n,N,-0.1), std::domain_error);
  EXPECT_THROW(binomial_log(n,N,1.1), std::domain_error);
}
TEST(ProbDistributionsBinomial,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  double result;
  int n = 4;
  int N = 10;
  double prob = 0.5;

  result = binomial_log(n,N,prob, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = binomial_log(0,N,prob, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = binomial_log(n,N,0.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = binomial_log(n,N,1.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = binomial_log(N,N,prob, errno_policy());
  EXPECT_FALSE(std::isnan(result));

  result = binomial_log(nan,N,prob, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = binomial_log(inf,N,prob, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = binomial_log(-inf,N,prob, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = binomial_log(-1,N,prob, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = binomial_log(n,nan,prob, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = binomial_log(n,inf,prob, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = binomial_log(n,-inf,prob, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = binomial_log(n,3,prob, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = binomial_log(n,N,nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = binomial_log(n,N,-inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = binomial_log(n,N,inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = binomial_log(n,N,-0.1, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = binomial_log(n,N,1.1, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
