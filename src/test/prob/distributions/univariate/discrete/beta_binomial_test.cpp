#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/discrete/beta_binomial.hpp>

TEST(ProbDistributionsBetaBinomial,BetaBinomial) {
  EXPECT_FLOAT_EQ(-1.854007, stan::prob::beta_binomial_log(5,20,10.0,25.0));
  EXPECT_FLOAT_EQ(-4.376696, stan::prob::beta_binomial_log(25,100,30.0,50.0));
}
TEST(ProbDistributionsBetaBinomial,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::beta_binomial_log<true>(5,20,10.0,25.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::beta_binomial_log<true>(25,100,30.0,50.0));
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

using stan::prob::beta_binomial_log;

TEST(ProbDistributionsBetaBinomial,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  int n = 5;
  int N = 15;
  double alpha = 3.0;
  double beta = 4.5;

  EXPECT_NO_THROW(beta_binomial_log(n,N,alpha,beta));
  EXPECT_NO_THROW(beta_binomial_log(n,0,alpha,beta));
  
  EXPECT_THROW(beta_binomial_log(n,-1,alpha,beta), std::domain_error);
  
  EXPECT_THROW(beta_binomial_log(n,N,nan,beta), std::domain_error);
  EXPECT_THROW(beta_binomial_log(n,N,0.0,beta), std::domain_error);
  EXPECT_THROW(beta_binomial_log(n,N,-1.0,beta), std::domain_error);
  EXPECT_THROW(beta_binomial_log(n,N,-inf,beta), std::domain_error);
  EXPECT_THROW(beta_binomial_log(n,N,inf,beta), std::domain_error);
  
  EXPECT_THROW(beta_binomial_log(n,N,alpha,nan), std::domain_error);
  EXPECT_THROW(beta_binomial_log(n,N,alpha,0.0), std::domain_error);
  EXPECT_THROW(beta_binomial_log(n,N,alpha,-1.0), std::domain_error);
  EXPECT_THROW(beta_binomial_log(n,N,alpha,-inf), std::domain_error);
  EXPECT_THROW(beta_binomial_log(n,N,alpha,inf), std::domain_error);
}
TEST(ProbDistributionsBetaBinomial,ErrnoPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  double result;
  int n = 5;
  int N = 15;
  double alpha = 3.0;
  double beta = 4.5;

  result = beta_binomial_log(n,N,alpha,beta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = beta_binomial_log(n,0,alpha,beta, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  result = beta_binomial_log(n,-1,alpha,beta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = beta_binomial_log(n,N,nan,beta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = beta_binomial_log(n,N,0.0,beta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = beta_binomial_log(n,N,-1.0,beta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = beta_binomial_log(n,N,-inf,beta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = beta_binomial_log(n,N,inf,beta, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  
  result = beta_binomial_log(n,N,alpha,nan, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = beta_binomial_log(n,N,alpha,0.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = beta_binomial_log(n,N,alpha,-1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = beta_binomial_log(n,N,alpha,-inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = beta_binomial_log(n,N,alpha,inf, errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
