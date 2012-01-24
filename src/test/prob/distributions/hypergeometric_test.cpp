#include <gtest/gtest.h>
#include <stan/prob/distributions/hypergeometric.hpp>

TEST(ProbDistributionsHypergeometric,Hypergeometric) {
  EXPECT_FLOAT_EQ(-4.119424, stan::prob::hypergeometric_log(5,15,10,10));
  EXPECT_FLOAT_EQ(-2.302585, stan::prob::hypergeometric_log(0,2,3,2));
}
TEST(ProbDistributionsHypergeometric,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::hypergeometric_log<true>(5,15,10,10));
  EXPECT_FLOAT_EQ(0.0, stan::prob::hypergeometric_log<true>(0,2,3,2));
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

using stan::prob::hypergeometric_log;

TEST(ProbDistributionsHypergeometric,DefaultPolicy) {
  int n = 2;
  int N = 10;
  int a = 5;
  int b = 11;
    
  EXPECT_NO_THROW(hypergeometric_log(n,N,a,b));

  EXPECT_THROW(hypergeometric_log(6,N,a,b), std::domain_error) << "n > a";
  EXPECT_THROW(hypergeometric_log(n,N,a,7), std::domain_error) << "N-n > b";
  EXPECT_THROW(hypergeometric_log(n,17,a,b), std::domain_error) << "N > a+b";
}
TEST(ProbDistributionsHypergeometric,ErrnoPolicy) {
  int n = 2;
  int N = 10;
  int a = 5;
  int b = 11;
  double result;
  
  result = hypergeometric_log(n,N,a,b, errno_policy());
  EXPECT_FALSE(std::isnan(result));

  result = hypergeometric_log(6,N,a,b, errno_policy());
  EXPECT_TRUE(std::isnan(result)) << "n > a";
  result = hypergeometric_log(n,N,a,7, errno_policy());
  EXPECT_TRUE(std::isnan(result)) << "N-n > b";
  result = hypergeometric_log(n,17,a,b, errno_policy());
  EXPECT_TRUE(std::isnan(result)) << "N > a+b";
}
