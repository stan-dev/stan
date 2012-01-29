#include <gtest/gtest.h>
#include "stan/prob/distributions/univariate/continuous/inv_gamma.hpp"

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

TEST(ProbDistributions,InvGamma) {
  EXPECT_FLOAT_EQ(-1, stan::prob::inv_gamma_log(1,1,1.0));
  EXPECT_FLOAT_EQ(-0.8185295, stan::prob::inv_gamma_log(0.5,2.9,3.1));
}
TEST(ProbDistributions,InvGammaPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::inv_gamma_log<true>(1,1,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::inv_gamma_log<true>(0.5,2.9,3.1));
}
TEST(ProbDistributions,InvGammaDefaultPolicy) {
  double y = 0.5;
  double alpha = 1.0;
  double beta = 2.0;
  
  EXPECT_NO_THROW(stan::prob::inv_gamma_log(y, alpha, beta));
  EXPECT_THROW(stan::prob::inv_gamma_log(0.0, alpha, beta), std::domain_error)
    << "exception expected when y = 0.0";
  EXPECT_THROW(stan::prob::inv_gamma_log(-1, alpha, beta), std::domain_error)
    << "exception expected when y < 0.";
  EXPECT_THROW(stan::prob::inv_gamma_log(y, 0.0, beta), std::domain_error)
    << "exception expected when alpha = 0.0";
  EXPECT_THROW(stan::prob::inv_gamma_log(y, -1.0, beta), std::domain_error)
    << "exception expected when alpha < 0.";
  EXPECT_THROW(stan::prob::inv_gamma_log(y, alpha, 0.0), std::domain_error)
    << "exception expected when beta = 0.0";
  EXPECT_THROW(stan::prob::inv_gamma_log(y, alpha, -1.0), std::domain_error)
    << "exception expected when beta < 0.";
}
TEST(ProbDistributions,InvGammaErrnoPolicy) {
  double y = 0.5;
  double alpha = 1.0;
  double beta = 2.0;
  double result;
  
  EXPECT_NO_THROW(result = stan::prob::inv_gamma_log(y, alpha, beta, errno_policy()));
  EXPECT_FALSE(std::isnan(result)) << "this should work fine";

  EXPECT_NO_THROW(result = stan::prob::inv_gamma_log(0.0, alpha, beta, errno_policy()));
  EXPECT_TRUE(std::isnan(result))
    << "exception expected when y = 0.0";
  
  EXPECT_NO_THROW(result = stan::prob::inv_gamma_log(-1.0, alpha, beta, errno_policy()));
  EXPECT_TRUE(std::isnan(result))
    << "exception expected when y < 0.";

  EXPECT_NO_THROW(result = stan::prob::inv_gamma_log(y, 0.0, beta, errno_policy()));
  EXPECT_TRUE(std::isnan(result))
    << "exception expected when alpha = 0.0";

  EXPECT_NO_THROW(result = stan::prob::inv_gamma_log(y, -1.0, beta, errno_policy()));
  EXPECT_TRUE(std::isnan(result))
    << "exception expected when alpha < 0.";

  EXPECT_NO_THROW(result = stan::prob::inv_gamma_log(y, alpha, 0.0, errno_policy()));
  EXPECT_TRUE(std::isnan(result))
    << "exception expected when beta = 0.0";

  EXPECT_NO_THROW(result = stan::prob::inv_gamma_log(y, alpha, -1.0, errno_policy()));
  EXPECT_TRUE(std::isnan(result))
    << "exception expected when beta < 0.";
}
