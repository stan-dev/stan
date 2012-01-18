#include <gtest/gtest.h>
#include "stan/prob/distributions/inv_chi_square.hpp"

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


TEST(ProbDistributionsInvChiSquare,InvChiSquare) {
  EXPECT_FLOAT_EQ(-0.3068528, stan::prob::inv_chi_square_log(0.5,2.0));
  EXPECT_FLOAT_EQ(-12.28905, stan::prob::inv_chi_square_log(3.2,9.1));
}
TEST(ProbDistributionsInvChiSquare,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::inv_chi_square_log<true>(0.5,2.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::inv_chi_square_log<true>(3.2,9.1));
}
TEST(ProbDistributionsInvChiSquare,DefaultPolicy) {
  double y = 0.0;
  double nu = 0.0;
  
  EXPECT_NO_THROW(stan::prob::inv_chi_square_log(1.0, 1.0));
  EXPECT_NO_THROW(stan::prob::inv_chi_square_log(-1.0, 1.0));
  EXPECT_NO_THROW(stan::prob::inv_chi_square_log(std::numeric_limits<double>::infinity(), 1.0));
  EXPECT_NO_THROW(stan::prob::inv_chi_square_log(-std::numeric_limits<double>::infinity(), 1.0));

  EXPECT_THROW(stan::prob::inv_chi_square_log(y, nu), std::domain_error);
  EXPECT_THROW(stan::prob::inv_chi_square_log(y, -1.0), std::domain_error);
  EXPECT_THROW(stan::prob::inv_chi_square_log(std::numeric_limits<double>::quiet_NaN(), 1.0), 
               std::domain_error);
  EXPECT_THROW(stan::prob::inv_chi_square_log(y, std::numeric_limits<double>::quiet_NaN()),
               std::domain_error);
  EXPECT_THROW(stan::prob::inv_chi_square_log(y, std::numeric_limits<double>::infinity()),
               std::domain_error);
  EXPECT_THROW(stan::prob::inv_chi_square_log(y, -std::numeric_limits<double>::infinity()),
               std::domain_error);
}
TEST(ProbDistributionsInvChiSquare,ErrnoPolicy) {
  double y = 0.0;
  double nu = 0.0;
  double result;
  
  result = stan::prob::inv_chi_square_log(1.0, 1.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = stan::prob::inv_chi_square_log(-1.0, 1.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = stan::prob::inv_chi_square_log(std::numeric_limits<double>::infinity(), 1.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = stan::prob::inv_chi_square_log(-std::numeric_limits<double>::infinity(), 1.0, errno_policy());
  EXPECT_FALSE(std::isnan(result));

  result = stan::prob::inv_chi_square_log(y, nu, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = stan::prob::inv_chi_square_log(y, -1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = stan::prob::inv_chi_square_log(std::numeric_limits<double>::quiet_NaN(), 1.0, errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = stan::prob::inv_chi_square_log(y, std::numeric_limits<double>::quiet_NaN(), errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = stan::prob::inv_chi_square_log(y, std::numeric_limits<double>::infinity(), errno_policy());
  EXPECT_TRUE(std::isnan(result));
  result = stan::prob::inv_chi_square_log(y, -std::numeric_limits<double>::infinity(), errno_policy());
  EXPECT_TRUE(std::isnan(result));
}
