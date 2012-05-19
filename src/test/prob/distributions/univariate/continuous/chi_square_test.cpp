#include <gtest/gtest.h>
#include "stan/prob/distributions/univariate/continuous/chi_square.hpp"

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


TEST(ProbDistributionsChiSquare,ChiSquare) {
  EXPECT_FLOAT_EQ(-3.835507, stan::prob::chi_square_log(7.9,3.0));
  EXPECT_FLOAT_EQ(-2.8927, stan::prob::chi_square_log(1.9,0.5));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  stan::prob::chi_square_log(-1.0,0.5));
}
TEST(ProbDistributionsChiSquare,Propto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::chi_square_log<true>(7.9,3.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::chi_square_log<true>(1.9,0.5));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  stan::prob::chi_square_log<true>(-1.0,0.5));
}
TEST(ProbDistributionsChiSquare,DefaultPolicy) {
  double y = 0.0;
  double nu = 0.0;
  EXPECT_NO_THROW (stan::prob::chi_square_log(1.0, 1.0));
  EXPECT_NO_THROW (stan::prob::chi_square_log(-1.0, 1.0));
  EXPECT_NO_THROW (stan::prob::chi_square_log(std::numeric_limits<double>::infinity(), 1.0));
  EXPECT_NO_THROW (stan::prob::chi_square_log(-std::numeric_limits<double>::infinity(), 1.0));


  EXPECT_THROW(stan::prob::chi_square_log(y, nu), std::domain_error);
  EXPECT_THROW(stan::prob::chi_square_log(y, -1), std::domain_error);
  EXPECT_THROW(stan::prob::chi_square_log(-1, nu), std::domain_error);
  EXPECT_THROW(stan::prob::chi_square_log(std::numeric_limits<double>::quiet_NaN(), 1.0), 
               std::domain_error);
  EXPECT_THROW(stan::prob::chi_square_log(1.0, std::numeric_limits<double>::quiet_NaN()),
               std::domain_error);
  EXPECT_THROW (stan::prob::chi_square_log(1.0, std::numeric_limits<double>::infinity()),
                std::domain_error);
}
TEST(ProbDistributionsChiSquare,ErrnoPolicy) {
  double result;
  double y = 0.0;
  double nu = 0.0;
  
  result = stan::prob::chi_square_log(1.0, 1.0, errno_policy());
  EXPECT_FALSE (std::isnan(result));
  result = stan::prob::chi_square_log(-1.0, 1.0, errno_policy());
  EXPECT_FALSE (std::isnan(result));

  result = stan::prob::chi_square_log(y, nu, errno_policy());
  EXPECT_TRUE (std::isnan(result));
  result = stan::prob::chi_square_log(y, -1.0, errno_policy());
  EXPECT_TRUE (std::isnan(result));
  result = stan::prob::chi_square_log(-1.0, nu, errno_policy());  
  EXPECT_TRUE (std::isnan(result));
}
TEST(ProbDistributionsChiSquare,Cumulative) {
  using stan::prob::chi_square_p;
  EXPECT_FLOAT_EQ(0.9518757, chi_square_p(7.9,3.0));
  EXPECT_FLOAT_EQ(0.9267752, chi_square_p(1.9,0.5));
}
