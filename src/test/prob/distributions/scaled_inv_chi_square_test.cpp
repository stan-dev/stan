#include <gtest/gtest.h>
#include <stan/prob/distributions/scaled_inv_chi_square.hpp>

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

using stan::prob::scaled_inv_chi_square_log;

TEST(ProbDistributionsScaledInvChiSquare,ScaledInvChiSquare) {
  EXPECT_FLOAT_EQ(-3.091965, scaled_inv_chi_square_log(12.7,6.1,3.0));
  EXPECT_FLOAT_EQ(-1.737086, scaled_inv_chi_square_log(1.0,1.0,0.5));
}
TEST(ProbDistributionsScaledInvChiSquare,Propto) {
  EXPECT_FLOAT_EQ(0.0, scaled_inv_chi_square_log<true>(12.7,6.1,3.0));
  EXPECT_FLOAT_EQ(0.0, scaled_inv_chi_square_log<true>(1.0,1.0,0.5));
}
TEST(ProbDistributionsScaledInvChiSquare,DefaultPolicy) {
  double inf = std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();

  double y_valid = 10;
  double nu_valid = 1.0;
  double scale_valid = 1.0;

  EXPECT_NO_THROW(scaled_inv_chi_square_log(y_valid, nu_valid, scale_valid));
  EXPECT_NO_THROW(scaled_inv_chi_square_log(inf, nu_valid, scale_valid));
  EXPECT_NO_THROW(scaled_inv_chi_square_log(-inf, nu_valid, scale_valid));
  
  EXPECT_THROW(scaled_inv_chi_square_log(nan, nu_valid, scale_valid),
               std::domain_error) << "error on y = nan";
  EXPECT_THROW(scaled_inv_chi_square_log(y_valid, 0.0, scale_valid),
               std::domain_error) << "error on nu <= 0";
  EXPECT_THROW(scaled_inv_chi_square_log(y_valid, -1.0, scale_valid),
               std::domain_error) << "error on nu <= 0";
  EXPECT_THROW(scaled_inv_chi_square_log(y_valid, inf, scale_valid),
               std::domain_error) << "error on nu = inf";
  EXPECT_THROW(scaled_inv_chi_square_log(y_valid, nan, scale_valid),
               std::domain_error) << "error on nu = nan";
  EXPECT_THROW(scaled_inv_chi_square_log(y_valid, nu_valid, 0.0),
               std::domain_error) << "error on scale <= 0";
  EXPECT_THROW(scaled_inv_chi_square_log(y_valid, nu_valid, -1.0),
               std::domain_error) << "error on scale <= 0";
  EXPECT_THROW(scaled_inv_chi_square_log(y_valid, nu_valid, inf),
               std::domain_error) << "error on scale = inf";
  EXPECT_THROW(scaled_inv_chi_square_log(y_valid, nu_valid, nan),
               std::domain_error) << "error on scale = nan";
  }
TEST(ProbDistributionsScaledInvChiSquare,ErrnoPolicy) {
  double inf = std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  double result = 0;

  double y_valid = 10;
  double nu_valid = 1.0;
  double scale_valid = 1.0;

  result = scaled_inv_chi_square_log(y_valid, nu_valid, scale_valid, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = scaled_inv_chi_square_log(inf, nu_valid, scale_valid, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  result = scaled_inv_chi_square_log(-inf, nu_valid, scale_valid, errno_policy());
  EXPECT_FALSE(std::isnan(result));
  
  EXPECT_NO_THROW(result = scaled_inv_chi_square_log(nan, nu_valid, scale_valid, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "error on y = nan";
  EXPECT_NO_THROW(result = scaled_inv_chi_square_log(y_valid, 0.0, scale_valid, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "error on nu <= 0";
  EXPECT_NO_THROW(result = scaled_inv_chi_square_log(y_valid, -1.0, scale_valid, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "error on nu <= 0";
  EXPECT_NO_THROW(result = scaled_inv_chi_square_log(y_valid, inf, scale_valid, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "error on nu = inf";
  EXPECT_NO_THROW(result = scaled_inv_chi_square_log(y_valid, nan, scale_valid, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "error on nu = nan";
  EXPECT_NO_THROW(result = scaled_inv_chi_square_log(y_valid, nu_valid, 0.0, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "error on scale <= 0";
  EXPECT_NO_THROW(result = scaled_inv_chi_square_log(y_valid, nu_valid, -1.0, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "error on scale <= 0";
  EXPECT_NO_THROW(result = scaled_inv_chi_square_log(y_valid, nu_valid, inf, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "error on scale = inf";
  EXPECT_NO_THROW(result = scaled_inv_chi_square_log(y_valid, nu_valid, nan, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "error on scale = nan";
}
