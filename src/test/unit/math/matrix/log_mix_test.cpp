#include <cmath>
#include <limits>
#include <stdexcept>
#include <stan/math/matrix/log_mix.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, log_mix_exceptions) {
  using stan::math::log_mix;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  Matrix<double, Dynamic, 1> alpha_bad(4);
  Matrix<double, Dynamic, 1> lambda_bad(5);
  
  alpha_bad << 0.1, 0.2, 0.3, 0.4;
  lambda_bad << -0.2, 0.4, 0.5, 0.4, 0.7;

  EXPECT_THROW(log_mix(alpha_bad,lambda_bad), std::invalid_argument);

  lambda_bad.resize(4);
  alpha_bad.resize(5);
  alpha_bad << 0.1, 0.2, 0.3, 0.3, 0.1;
  EXPECT_THROW(log_mix(alpha_bad,lambda_bad), std::invalid_argument);

  alpha_bad.resize(4);
  alpha_bad << 0.1, 0.2, 0.3, 0.4;
  lambda_bad << -0.2, std::numeric_limits<double>::quiet_NaN(), 0.5, 0.4;
  EXPECT_THROW(log_mix(alpha_bad,lambda_bad), std::domain_error);

  alpha_bad << 0.1, std::numeric_limits<double>::quiet_NaN(), 0.3, 0.4;
  EXPECT_THROW(log_mix(alpha_bad,lambda_bad), std::domain_error);

  alpha_bad << 0.1, std::numeric_limits<double>::quiet_NaN(), 0.3, 0.4;
  lambda_bad << -0.2, std::numeric_limits<double>::quiet_NaN(), 0.5, 0.4;
  EXPECT_THROW(log_mix(alpha_bad,lambda_bad), std::domain_error);

  alpha_bad << 0.1, 0.2, 0.3, 0.5;
  EXPECT_THROW(log_mix(alpha_bad,lambda_bad), std::domain_error);

  alpha_bad << -0.1, 0.2, 0.3, 0.4;
  EXPECT_THROW(log_mix(alpha_bad,lambda_bad), std::domain_error);
}

void test_log_mix(Eigen::Matrix<double, Eigen::Dynamic, 1> alpha,
                  Eigen::Matrix<double, Eigen::Dynamic, 1> lambda) {
  using std::exp;
  using std::log;
  using stan::math::log_mix;

  double validation = 0.0;
  for (int i = 0; i < alpha.size(); ++i)
    validation += alpha(i) * exp(lambda(i));
  validation = log(validation);

  EXPECT_FLOAT_EQ(validation, log_mix(alpha,lambda));
}

TEST(MathFunctions, log_mix_values) {
  using Eigen::Matrix;
  using Eigen::Dynamic;

  Matrix<double, Dynamic, 1> alpha_test(5);
  Matrix<double, Dynamic, 1> lambda_test(5);

  alpha_test << 0.1, 0.3, 0.1, 0.3, 0.2;
  lambda_test << -0.1, 0.3, -0.1, 0.3, 0.2;

  test_log_mix(alpha_test,lambda_test);
//  test_log_mix(0.0001, 197, -3000);
//  test_log_mix(0.999999, 197, -3000);
}

