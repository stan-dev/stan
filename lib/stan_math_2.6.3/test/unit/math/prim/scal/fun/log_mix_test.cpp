#include <cmath>
#include <limits>
#include <stdexcept>
#include <stan/math/prim/scal/fun/log_mix.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, log_mix_exceptions) {
  using stan::math::log_mix;
  EXPECT_THROW(log_mix(-1, 10, 20), std::domain_error);
  EXPECT_THROW(log_mix(std::numeric_limits<double>::quiet_NaN(), 10, 20), 
               std::domain_error);
  EXPECT_THROW(log_mix(0.5, std::numeric_limits<double>::quiet_NaN(), 10),
               std::domain_error);
  EXPECT_THROW(log_mix(0.5, 10, std::numeric_limits<double>::quiet_NaN()),
               std::domain_error);
}
void test_log_mix(double theta, double lambda1, double lambda2) {
  using std::exp;
  using std::log;
  using stan::math::log_mix;
  EXPECT_FLOAT_EQ(log(theta * exp(lambda1) + (1 - theta) * exp(lambda2)),
                  log_mix(theta,lambda1,lambda2));
}

TEST(MathFunctions, log_mix_values) {
  test_log_mix(0.3, 1.7, -3.9);
  test_log_mix(0.0001, 197, -3000);
  test_log_mix(0.999999, 197, -3000);
}

