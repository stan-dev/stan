#include <stan/math/prim/mat.hpp>
#include <stan/analyze/mcmc/welford_variance.hpp>
#include <gtest/gtest.h>

TEST(WelfordVariance, sample_variance_unbiased) {
  const int N = 10;
  Eigen::VectorXd y(N);
  for (int n = 0; n < N; ++n) {
    y(n) = n;
  }

  EXPECT_EQ(55.0 / 6.0, stan::analyze::welford_variance(y));
}

TEST(WelfordVariance, sample_variance_biased) {
  const int N = 10;
  Eigen::VectorXd y(N);
  for (int n = 0; n < N; ++n) {
    y(n) = n;
  }

  EXPECT_EQ(33.0 / 4.0, stan::analyze::welford_variance(y, 0));
}
