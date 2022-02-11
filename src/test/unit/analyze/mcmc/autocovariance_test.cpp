#include <stan/math/prim.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <stdexcept>

TEST(ProbAutocovariance, test1) {
  // ar1.csv generated in R with
  //   > x[1] <- rnorm(1, 0, 1)
  //   > for (n in 2:1000) x[n] <- rnorm(1, 0.8 * x[n-1], 1)
  std::fstream f("src/test/unit/analyze/mcmc/ar1.csv");
  size_t N = 1000;
  std::vector<double> y;
  for (size_t i = 0; i < N; ++i) {
    double temp;
    f >> temp;
    y.push_back(temp);
  }

  // 10K 1K-length AC in 2.9s with g++ -O3 on Bob's Macbook Air
  std::vector<double> ac;
  stan::analyze::autocovariance(y, ac);

  EXPECT_EQ(1000U, ac.size());

  EXPECT_NEAR(2.69, ac[0], 0.01);
  EXPECT_NEAR(2.16, ac[1], 0.01);
  EXPECT_NEAR(1.73, ac[2], 0.01);
  EXPECT_NEAR(1.38, ac[3], 0.01);
  EXPECT_NEAR(1.10, ac[4], 0.01);
  EXPECT_NEAR(0.89, ac[5], 0.01);
}

TEST(ProbAutocovariance, test2) {
  // ar1.csv generated in R with
  //   > x[1] <- rnorm(1, 0, 1)
  //   > for (n in 2:1000) x[n] <- rnorm(1, 0.8 * x[n-1], 1)
  std::fstream f("src/test/unit/analyze/mcmc/ar1.csv");
  size_t N = 1000;
  Eigen::VectorXd y(N);
  for (size_t i = 0; i < N; ++i) {
    double temp;
    f >> temp;
    y(i) = temp;
  }

  // 10K 1K-length AC in 2.9s with g++ -O3 on Bob's Macbook Air
  Eigen::VectorXd ac(N);
  stan::analyze::autocovariance<double>(y, ac);

  EXPECT_EQ(1000U, ac.size());

  EXPECT_NEAR(2.69, ac(0), 0.01);
  EXPECT_NEAR(2.16, ac(1), 0.01);
  EXPECT_NEAR(1.73, ac(2), 0.01);
  EXPECT_NEAR(1.38, ac(3), 0.01);
  EXPECT_NEAR(1.10, ac(4), 0.01);
  EXPECT_NEAR(0.89, ac(5), 0.01);
}
