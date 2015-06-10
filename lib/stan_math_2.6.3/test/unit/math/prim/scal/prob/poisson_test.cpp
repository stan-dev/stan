#include <stan/math/prim/scal/prob/poisson_rng.hpp>
#include <stan/math/prim/scal/prob/poisson_log_rng.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionsPoisson, error_check) {
  using std::log;

  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::math::poisson_rng(6, rng));

  EXPECT_THROW(stan::math::poisson_rng(-6, rng),std::domain_error);

  EXPECT_NO_THROW(stan::math::poisson_rng(1e9, rng));

  EXPECT_THROW(stan::math::poisson_rng(pow(2.0,31), rng),std::domain_error);

  EXPECT_NO_THROW(stan::math::poisson_log_rng(6, rng));

  EXPECT_NO_THROW(stan::math::poisson_log_rng(-6, rng));

  EXPECT_NO_THROW(stan::math::poisson_log_rng(log(1e9), rng));

  EXPECT_THROW(stan::math::poisson_log_rng(log(pow(2.0,31)), rng),std::domain_error);
}

TEST(ProbDistributionsPoisson, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 1000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::poisson_distribution<>dist (5);
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }
  expect[K-1] = N * (1 - cdf(dist, K - 1));

  while (count < N) {
    int a = stan::math::poisson_rng(5,rng);
    int i = 0;
    while (i < K-1 && a > loc[i])
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

TEST(ProbDistributionsPoisson, chiSquareGoodnessFitTest2) {
  using std::log;

  boost::random::mt19937 rng;
  int N = 1000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::poisson_distribution<>dist (5);
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }
  expect[K-1] = N * (1 - cdf(dist, K - 1));

  while (count < N) {
    int a = stan::math::poisson_log_rng(log(5),rng);
    int i = 0;
    while (i < K-1 && a > loc[i])
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}
