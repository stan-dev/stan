#include <stan/math/prim/scal/prob/binomial_rng.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionBinomiali, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::math::binomial_rng(4,0.6,rng));
  EXPECT_THROW(stan::math::binomial_rng(-4,0.6,rng),std::domain_error);
  EXPECT_THROW(stan::math::binomial_rng(4,-0.6,rng),std::domain_error);
  EXPECT_THROW(stan::math::binomial_rng(4,2.6,rng),std::domain_error);
  EXPECT_THROW(stan::math::binomial_rng(4,stan::math::positive_infinity(),rng),
               std::domain_error);
}

TEST(ProbDistributionsBinomial, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::binomial_distribution<>dist (100,0.6);
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }
  expect[K-1] = N * (1 - cdf(dist, K-2));

  while (count < N) {
    int a = stan::math::binomial_rng(100,0.6,rng);
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

