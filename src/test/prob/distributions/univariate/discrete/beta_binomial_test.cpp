#include <stan/prob/distributions/univariate/discrete/beta_binomial.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionBetaBinomial, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::beta_binomial_rng(4,0.6,2.0,rng));
}

TEST(ProbDistributionsBetaBinomial, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int num_N = 10;
  int K = num_N + 1;
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++)
  {
    bin[i] = 0;
    expect[i] = N * exp(stan::prob::beta_binomial_log(i,10,0.6,2.0));
  }

  while (count < N) {
    int a = stan::prob::beta_binomial_rng(num_N,0.6,2.0,rng);
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
