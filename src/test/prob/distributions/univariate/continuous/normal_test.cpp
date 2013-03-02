#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include<boost/math/distributions.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsNormal, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::normal_rng(10.0,2.0,rng));
}

TEST(ProbDistributionsNormal, doesFit) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = 5;
  boost::math::normal_distribution<>dist (10.0, 2.0);
  boost::math::chi_squared mydist(K-1);

  double loc[4];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, 0.2 * i);

  int count = 0;
  int bin [5] = {0, 0, 0, 0, 0};

  while (count < N) {
    double a = stan::prob::normal_rng(10.0,2.0,rng);
    int i = 0;
    while (i < K-1 && a > loc[i]) 
	++i;
    ++bin[i];
    count++;
   }

  double chi = 0;
  double expect [5] = {N / K, N / K, N / K, N / K, N / K};

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}
