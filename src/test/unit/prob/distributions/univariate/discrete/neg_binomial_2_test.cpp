#include <stan/prob/distributions/univariate/discrete/neg_binomial_2.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionsNegBinomial, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::neg_binomial_2_rng(6, 2, rng));
  EXPECT_NO_THROW(stan::prob::neg_binomial_2_rng(0.5,1,rng));
  EXPECT_NO_THROW(stan::prob::neg_binomial_2_rng(1e8,1,rng));

  EXPECT_THROW(stan::prob::neg_binomial_2_rng(0, -2, rng),std::domain_error);
  EXPECT_THROW(stan::prob::neg_binomial_2_rng(-6, 2, rng),std::domain_error);
  EXPECT_THROW(stan::prob::neg_binomial_2_rng(-6, -0.1, rng),std::domain_error);

  EXPECT_THROW(stan::prob::neg_binomial_2_rng(1e10, 1e20, rng),std::domain_error);

  EXPECT_THROW(stan::prob::neg_binomial_2_rng(stan::math::positive_infinity(), 2,
                                            rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::neg_binomial_2_rng(stan::math::positive_infinity(),
                                            6, rng),
               std::domain_error);
}

TEST(ProbDistributionsNegBinomial, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 1000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::negative_binomial_distribution<>dist (1.1,1.1/(1.1+2.4));
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++)  {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }
  expect[K-1] = N * (1 - cdf(dist, K - 1));

  while (count < N) {
    int a = stan::prob::neg_binomial_2_rng(2.4,1.1,rng);
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

TEST(ProbDistributionsNegBinomial, chiSquareGoodnessFitTest2) {
  boost::random::mt19937 rng;
  int N = 1000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::negative_binomial_distribution<>dist (0.6,0.6/(0.6+2.4));
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++)  {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }
  expect[K-1] = N * (1 - cdf(dist, K - 1));

  while (count < N) {
    int a = stan::prob::neg_binomial_2_rng(2.4,0.6,rng);
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

TEST(ProbDistributionsNegBinomial, chiSquareGoodnessFitTest3) {
  boost::random::mt19937 rng;
  int N = 1000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::negative_binomial_distribution<>dist (30,30/(30+60.4));
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++)  {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }
  expect[K-1] = N * (1 - cdf(dist, K - 1));

  while (count < N) {
    int a = stan::prob::neg_binomial_2_rng(60.4,30,rng);
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

TEST(ProbDistributionsNegBinomial, chiSquareGoodnessFitTest4) {
  boost::random::mt19937 rng;
  int N = 1000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::negative_binomial_distribution<>dist (80,80/(80+30.4));
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++)  {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }
  expect[K-1] = N * (1 - cdf(dist, K - 1));

  while (count < N) {
    int a = stan::prob::neg_binomial_2_rng(30.4,80,rng);
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
