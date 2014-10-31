#include <stan/prob/distributions/univariate/discrete/neg_binomial.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionsNegBinomial, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::neg_binomial_rng(6, 2, rng));
  EXPECT_NO_THROW(stan::prob::neg_binomial_rng(0.5,1,rng));
  EXPECT_NO_THROW(stan::prob::neg_binomial_rng(1e9,1,rng));

  EXPECT_THROW(stan::prob::neg_binomial_rng(0, 2, rng),std::domain_error);
  EXPECT_THROW(stan::prob::neg_binomial_rng(-6, 2, rng),std::domain_error);
  EXPECT_THROW(stan::prob::neg_binomial_rng(6, -2, rng),std::domain_error);
  EXPECT_THROW(stan::prob::neg_binomial_rng(stan::math::positive_infinity(), 2, 
                                            rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::neg_binomial_rng(1e10,1,rng),std::domain_error);

  EXPECT_THROW(stan::prob::neg_binomial_rng(6,stan::math::positive_infinity(), 
                                            rng),
               std::domain_error);
}

void expected_bin_sizes(double *expect, const int K, 
                        const int N, 
                        const double alpha, const double beta) {
  double p = 0;
  for(int i = 0 ; i < K; i++)  {
    expect[i] = N * std::exp(stan::prob::neg_binomial_log(i, alpha, beta));
    p += std::exp(stan::prob::neg_binomial_log(i, alpha, beta));
  }
  expect[K-1] = N * (1.0 - p);
}

TEST(ProbDistributionsNegBinomial, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  double p = 0.6;
  double alpha = 5;
  double beta = p / (1 - p);
  int N = 1000;
  int K = boost::math::round(2 * std::pow(N, (1-p)));
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin [K];
  double expect [K];

  for(int i = 0 ; i < K; i++) 
    bin[i] = 0;
  expected_bin_sizes(expect, K, N, alpha, beta);

  while (count < N) {
    int a = stan::prob::neg_binomial_rng(alpha, beta, rng);
    int i = 0;
    while (i < K-1 && a > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < boost::math::quantile(boost::math::complement(mydist, 1e-6)));
}

TEST(ProbDistributionsNegBinomial, chiSquareGoodnessFitTest2) {
  boost::random::mt19937 rng;
  double p = 0.8;
  double alpha = 2.4;
  double beta = p / (1 - p);
  int N = 1000;
  int K = boost::math::round(2 * std::pow(N, (1-p)));
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin [K];
  double expect [K];

  for(int i = 0 ; i < K; i++) 
    bin[i] = 0;
  expected_bin_sizes(expect, K, N, alpha, beta);

  while (count < N) {
    int a = stan::prob::neg_binomial_rng(alpha, beta, rng);
    int i = 0;
    while (i < K-1 && a > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < boost::math::quantile(boost::math::complement(mydist, 1e-6)));
}

TEST(ProbDistributionsNegBinomial, chiSquareGoodnessFitTest3) {
  boost::random::mt19937 rng;
  double p = 0.2;
  double alpha = 0.4;
  double beta = p / (1 - p);
  int N = 1000;
  int K = boost::math::round(2 * std::pow(N, (1-p)));
  boost::math::chi_squared mydist(K-1);

  int loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  double bin [K];
  double expect [K];

  for(int i = 0 ; i < K; i++) 
    bin[i] = 0;
  expected_bin_sizes(expect, K, N, alpha, beta);

  while (count < N) {
    int a = stan::prob::neg_binomial_rng(alpha, beta, rng);
    int i = 0;
    while (i < K-1 && a > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < boost::math::quantile(boost::math::complement(mydist, 1e-6)));
}
