#include <stan/prob/distributions/univariate/continuous/scaled_inv_chi_square.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include<boost/math/distributions.hpp>

TEST(ProbDistributionsScaledInvChiSquare, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::scaled_inv_chi_square_rng(2.0,1.0,rng));

  EXPECT_THROW(stan::prob::scaled_inv_chi_square_rng(-2.0,1.0,rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::scaled_inv_chi_square_rng(2.0,-1.0,rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::scaled_inv_chi_square_rng(stan::math::positive_infinity(),
                                                     1.0,rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::scaled_inv_chi_square_rng(2,
                                                     stan::math::positive_infinity(),
                                                     rng),
               std::domain_error);
}

TEST(ProbDistributionsScaledInvChiSquare, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  double K = 5;
  boost::math::inverse_chi_squared_distribution<>dist (2.0);
  boost::math::chi_squared mydist(K-1);

  double loc[4];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, 0.2 * i);

  int count = 0;
  int bin [5] = {0, 0, 0, 0, 0};

  while (count < N) {
    double a = stan::prob::scaled_inv_chi_square_rng(2.0,1.0,rng) / (2.0 * 1.0);
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
