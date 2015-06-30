#include <stan/math/prim/scal/prob/chi_square_rng.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionsChiSquare, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::math::chi_square_rng(2.0,rng));

  EXPECT_THROW(stan::math::chi_square_rng(-2.0,rng),std::domain_error);
  EXPECT_THROW(stan::math::chi_square_rng(stan::math::positive_infinity(),rng),
               std::domain_error);
}

TEST(ProbDistributionsChiSquare, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::chi_squared_distribution<>dist (2.0);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }

  while (count < N) {
    double a = stan::math::chi_square_rng(2.0,rng);
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
