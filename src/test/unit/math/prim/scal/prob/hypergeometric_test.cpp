#include <vector>
#include <stan/math/prim/scal/prob/hypergeometric_rng.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionsHypergeometric, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::math::hypergeometric_rng(10, 10, 15,rng));

  EXPECT_THROW(stan::math::hypergeometric_rng(30, 10, 15,rng),
               std::domain_error);
  EXPECT_THROW(stan::math::hypergeometric_rng(-30, 10, 15,rng),
               std::domain_error);
  EXPECT_THROW(stan::math::hypergeometric_rng(30, -10, 15,rng),
               std::domain_error);
  EXPECT_THROW(stan::math::hypergeometric_rng(30, 10, -15,rng),
               std::domain_error);
}

TEST(ProbDistributionsHypergeometric, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int num_draws = 10;
  int K = num_draws;
  boost::math::hypergeometric_distribution<>dist (15, num_draws, 25);
  boost::math::chi_squared mydist(K-1);

  std::vector<int> loc(K - 1);
  for(int i = 1; i < K; i++)
    loc[i - 1] = i - 1;

  int count = 0;
  std::vector<int> bin(K);
  std::vector<double> expect(K);
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N * pdf(dist, i);
  }

  while (count < N) {
    int a = stan::math::hypergeometric_rng(num_draws, 10, 15,rng);
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
