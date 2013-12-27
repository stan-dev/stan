#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionsBernoulli, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::bernoulli_rng(0.6,rng));

  EXPECT_THROW(stan::prob::bernoulli_rng(1.6,rng),std::domain_error);
  EXPECT_THROW(stan::prob::bernoulli_rng(-0.6,rng),std::domain_error);
  EXPECT_THROW(stan::prob::bernoulli_rng(stan::math::positive_infinity(),rng),
               std::domain_error);
}

TEST(ProbDistributionsBernoulli, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  boost::math::bernoulli_distribution<>dist (0.4);
  boost::math::chi_squared mydist(1);
 
  int bin[2] = {0, 0};
  double expect [2] = {N * (1 - 0.4), N * (0.4)};

  int count = 0;

  while (count < N) {
    int a = stan::prob::bernoulli_rng(0.4,rng);
    if(a == 1)
      ++bin[1];
    else
      ++bin[0];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < 2; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}
