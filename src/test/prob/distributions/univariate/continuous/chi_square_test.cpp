#include <stan/prob/distributions/univariate/continuous/chi_square.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <iostream>
#include<boost/math/distributions.hpp>

TEST(ProbDistributionsChiSquare, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::chi_square_rng(2.0,rng));
}

TEST(ProbDistributionsChiSquare, isnormal) {
  boost::random::mt19937 rng;
  boost::math::chi_squared_distribution<>dist (2.0);

  double loc[4];
  for(int i = 1; i < 5; i++)
    loc[i - 1] = quantile(dist, 0.2 * i);

  int count = 0;
  double a = 0;
  int bin [5] = {0, 0, 0, 0, 0};

  while(count < 10000)
    {
      a = stan::prob::chi_square_rng(2.0,rng);
      if(a > loc[3])
	bin[4]++;
      else if(a < loc[3] && a > loc[2])
	bin[3]++;
      else if(a < loc[2] && a > loc[1])
	bin[2]++;
      else if(a < loc[1] && a > loc[0])
	bin[1]++;
      else
	bin[0]++;
      count++;
    }

  double chi = 0;
  double expect [5] = {2000, 2000, 2000, 2000, 2000};

  for(int j = 0; j < 5; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < 9.49);
}
