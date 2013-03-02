#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include<boost/math/distributions.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <iostream>

TEST(ProbDistributionsNormal, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::normal_rng(10.0,2.0,rng));
}

TEST(ProbDistributionsNormal, isnormal) {
  boost::random::mt19937 rng;

  boost::math::normal_distribution<>dist (10.0, 2.0);

  double loc[4];
  for(int i = 1; i < 5; i++)
    loc[i - 1] = quantile(dist, 0.2 * i);

  int count = 0;
  double a = 0;
  int bin [5] = {0, 0, 0, 0, 0};

  while(count < 10000)
    {
      a = stan::prob::normal_rng(10.0,2.0,rng);
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
