#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <boost/math/distributions.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsNormal, intVsDouble) {
  using stan::agrad::var;
  for (double thetaval = -5.0; thetaval < 6.0; thetaval += 0.5) {
    var theta(thetaval);
    var lp1(0.0);
    lp1 += stan::prob::normal_log<true>(0, theta, 1);
    double lp1val = lp1.val();
    lp1.grad();
    double lp1adj = lp1.adj();

    var theta2(thetaval);
    var lp2(0.0);
    lp2 += stan::prob::normal_log<true>(theta2, 0, 1);
    double lp2val = lp2.val();
    lp2.grad();
    double lp2adj = lp2.adj();
    EXPECT_FLOAT_EQ(lp1val,lp2val);
    EXPECT_FLOAT_EQ(lp1adj,lp2adj);
    std::cout << lp1val << std::endl;
    std::cout << lp2val << std::endl;
  }
}

TEST(ProbDistributionsNormal, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::normal_rng(10.0,2.0,rng));
}

TEST(ProbDistributionsNormal, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::normal_distribution<>dist (2.0,1.0);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++){
    bin[i] = 0;
    expect[i] = N / K;
  }

  while (count < N) {
    double a = stan::prob::normal_rng(2.0,1.0,rng);
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

