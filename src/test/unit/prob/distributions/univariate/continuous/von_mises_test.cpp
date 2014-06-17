#include <stan/prob/distributions/univariate/continuous/von_mises.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/matrix/typedefs.hpp>

TEST(ProbDistributionsVonMises, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::von_mises_rng(1.0,2.0,rng));

  EXPECT_NO_THROW(stan::prob::von_mises_rng(stan::math::negative_infinity(),2.0,rng));
  EXPECT_THROW(stan::prob::von_mises_rng(1,stan::math::positive_infinity(),rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::von_mises_rng(1,-3,rng), std::domain_error);
  EXPECT_NO_THROW(stan::prob::von_mises_rng(2,1,rng));
}

TEST(ProbDistributionsVonMises, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = 80;
  boost::math::chi_squared mydist(K-1);

  stan::math::vector_d loc(K - 1);
  loc << 1.589032, 1.835082, 1.977091, 2.078807, 2.158994, 2.225759, 2.283346,
    2.334261, 2.380107, 2.421973, 2.460635, 2.496664, 2.530492, 2.562457,
    2.592827, 2.621819, 2.649609, 2.676346, 2.702153, 2.727137, 2.751389,
    2.774988, 2.798002, 2.820493, 2.842515, 2.864116, 2.885340, 2.906227,
    2.926814, 2.947132, 2.967214, 2.987089, 3.006782, 3.026321, 3.045728,
    3.065028, 3.084243, 3.103394, 3.122504, 3.141593, 3.160681, 3.179791,
    3.198942, 3.218157, 3.237457, 3.256865, 3.276403, 3.296097, 3.315971, 
    3.336053, 3.356372, 3.376958, 3.397845, 3.419069, 3.440671, 3.462693,
    3.485184, 3.508198, 3.531796, 3.556048, 3.581032, 3.606840, 3.633576,
    3.661367, 3.690358, 3.720728, 3.752694, 3.786522, 3.822550, 3.861212,
    3.903079, 3.948925, 3.999839, 4.057427, 4.124191, 4.204379, 4.306094, 
    4.448103, 4.694153;
  for (int i = 0; i < K; i ++)
    loc[i] = loc[i] - stan::math::pi();

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }

  while (count < N) {
    double a = stan::prob::von_mises_rng(0,3.0,rng);
    int i = 0;
    while (i < K-1 && a > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++) {
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);
  }

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

