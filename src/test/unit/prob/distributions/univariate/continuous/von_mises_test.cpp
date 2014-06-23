#include <stan/prob/distributions/univariate/continuous/von_mises.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/matrix/typedefs.hpp>

TEST(ProbDistributionsVonMises, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::von_mises_rng(1.0,2.0,rng));

  EXPECT_THROW(stan::prob::von_mises_rng(stan::math::negative_infinity(),2.0,
                                         rng), std::domain_error);
  EXPECT_THROW(stan::prob::von_mises_rng(1,stan::math::positive_infinity(),rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::von_mises_rng(1,-3,rng), std::domain_error);
  EXPECT_NO_THROW(stan::prob::von_mises_rng(2,1,rng));
}

TEST(ProbDistributionsVonMises, chiSquareGoodnessFitTest1) {
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
  for (int i = 0; i < K-1; i ++)
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


TEST(ProbDistributionsVonMises, chiSquareGoodnessFitTest2) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = 80;
  boost::math::chi_squared mydist(K-1);

  stan::math::vector_d loc(K - 1);

  loc << 1.5890, 1.8351, 1.9771, 2.0788, 2.1590, 2.2258, 2.2833, 2.3343,
    2.3801, 2.4220, 2.4606, 2.4967, 2.5305, 2.5625, 2.5928, 2.6218, 2.6496,
    2.6763, 2.7022, 2.7271, 2.7514, 2.7750, 2.7980, 2.8205, 2.8425, 2.8641,
    2.8853, 2.9062, 2.9268, 2.9471, 2.9672, 2.9871, 3.0068, 3.0263, 3.0457,
    3.0650, 3.0842, 3.1034, 3.1225, 3.1416, 3.1607, 3.1798, 3.1989, 3.2182,
    3.2375, 3.2569, 3.2764, 3.2961, 3.3160, 3.3361, 3.3564, 3.3770, 3.3978,
    3.4191, 3.4407, 3.4627, 3.4852, 3.5082, 3.5318, 3.5560, 3.5810, 3.6068,
    3.6336, 3.6614, 3.6904, 3.7207, 3.7527, 3.7865, 3.8226, 3.8612, 3.9031,
    3.9489, 3.9998, 4.0574, 4.1242, 4.2044, 4.3061, 4.4481, 4.6942;

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }

  while (count < N) {
    double a = stan::prob::von_mises_rng(11*3.14,3.0,rng);
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

TEST(ProbDistributionsVonMises, chiSquareGoodnessFitTest3) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = 80;
  boost::math::chi_squared mydist(K-1);

  stan::math::vector_d loc(K - 1);
  loc << -0.552560, -0.306511, -0.164502, -0.062786,  0.017401,  0.084166, 
    0.141753, 0.192668,  0.238514,  0.280381, 0.319043, 0.355071, 0.388899,
    0.420864, 0.451235,  0.480226,  0.508016,  0.534753,  0.560560,  0.585545,
    0.609796, 0.633395,  0.656409,  0.678900,  0.700922,  0.722523,  0.743748, 
    0.764635, 0.785221,  0.805540,  0.825622,  0.845496,  0.865190,  0.884728,
    0.904136, 0.923435,  0.942650,  0.961802,  0.980911,  1.000000,  1.019089,  
    1.038198, 1.057350,  1.076565,  1.095864,  1.115272,  1.134810,  1.154504,  
    1.174378, 1.194460,  1.214779,  1.235365,  1.256252,  1.277477,  1.299078, 
    1.321100, 1.343591,  1.366605,  1.390204,  1.414455,  1.439440,  1.465247, 
    1.491984, 1.519774,  1.548765,  1.579136,  1.611101,  1.644929,  1.680957,  
    1.719619, 1.761486,  1.807332,  1.858247,  1.915834,  1.982599,  2.062786, 
    2.164502, 2.306511,  2.552560;

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }

  while (count < N) {
    double a = stan::prob::von_mises_rng(-17.85,3.0,rng);
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
