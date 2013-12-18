#include <stan/prob/distributions/univariate/continuous/student_t.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionsStudentT, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::student_t_rng(3.0, 2.0, 2.0, rng));

  EXPECT_THROW(stan::prob::student_t_rng(3.0, 2.0, -2.0, rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::student_t_rng(-3.0, 2.0, 2.0, rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::student_t_rng(stan::math::positive_infinity(), 2.0,
                                         2.0, rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::student_t_rng(3,stan::math::positive_infinity(),
                                         2.0, rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::student_t_rng(3,2,stan::math::positive_infinity(),
                                         rng),
               std::domain_error);

}

TEST(ProbDistributionsStudentT, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  boost::math::students_t_distribution<>dist (3.0);
  int N = 10000;
  double K = 5;
  boost::math::chi_squared mydist(K-1);

  double loc[4];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, 0.2 * i);

  int count = 0;
  int bin [5] = {0, 0, 0, 0, 0};

  while (count < N) {
    double a = (stan::prob::student_t_rng(3.0,2.0,2.0,rng) - 2.0) / 2.0;
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
