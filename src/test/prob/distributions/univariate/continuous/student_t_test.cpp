#include <stan/prob/distributions/univariate/continuous/student_t.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsStudentT, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::student_t_rng(3.0, 2.0, 1.0, rng));
}
