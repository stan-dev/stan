#include <gtest/gtest.h>
#include <stan/prob/distributions/student_t.hpp>

TEST(ProbDistributions,StudentT) {
  EXPECT_FLOAT_EQ(-1.837877, stan::prob::student_t_log(1.0,1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-3.596843, stan::prob::student_t_log(-3.0,2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-2.531024, stan::prob::student_t_log(2.0,1.0,0.0,2.0));
  // need test with scale != 1
}
TEST(ProbDistributions,StudentTPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::student_t_log<true>(1.0,1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::student_t_log<true>(-3.0,2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::student_t_log<true>(2.0,1.0,0.0,2.0));
}
