#include <gtest/gtest.h>
#include "stan/prob/distributions_pareto.hpp"

TEST(ProbDistributions,Pareto) {
  EXPECT_FLOAT_EQ(-1.909543, stan::prob::pareto_log(1.5,0.5,2.0));
  EXPECT_FLOAT_EQ(-25.69865, stan::prob::pareto_log(19.5,0.15,5.0));
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::pareto_log(0.0,0.15,5.0));
}
TEST(ProbDistributions,ParetoPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::pareto_log<true>(1.5,0.5,2.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::pareto_log<true>(19.5,0.15,5.0));
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::pareto_log(0.0,0.15,5.0));
}
