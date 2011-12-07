#include <gtest/gtest.h>
#include "stan/prob/distributions_multinomial.hpp"
#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::Dynamic;


TEST(ProbDistributions,Multinomial) {
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  EXPECT_FLOAT_EQ(-2.002481, stan::prob::multinomial_log(ns,theta));
}
