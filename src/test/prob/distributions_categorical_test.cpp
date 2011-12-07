#include <gtest/gtest.h>
#include <boost/exception/all.hpp>
#include "stan/prob/distributions_categorical.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributions,Categorical) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  EXPECT_FLOAT_EQ(-1.203973, stan::prob::categorical_log(0,theta));
  EXPECT_FLOAT_EQ(-0.6931472, stan::prob::categorical_log(1,theta));
}
