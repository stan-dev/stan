#include "stan/math/functions/softmax.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, softmax) {
  std::vector<double> x;
  x.push_back(1.0);
  x.push_back(-1.0);
  x.push_back(2.0);
  std::vector<double> simplex(3);
  stan::math::softmax<std::vector<double>,double>(x,simplex);
  EXPECT_FLOAT_EQ(1.0, simplex[0] + simplex[1] + simplex[2]);
  double sum = exp(1.0) + exp(-1.0) + exp(2.0);
  EXPECT_FLOAT_EQ(exp(1.0)/sum, simplex[0]);
  EXPECT_FLOAT_EQ(exp(-1.0)/sum, simplex[1]);
  EXPECT_FLOAT_EQ(exp(2.0)/sum, simplex[2]);
}
TEST(MathFunctions, softmax_exception) {
  std::vector<double> x;
  x.push_back(1.0);
  x.push_back(-1.0);
  x.push_back(2.0);
  std::vector<double> simplex(2);
  
  // dl: EXPECT_THROW is not able to handle this templating
  //EXPECT_THROW(stan::math::softmax<std::vector<double>,double>(x,simplex), std::invalid_argument);
  try{
    stan::math::softmax<std::vector<double>,double>(x,simplex);
    FAIL();
  } catch (std::invalid_argument e) {
    SUCCEED();
  }
}
