#include "stan/math/functions/inverse_softmax.hpp"
#include <gtest/gtest.h>

TEST(MathsSpecialFunctions, inverse_softmax_exception) {
  std::vector<double> simplex(2);
  std::vector<double> y(3);
  EXPECT_THROW(stan::math::inverse_softmax< std::vector<double> >(simplex, y), 
               std::invalid_argument);
}
