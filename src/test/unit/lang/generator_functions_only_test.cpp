#include <gtest/gtest.h>
#include <test/test-models/good/double_functions-fun.hpp>
#include <sstream>


class DoublesFunctions : public testing::Test { };


TEST_F(DoublesFunctions, f01) {
  double result;
  result = f01(1, &std::cout);
  EXPECT_EQ(2, result);
  result = f01(3, &std::cout);
  EXPECT_EQ(6, result);
}




