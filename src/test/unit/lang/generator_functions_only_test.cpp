#include <gtest/gtest.h>
#include <test/test-models/good/double_functions-fun.hpp>
#include <sstream>


class DoublesFunctions : public testing::Test { };


TEST_F(DoublesFunctions, f01) {
  double result;
  result = f01(1.0, &std::cout);
  EXPECT_EQ(2, result);
  result = f01(3.0, &std::cout);
  EXPECT_EQ(6, result);
}

TEST_F(DoublesFunctions, f02) {
  double result;
  result = f02(1.0, 2,  &std::cout);
  EXPECT_EQ(3, result);
  result = f02(1.0, 3, &std::cout);
  EXPECT_EQ(4, result);
}

TEST_F(DoublesFunctions, f03) {
  double result;
  result = f03(1.0, 2, 33.3,  &std::cout);
  EXPECT_EQ(36.3, result);
  result = f03(1.0, 3, 53.3, &std::cout);
  EXPECT_EQ(57.3, result);
}


