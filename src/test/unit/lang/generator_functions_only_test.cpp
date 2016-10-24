#include <gtest/gtest.h>
#include <test/test-models/good/double_functions-fun.hpp>
#include <sstream>


class DoublesFunctions : public testing::Test { };


TEST_F(DoublesFunctions, f01) {
  double out;
  out = f01(1);
  EXPECT_EQ(2, out);
}




