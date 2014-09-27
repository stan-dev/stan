#include <stan/math/functions/primitive_value.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, primitiveValue) {
  using stan::math::primitive_value;
  EXPECT_EQ(5,primitive_value(5));       // int
  EXPECT_EQ(5U, primitive_value(5U));    // uint
  EXPECT_EQ(10000000000L, primitive_value(10000000000L));  // long >> int
  EXPECT_EQ('a', primitive_value('a'));  // char

  EXPECT_EQ(7.3,primitive_value(7.3));  // double
  EXPECT_EQ(7.3f,primitive_value(7.3f));  // float
}

TEST(MathFunctions, primiviteValueNaN) {
  using boost::math::isnan;
  using std::numeric_limits;
  using stan::math::primitive_value;
  
  EXPECT_TRUE(isnan<double>(primitive_value(numeric_limits<double>::quiet_NaN())));
}
