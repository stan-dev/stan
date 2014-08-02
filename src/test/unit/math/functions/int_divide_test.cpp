#include <stan/math/functions/int_divide.hpp>
#include <stan/math/functions/modulus.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, int_divide) {
  using stan::math::int_divide;
  EXPECT_EQ(2, int_divide(4, 2));
  EXPECT_EQ(2, int_divide(6, 3));
  EXPECT_EQ(4, int_divide(16, 4));
  EXPECT_EQ(2, int_divide(34, 17));
  EXPECT_EQ(-2, int_divide(-4, 2));
  EXPECT_EQ(-2, int_divide(-6, 3));
  EXPECT_EQ(-4, int_divide(-16, 4));
  EXPECT_EQ(-2, int_divide(34, -17));
  EXPECT_EQ(-2, int_divide(4, -2));
  EXPECT_EQ(-2, int_divide(6, -3));
  EXPECT_EQ(-4, int_divide(16, -4));  
  EXPECT_EQ(-2, int_divide(-34, 17));
    
  EXPECT_EQ(4, int_divide(17, 4));
  EXPECT_EQ(7, int_divide(22, 3));
  EXPECT_EQ(5, int_divide(22, 4));
  EXPECT_EQ(4, int_divide(34, 7));
  EXPECT_EQ(2, int_divide(44, 17));

  EXPECT_EQ(-4, int_divide(17, -4));
  EXPECT_EQ(-7, int_divide(22, -3));
  EXPECT_EQ(-5, int_divide(22, -4));
  EXPECT_EQ(-4, int_divide(34, -7));
  EXPECT_EQ(-2, int_divide(44, -17));
  
  EXPECT_EQ(-4, int_divide(-17, 4));
  EXPECT_EQ(-7, int_divide(-22, 3));
  EXPECT_EQ(-5, int_divide(-22, 4));
  EXPECT_EQ(-4, int_divide(-34, 7));
  EXPECT_EQ(-2, int_divide(-44, 17));
}

void test_divide_modulus(int a, int b) {
  using stan::math::int_divide;
  using stan::math::modulus;
  EXPECT_EQ(a, int_divide(a, b) * b + modulus(a, b)); 
}

TEST(MathFunctions, divide_modulus) {
  for(int i = 1; i < 50; i++)
    for(int j = 1; j < 50; j++)
      test_divide_modulus(i, j);
}
