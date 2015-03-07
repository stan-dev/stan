#include <stan/math/prim/scal/fun/divide.hpp>
#include <stan/math/prim/scal/fun/modulus.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, divide) {
  using stan::math::divide;
  EXPECT_EQ(2, divide(4, 2));
  EXPECT_EQ(2, divide(6, 3));
  EXPECT_EQ(4, divide(16, 4));
  EXPECT_EQ(2, divide(34, 17));
  EXPECT_EQ(-2, divide(-4, 2));
  EXPECT_EQ(-2, divide(-6, 3));
  EXPECT_EQ(-4, divide(-16, 4));
  EXPECT_EQ(-2, divide(34, -17));
  EXPECT_EQ(-2, divide(4, -2));
  EXPECT_EQ(-2, divide(6, -3));
  EXPECT_EQ(-4, divide(16, -4));  
  EXPECT_EQ(-2, divide(-34, 17));
    
  EXPECT_EQ(4, divide(17, 4));
  EXPECT_EQ(7, divide(22, 3));
  EXPECT_EQ(5, divide(22, 4));
  EXPECT_EQ(4, divide(34, 7));
  EXPECT_EQ(2, divide(44, 17));

  EXPECT_EQ(-4, divide(17, -4));
  EXPECT_EQ(-7, divide(22, -3));
  EXPECT_EQ(-5, divide(22, -4));
  EXPECT_EQ(-4, divide(34, -7));
  EXPECT_EQ(-2, divide(44, -17));
  
  EXPECT_EQ(-4, divide(-17, 4));
  EXPECT_EQ(-7, divide(-22, 3));
  EXPECT_EQ(-5, divide(-22, 4));
  EXPECT_EQ(-4, divide(-34, 7));
  EXPECT_EQ(-2, divide(-44, 17));
}

void test_divide_modulus(int a, int b) {
  using stan::math::divide;
  using stan::math::modulus;
  EXPECT_EQ(a, divide(a, b) * b + modulus(a, b)); 
}

TEST(MathFunctions, divide_modulus) {
  for(int i = 1; i < 50; i++)
    for(int j = 1; j < 50; j++)
      test_divide_modulus(i, j);
}
