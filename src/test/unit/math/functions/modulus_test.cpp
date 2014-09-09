#include <stan/math/functions/modulus.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, modulus) {
  using stan::math::modulus;
  EXPECT_EQ(0, modulus(4, 2));
  EXPECT_EQ(0, modulus(6, 3));
  EXPECT_EQ(0, modulus(16, 4));
  EXPECT_EQ(0, modulus(34, 17));
  EXPECT_EQ(0, modulus(-4, 2));
  EXPECT_EQ(0, modulus(-6, 3));
  EXPECT_EQ(0, modulus(-16, 4));
  EXPECT_EQ(0, modulus(34, -17));
  EXPECT_EQ(0, modulus(4, -2));
  EXPECT_EQ(0, modulus(6, -3));
  EXPECT_EQ(0, modulus(16, -4));  
  EXPECT_EQ(0, modulus(-34, 17));
    
  EXPECT_EQ(1, modulus(17, 4));
  EXPECT_EQ(1, modulus(22, 3));
  EXPECT_EQ(2, modulus(22, 4));
  EXPECT_EQ(6, modulus(34, 7));
  EXPECT_EQ(10, modulus(44, 17));

  EXPECT_EQ(1, modulus(17, -4));
  EXPECT_EQ(1, modulus(22, -3));
  EXPECT_EQ(2, modulus(22, -4));
  EXPECT_EQ(6, modulus(34, -7));
  EXPECT_EQ(10, modulus(44, -17));
  
  EXPECT_EQ(-1, modulus(-17, 4));
  EXPECT_EQ(-1, modulus(-22, 3));
  EXPECT_EQ(-2, modulus(-22, 4));
  EXPECT_EQ(-6, modulus(-34, 7));
  EXPECT_EQ(-10, modulus(-44, 17));
}
