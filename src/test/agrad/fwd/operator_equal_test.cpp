#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, eq) {
  using stan::agrad::fvar;
  fvar<double> v4 = 4;
  fvar<double> v5 = 5;
  double d4 = 4;
  double d5 = 5;
  
  EXPECT_FALSE(v5 == v4);
  EXPECT_FALSE(d5 == v4);
  EXPECT_FALSE(v5 == d4);
  EXPECT_FALSE(d5 == d4);

  int i4 = 4;
  int i5 = 5;
  int i6 = 5;

  EXPECT_FALSE(i5 == v4);
  EXPECT_FALSE(v5 == i4);
  EXPECT_FALSE(i5 == i4);
  EXPECT_FALSE(d5 == i4);
  EXPECT_FALSE(i5 == d4);
  EXPECT_TRUE(i6 == i5);
  EXPECT_TRUE(i6 == v5);
}
