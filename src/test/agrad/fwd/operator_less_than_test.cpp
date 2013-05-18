#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar,lt) {
  using stan::agrad::fvar;
  fvar<double> v4 = 4;
  fvar<double> v5 = 5;
  double d4 = 4;
  double d5 = 5;
  
  EXPECT_TRUE(v4 < v5);
  EXPECT_TRUE(v4 < d5);
  EXPECT_TRUE(d4 < v5);
  EXPECT_TRUE(d4 < d5);

  int i4 = 4;
  int i5 = 5;

  EXPECT_TRUE(v4 < v5);
  EXPECT_TRUE(v4 < i5);
  EXPECT_TRUE(i4 < v5);
  EXPECT_TRUE(i4 < i5);
  EXPECT_TRUE(i4 < d5);
  EXPECT_TRUE(d4 < i5);
}
