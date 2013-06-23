#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar,ne) {
  using stan::agrad::fvar;
  fvar<double> v4 = 4;
  fvar<double> v5 = 5;
  double d4 = 4;
  double d5 = 5;
  
  EXPECT_TRUE(v5 != v4);
  EXPECT_TRUE(d5 != v4);
  EXPECT_TRUE(v5 != d4);
  EXPECT_TRUE(d5 != d4);

  int i4 = 4;
  int i5 = 5;
  int i6 = 5;

  EXPECT_TRUE(i5 != v4);
  EXPECT_TRUE(v5 != i4);
  EXPECT_TRUE(i5 != i4);
  EXPECT_TRUE(d5 != i4);
  EXPECT_TRUE(i5 != d4);
  EXPECT_FALSE(i6 != i5);
  EXPECT_FALSE(i6 != v5);
}
