#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, operatorMinusMinus){
  using stan::agrad::fvar;

  fvar<double> x(0.5);
  x.d_ = 1.0;
  x--;

  EXPECT_FLOAT_EQ(0.5 - 1.0, x.val_);
  EXPECT_FLOAT_EQ(1.0, x.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  y--;

  EXPECT_FLOAT_EQ(-0.5 - 1.0, y.val_);
  EXPECT_FLOAT_EQ(1.0, y.d_);
}
