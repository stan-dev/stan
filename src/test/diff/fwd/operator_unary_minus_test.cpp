#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, operatorUnaryMinus){
  using stan::agrad::fvar;

  fvar<double> x1(0.5);
  fvar<double> x2(0.4);
  x1.d_ = 1.0;
  
  fvar<double> a = -x1;
  EXPECT_FLOAT_EQ(-0.5, a.val_);
  EXPECT_FLOAT_EQ(-1.0, a.d_);
}
