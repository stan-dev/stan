#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, sin) {
  using stan::agrad::fvar;
  using stan::agrad::bessel_first_kind;

  fvar<double> a;
  a.val_ = 4.0;
  a.d_ = 1.0;

  fvar<double> x = bessel_first_kind(0,a);
  EXPECT_FLOAT_EQ(-0.39714980986384737228659076845169804197561868528938, 
                  x.val_);
  EXPECT_FLOAT_EQ(0.0660433280235491361431854208032750287274234195317,
                  x.d_);
}
