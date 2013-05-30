#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, bessel_first_kind) {
  using stan::agrad::fvar;
  using stan::agrad::bessel_first_kind;

  fvar<double> a(4.0,1.0);
  int b = 0;
  fvar<double> x = bessel_first_kind(b,a);
  EXPECT_FLOAT_EQ(-0.39714980986384737228659076845169804197561868528938, 
                  x.val_);
  EXPECT_FLOAT_EQ(0.0660433280235491361431854208032750287274234195317,
                  x.d_);

  fvar<double> c(-3.0,2.0);

  x = bessel_first_kind(1, c);
  EXPECT_FLOAT_EQ(-0.33905895852593645892551459720647889697308041819800,
                  x.val_);
  EXPECT_FLOAT_EQ(-0.7461432154878245145319857900923154716212709191545920,
                  x.d_);
}
