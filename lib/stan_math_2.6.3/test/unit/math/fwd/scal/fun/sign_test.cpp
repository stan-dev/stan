#include <stan/math/prim/scal/fun/sign.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdSign, Fvar) {
  using stan::math::fvar;
  fvar<double> x;
  x = 0;
  EXPECT_EQ(0, stan::math::sign(x));
  x = 0.0000001;
  EXPECT_EQ(1, stan::math::sign(x));
  x = -0.001;
  EXPECT_EQ(-1, stan::math::sign(x));
}

TEST(AgradFwdSign, FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::sign;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;
  fvar<fvar<double> > a = sign(x);

  EXPECT_FLOAT_EQ(sign(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = sign(y);
  EXPECT_FLOAT_EQ(sign(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
