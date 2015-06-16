#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>

TEST(AgradFwdOperatorPlusPlus, Fvar) {
  using stan::math::fvar;

  fvar<double> x(0.5,1.0);
  x++;

  EXPECT_FLOAT_EQ(0.5 + 1.0, x.val_);
  EXPECT_FLOAT_EQ(1.0, x.d_);

  fvar<double> y(-0.5,1.0);
  y++;

  EXPECT_FLOAT_EQ(-0.5 + 1.0, y.val_);
  EXPECT_FLOAT_EQ(1.0, y.d_);
}

TEST(AgradFwdOperatorPlusPlus, FvarFvarDouble) {
  using stan::math::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  x++;
  EXPECT_FLOAT_EQ(0.5 + 1.0, x.val_.val_);
  EXPECT_FLOAT_EQ(1, x.val_.d_);
  EXPECT_FLOAT_EQ(0, x.d_.val_);
  EXPECT_FLOAT_EQ(0, x.d_.d_);
}

struct plus_plus_fun {
  template <typename T0>
  inline T0
  operator()(T0 arg1) const {
    return (arg1++);
  }
};

TEST(AgradFwdOperatorPlusPlus, plus_plus_nan) {
  plus_plus_fun plus_plus_;

  test_nan_fwd(plus_plus_,false);
}
