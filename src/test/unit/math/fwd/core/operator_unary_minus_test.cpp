#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdOperatorUnaryMinus, Fvar) {
  using stan::math::fvar;

  fvar<double> x1(0.5,1.0);
  fvar<double> a = -x1;
  EXPECT_FLOAT_EQ(-0.5, a.val_);
  EXPECT_FLOAT_EQ(-1.0, a.d_);
}

TEST(AgradFwdOperatorUnaryMinus, FvarFvarDouble) {
  using stan::math::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > z = -x;
  EXPECT_FLOAT_EQ(-0.5, z.val_.val_);
  EXPECT_FLOAT_EQ(-1.0, z.val_.d_);
  EXPECT_FLOAT_EQ(0, z.d_.val_);
  EXPECT_FLOAT_EQ(0, z.d_.d_);
}

struct neg_fun {
  template <typename T0>
  inline T0
  operator()(T0 arg1) const {
    return (-arg1);
  }
};

TEST(AgradFwdOperatorUnaryMinus, neg_nan) {
  neg_fun neg_;

  test_nan_fwd(neg_,false);
}
