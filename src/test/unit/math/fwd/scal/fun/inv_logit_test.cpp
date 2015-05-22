#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/inv_logit.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/inv_logit.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>

TEST(AgradFwdInvLogit,Fvar) {
  using stan::math::fvar;
  using stan::math::inv_logit;

  fvar<double> x(0.5,1.0);

  fvar<double> a = inv_logit(x);
  EXPECT_FLOAT_EQ(inv_logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(inv_logit(0.5) * (1 - inv_logit(0.5)), a.d_);

  fvar<double> y(-1.2,1.0);

  fvar<double> b = inv_logit(y);
  EXPECT_FLOAT_EQ(inv_logit(-1.2), b.val_);
  EXPECT_FLOAT_EQ(inv_logit(-1.2) * (1 - inv_logit(-1.2)), b.d_);

  fvar<double> z(1.5,1.0);

  fvar<double> c = inv_logit(z);
  EXPECT_FLOAT_EQ(inv_logit(1.5), c.val_);
  EXPECT_FLOAT_EQ(inv_logit(1.5) * (1 - inv_logit(1.5)), c.d_);
}

TEST(AgradFwdInvLogit,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::inv_logit;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = inv_logit(x);

  EXPECT_FLOAT_EQ(inv_logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(inv_logit(0.5) * (1 - inv_logit(0.5)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = inv_logit(y);
  EXPECT_FLOAT_EQ(inv_logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(inv_logit(0.5) * (1 - inv_logit(0.5)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct inv_logit_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv_logit(arg1);
  }
};

TEST(AgradFwdInvLogit,inv_logit_NaN) {
  inv_logit_fun inv_logit_;
  test_nan_fwd(inv_logit_,false);
}
