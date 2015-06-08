#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/inv_cloglog.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/inv_cloglog.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>

TEST(AgradFwdInvCLogLog,Fvar) {
  using stan::math::fvar;
  using stan::math::inv_cloglog;

  fvar<double> x(0.5,1.0);

  fvar<double> a = inv_cloglog(x);
  EXPECT_FLOAT_EQ(inv_cloglog(0.5), a.val_);
  EXPECT_FLOAT_EQ(exp(0.5 -exp(0.5)), a.d_);

  fvar<double> y(-1.2,1.0);

  fvar<double> b = inv_cloglog(y);
  EXPECT_FLOAT_EQ(inv_cloglog(-1.2), b.val_);
  EXPECT_FLOAT_EQ(exp(-1.2 -exp(-1.2)), b.d_);

  fvar<double> z(1.5,2.0);

  fvar<double> c = inv_cloglog(z);
  EXPECT_FLOAT_EQ(inv_cloglog(1.5), c.val_);
  EXPECT_FLOAT_EQ(2.0 * exp(1.5 -exp(1.5)), c.d_);
}

TEST(AgradFwdInvCLogLog,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::inv_cloglog;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = inv_cloglog(x);

  EXPECT_FLOAT_EQ(inv_cloglog(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = inv_cloglog(y);
  EXPECT_FLOAT_EQ(inv_cloglog(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(exp(0.5 - exp(0.5)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct inv_cloglog_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv_cloglog(arg1);
  }
};

TEST(AgradFwdInvCLogLog,inv_cloglog_NaN) {
  inv_cloglog_fun inv_cloglog_;
  test_nan_fwd(inv_cloglog_,false);
}
