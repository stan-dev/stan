#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>

TEST(AgradFwdCeil,Fvar) {
  using stan::math::fvar;
  using std::ceil;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.0,2.0);

  fvar<double> a = ceil(x);
  EXPECT_FLOAT_EQ(ceil(0.5), a.val_);
  EXPECT_FLOAT_EQ(0, a.d_);

  fvar<double> b = ceil(y);
  EXPECT_FLOAT_EQ(ceil(2.0), b.val_);
   EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = ceil(2 * x);
  EXPECT_FLOAT_EQ(ceil(2 * 0.5), c.val_);
   EXPECT_FLOAT_EQ(0.0, c.d_);
}


TEST(AgradFwdCeil,FvarFvarDouble) {
  using stan::math::fvar;
  using std::ceil;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = ceil(x);

  EXPECT_FLOAT_EQ(ceil(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = ceil(y);
  EXPECT_FLOAT_EQ(ceil(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}


struct ceil_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return ceil(arg1);
  }
};

TEST(AgradFwdCeil,ceil_NaN) {
  ceil_fun ceil_;
  test_nan_fwd(ceil_,false);
}
