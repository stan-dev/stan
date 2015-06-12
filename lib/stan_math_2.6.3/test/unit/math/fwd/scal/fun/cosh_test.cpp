#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/cosh.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/sinh.hpp>

TEST(AgradFwdCosh,Fvar) {
  using stan::math::fvar;
  using std::sinh;
  using std::cosh;

  fvar<double> x(0.5,1.0);

  fvar<double> a = cosh(x);
  EXPECT_FLOAT_EQ(cosh(0.5), a.val_);
  EXPECT_FLOAT_EQ(sinh(0.5), a.d_);

  fvar<double> y(-1.2,1.0);

  fvar<double> b = cosh(y);
  EXPECT_FLOAT_EQ(cosh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(sinh(-1.2), b.d_);

  fvar<double> c = cosh(-x);
  EXPECT_FLOAT_EQ(cosh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-sinh(-0.5), c.d_);
}


TEST(AgradFwdCosh,FvarFvarDouble) {
  using stan::math::fvar;
  using std::sinh;
  using std::cosh;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = cosh(x);

  EXPECT_FLOAT_EQ(cosh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 * sinh(1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = cosh(y);
  EXPECT_FLOAT_EQ(cosh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 * sinh(1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct cosh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return cosh(arg1);
  }
};

TEST(AgradFwdCosh,cosh_NaN) {
  cosh_fun cosh_;
  test_nan_fwd(cosh_,false);
}
