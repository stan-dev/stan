#include <gtest/gtest.h>
#include <boost/math/special_functions/acosh.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/acosh.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdAcosh,Fvar) {
  using stan::math::fvar;
  using boost::math::acosh;
  using std::sqrt;
  using std::isnan;

  fvar<double> x(1.5,1.0);

  fvar<double> a = acosh(x);
  EXPECT_FLOAT_EQ(acosh(1.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(-1 + (1.5) * (1.5)), a.d_);

  fvar<double> y(-1.2,1.0);

  fvar<double> b = acosh(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> z(0.5,1.0);

  fvar<double> c = acosh(z);
  isnan(c.val_);
  isnan(c.d_);
}


TEST(AgradFwdAcosh,FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::acosh;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = acosh(x);

  EXPECT_FLOAT_EQ(acosh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(-1.0 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = acosh(y);
  EXPECT_FLOAT_EQ(acosh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(-1.0 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct acosh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return acosh(arg1);
  }
};

TEST(AgradFwdAcosh,acosh_NaN) {
  acosh_fun acosh_;
  test_nan_fwd(acosh_,false);
}
