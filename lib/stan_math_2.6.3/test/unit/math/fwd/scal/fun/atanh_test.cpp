#include <gtest/gtest.h>
#include <boost/math/special_functions/atanh.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/atanh.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdAtanh,Fvar) {
  using stan::math::fvar;
  using boost::math::atanh;

  fvar<double> x(0.5,1.0);

  fvar<double> a = atanh(x);
  EXPECT_FLOAT_EQ(atanh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 - 0.5 * 0.5), a.d_);

  fvar<double> y(-0.9,1.0);

  fvar<double> b = atanh(y);
  EXPECT_FLOAT_EQ(atanh(-0.9), b.val_);
  EXPECT_FLOAT_EQ(1 / (1 - 0.9 * 0.9), b.d_);
}


TEST(AgradFwdAtanh,FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::atanh;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<double> > a = atanh(x);

  EXPECT_FLOAT_EQ(atanh(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = atanh(y);

  EXPECT_FLOAT_EQ(atanh(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}


struct atanh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return atanh(arg1);
  }
};

TEST(AgradFwdAtanh,atanh_NaN) {
  atanh_fun atanh_;
  test_nan_fwd(atanh_,false);
}
