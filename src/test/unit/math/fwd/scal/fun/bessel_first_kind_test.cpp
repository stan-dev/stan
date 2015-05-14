#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/scal/fun/bessel_first_kind.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdBesselFirstKind,Fvar) {
  using stan::math::fvar;
  using stan::math::bessel_first_kind;

  fvar<double> a(4.0,1.0);
  int b = 0;
  fvar<double> x = bessel_first_kind(b,a);
  EXPECT_FLOAT_EQ(-0.397149809863847372, x.val_);
  EXPECT_FLOAT_EQ(0.0660433280235491361, x.d_);

  fvar<double> c(-3.0,2.0);

  x = bessel_first_kind(1, c);
  EXPECT_FLOAT_EQ(-0.3390589585259364589255, x.val_);
  EXPECT_FLOAT_EQ(-0.7461432154878245145319, x.d_);
}


TEST(AgradFwdBesselFirstKind,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::bessel_first_kind;

  fvar<fvar<double> > x;
  x.val_.val_ = -3.0;
  x.val_.d_ = 2.0;
  fvar<fvar<double> > y;
  y.val_.val_ = -3.0;
  y.d_.val_ = 2.0;

  fvar<fvar<double> > a = bessel_first_kind(1,y);

  EXPECT_FLOAT_EQ(bessel_first_kind(1,-3.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(-0.7461432154878245145319, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > b = bessel_first_kind(1, x);

  EXPECT_FLOAT_EQ(bessel_first_kind(1,-3.0), b.val_.val_);
  EXPECT_FLOAT_EQ(-0.7461432154878245145319, b.val_.d_);
  EXPECT_FLOAT_EQ(0, b.d_.val_);
  EXPECT_FLOAT_EQ(0, b.d_.d_);
}

struct bessel_first_kind_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return bessel_first_kind(1,arg1);
  }
};

TEST(AgradFwdBesselFirstKind,bessel_first_kind_NaN) {
  bessel_first_kind_fun bessel_first_kind_;
  test_nan_fwd(bessel_first_kind_,true);
}
