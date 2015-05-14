#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/bessel_second_kind.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/scal/fun/bessel_second_kind.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdBesselSecondKind,Fvar) {
  using stan::math::fvar;
  using stan::math::bessel_second_kind;

  fvar<double> a(4.0,1.0);
  int b = 0;
  fvar<double> x = bessel_second_kind(b,a);
  EXPECT_FLOAT_EQ(-0.01694073932506499190, x.val_);
  EXPECT_FLOAT_EQ(-0.39792571055710000525, x.d_);

  fvar<double> c(3.0,2.0);

  x = bessel_second_kind(1, c);
  EXPECT_FLOAT_EQ(0.32467442479179997, x.val_);
  EXPECT_FLOAT_EQ(0.53725040349771411, x.d_);

  EXPECT_THROW(bessel_second_kind(0, -a), std::domain_error);
}


TEST(AgradFwdBesselSecondKind,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::bessel_second_kind;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 2.0;

  fvar<fvar<double> > a = stan::math::bessel_second_kind(1,y);

  EXPECT_FLOAT_EQ(stan::math::bessel_second_kind(1,3.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0.53725040349771411, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > b = stan::math::bessel_second_kind(1, x);

  EXPECT_FLOAT_EQ(stan::math::bessel_second_kind(1,3.0), b.val_.val_);
  EXPECT_FLOAT_EQ(0.53725040349771411, b.val_.d_);
  EXPECT_FLOAT_EQ(0, b.d_.val_);
  EXPECT_FLOAT_EQ(0, b.d_.d_);
}

struct bessel_second_kind_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return bessel_second_kind(1,arg1);
  }
};

TEST(AgradFwdBesselSecondKind,bessel_second_kind_NaN) {
  bessel_second_kind_fun bessel_second_kind_;
  test_nan_fwd(bessel_second_kind_,false);
}
