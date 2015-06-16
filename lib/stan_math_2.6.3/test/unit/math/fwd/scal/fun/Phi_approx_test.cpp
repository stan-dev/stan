// Phi_approx needs inv_logit in order for this to work
#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/Phi_approx.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/inv_logit.hpp>

TEST(AgradFwdPhi_approx,Fvar) {
  using stan::math::fvar;
  using stan::math::Phi_approx;
  fvar<double> x = 1.0;
  x.d_ = 1.0;
  
  fvar<double> Phi_approx_x = Phi_approx(x);

  EXPECT_FLOAT_EQ(Phi_approx(1.0), Phi_approx_x.val_);
  EXPECT_FLOAT_EQ(0.24152729,Phi_approx_x.d_);
}
TEST(AgradFwdPhi_approx, FvarDerivUnderOverFlow) {
  using stan::math::fvar;
  using stan::math::Phi_approx;

  fvar<double> x = -27.5;
  x.d_ = 1.0;
  fvar<double> Phi_approx_x = Phi_approx(x);
  EXPECT_FLOAT_EQ(0, Phi_approx_x.d_);

  fvar<double> y = 27.5;
  y.d_ = 1.0;
  fvar<double> Phi_approx_y = Phi_approx(y);
  EXPECT_FLOAT_EQ(0, Phi_approx_y.d_);
}

TEST(AgradFwdPhi_approx, FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::Phi_approx;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = Phi_approx(x);

  EXPECT_FLOAT_EQ(Phi_approx(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0.24152729, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  a = Phi_approx(y);
  EXPECT_FLOAT_EQ(Phi_approx(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0.24152729, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct Phi_approx_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return stan::math::Phi_approx(arg1);
  }
};

TEST(AgradFwdPhi_approx,Phi_approx_NaN) {
  Phi_approx_fun Phi_approx_;
  test_nan_fwd(Phi_approx_,false);
}
