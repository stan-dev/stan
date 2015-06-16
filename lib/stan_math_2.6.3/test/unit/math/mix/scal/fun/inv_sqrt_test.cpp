#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/inv_sqrt.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/inv_sqrt.hpp>
#include <stan/math/rev/scal/fun/inv_sqrt.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>

   

TEST(AgradFwdInvSqrt,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv_sqrt;

  fvar<var> x(0.5,1.0);
  fvar<var> a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.5 / (0.5 * std::sqrt(0.5)), g[0]);
}
TEST(AgradFwdInvSqrt,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv_sqrt;

  fvar<var> x(0.5,1.0);
  fvar<var> a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.5 * -1.5 / (0.25 * std::sqrt(0.5)), g[0]);
}

TEST(AgradFwdInvSqrt,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv_sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), g[0]);
}
TEST(AgradFwdInvSqrt,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv_sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.5 * -1.5 / (0.25 * std::sqrt(0.5)), g[0]);
}
TEST(AgradFwdInvSqrt,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv_sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = inv_sqrt(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-21.2132034355964257320253308631, g[0]);
}

struct inv_sqrt_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv_sqrt(arg1);
  }
};

TEST(AgradFwdInvSqrt,inv_sqrt_NaN) {
  inv_sqrt_fun inv_sqrt_;
  test_nan_mix(inv_sqrt_,false);
}
