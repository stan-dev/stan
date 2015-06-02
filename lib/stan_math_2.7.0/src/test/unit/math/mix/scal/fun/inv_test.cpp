#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/inv.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/inv.hpp>
#include <stan/math/rev/scal/fun/inv.hpp>

   

TEST(AgradFwdInv,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv;

  fvar<var> x(0.5,1.0);
  fvar<var> a = inv(x);

  EXPECT_FLOAT_EQ(inv(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(-inv(0.5) * inv(0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.0 / (0.5 * 0.5), g[0]);
}
TEST(AgradFwdInv,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv;

  fvar<var> x(0.5,1.0);
  fvar<var> a = inv(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.0 * -2.0 / (0.5 * 0.5 * 0.5), g[0]);
}

TEST(AgradFwdInv,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv;
  using std::log;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = inv(x);

  EXPECT_FLOAT_EQ(inv(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-inv(0.5) * inv(0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-inv(0.5) * inv(0.5), g[0]);
}
TEST(AgradFwdInv,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::inv;
  using std::log;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = inv(x);

  EXPECT_FLOAT_EQ(inv(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-inv(0.5) * inv(0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-1.0 * -2.0 / (0.5 * 0.5 * 0.5), g[0]);
}
TEST(AgradFwdInv,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = inv(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-96, g[0]);
}

struct inv_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv(arg1);
  }
};

TEST(AgradFwdInv,inv_NaN) {
  inv_fun inv_;
  test_nan_mix(inv_,false);
}
