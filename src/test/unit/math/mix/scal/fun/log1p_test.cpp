#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log1p.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log1p.hpp>
#include <stan/math/rev/scal/fun/log1p.hpp>

TEST(AgradFwdLog1p,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1p;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log1p(x);

  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (1 + 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1 / (1.5), g[0]);
}
TEST(AgradFwdLog1p,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1p;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log1p(x);

  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (1 + 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.3 / (1.5 * 1.5), g[0]);
}
TEST(AgradFwdLog1p,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1p;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log1p(x);

  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1 / (1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0 / 1.5, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log1p(y);
  EXPECT_FLOAT_EQ(log1p(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(1 / (1.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0 / 1.5, r[0]);
}
TEST(AgradFwdLog1p,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1p;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log1p(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-1.0 / 2.25, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log1p(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-1.0 / 2.25, r[0]);
}
TEST(AgradFwdLog1p,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = log1p(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2 / (1.5*1.5*1.5), g[0]);
}

struct log1p_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log1p(arg1);
  }
};

TEST(AgradFwdLog1p,log1p_NaN) {
  log1p_fun log1p_;
  test_nan_mix(log1p_,false);
}
