#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log1m.hpp>
#include <stan/math/rev/scal/fun/log1m.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>

TEST(AgradFwdLog1m,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log1m(x);

  EXPECT_FLOAT_EQ(log1m(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(-1.3 / (0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-1 / (0.5), g[0]);
}
TEST(AgradFwdLog1m,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log1m(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.3 / (0.25), g[0]);
}

TEST(AgradFwdLog1m,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log1m(x);

  EXPECT_FLOAT_EQ(log1m(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-1 / (0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-1 / 0.5, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log1m(y);
  EXPECT_FLOAT_EQ(log1m(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(-1 / (0.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-1 / 0.5, r[0]);
}
TEST(AgradFwdLog1m,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log1m(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-1 / 0.25, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log1m(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-1 / 0.25, r[0]);
}
TEST(AgradFwdLog1m,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = log1m(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-2 / (0.5*0.5*0.5), g[0]);
}

struct log1m_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log1m(arg1);
  }
};

TEST(AgradFwdLog1m,log1m_NaN) {
  log1m_fun log1m_;
  test_nan_mix(log1m_,false);
}
