#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/trunc.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/scal/fun/trunc.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradFwdTrunc, FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::trunc;

  fvar<var> x(1.5,1.3);
  fvar<var> a = trunc(x);

  EXPECT_FLOAT_EQ(trunc(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdTrunc, FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::trunc;

  fvar<var> x(1.5,1.3);
  fvar<var> a = trunc(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdTrunc, FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::trunc;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = trunc(x);

  EXPECT_FLOAT_EQ(trunc(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = trunc(y);
  EXPECT_FLOAT_EQ(trunc(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdTrunc, FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::trunc;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = trunc(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = trunc(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdTrunc, FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::trunc;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = trunc(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdTrunc, FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::trunc;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = trunc(x);

  EXPECT_FLOAT_EQ(trunc(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = trunc(y);
  EXPECT_FLOAT_EQ(trunc(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct trunc_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return trunc(arg1);
  }
};

TEST(AgradFwdTrunc,trunc_NaN) {
  trunc_fun trunc_;
  test_nan_mix(trunc_,false);
}
