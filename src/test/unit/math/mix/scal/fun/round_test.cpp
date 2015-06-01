#include <gtest/gtest.h>
#include <boost/math/special_functions/round.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/round.hpp>
#include <stan/math/rev/scal/fun/round.hpp>

TEST(AgradFwdRound, FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::round;

  fvar<var> x(1.5,1.3);
  fvar<var> a = round(x);

  EXPECT_FLOAT_EQ(round(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdRound, FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::round;

  fvar<var> x(1.5,1.3);
  fvar<var> a = round(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdRound, FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::round;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = round(x);

  EXPECT_FLOAT_EQ(round(1.5), a.val_.val_.val());
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

  fvar<fvar<var> > b = round(y);
  EXPECT_FLOAT_EQ(round(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdRound, FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::round;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = round(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = round(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdRound, FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::round;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > b = round(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}

struct round_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return round(arg1);
  }
};

TEST(AgradFwdRound,round_NaN) {
  round_fun round_;
  test_nan_mix(round_,false);
}
