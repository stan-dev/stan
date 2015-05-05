#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log2.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log2.hpp>
#include <stan/math/rev/scal/fun/log2.hpp>


TEST(AgradFwdLog2,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::log;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log2(x);

  EXPECT_FLOAT_EQ(log2(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (0.5 * log(2)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(2)), g[0]);
}
TEST(AgradFwdLog2,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::log;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log2(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.3 / (0.25 * log(2)), g[0]);
}
TEST(AgradFwdLog2,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::log;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log2(x);

  EXPECT_FLOAT_EQ(log2(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1 / (0.5 * log(2)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0 / (0.5 * log(2)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log2(y);
  EXPECT_FLOAT_EQ(log2(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(1 / (0.5 * log(2)), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0 / (0.5 * log(2)), r[0]);
}
TEST(AgradFwdLog2,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::log;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log2(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-1.0 / (0.25 * log(2)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log2(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-1.0 / (0.5 * 0.5 * log(2)), r[0]);
}
TEST(AgradFwdLog2,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = log2(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(23.0831206542234145177587948960, g[0]);
}

struct log2_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log2(arg1);
  }
};

TEST(AgradFwdLog2,log2_NaN) {
  log2_fun log2_;
  test_nan_mix(log2_,false);
}
