#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/log2.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdLog2,Fvar) {
  using stan::agrad::fvar;
  using std::log;
  using std::isnan;
  using stan::math::log2;

  fvar<double> x(0.5,1.0);
  
  fvar<double> a = log2(x);
  EXPECT_FLOAT_EQ(log2(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(2)), a.d_);

  fvar<double> b = 2 * log2(x) + 4;
  EXPECT_FLOAT_EQ(2 * log2(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (0.5 * log(2)), b.d_);

  fvar<double> c = -log2(x) + 5;
  EXPECT_FLOAT_EQ(-log2(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (0.5 * log(2)), c.d_);

  fvar<double> d = -3 * log2(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * log2(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (0.5 * log(2)) + 5, d.d_);

  fvar<double> y(-0.5,1.0);
  fvar<double> e = log2(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> f = log2(z);
  isnan(f.val_);
  isnan(f.d_);
}

TEST(AgradFwdLog2,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log2(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.3 / (0.25 * log(2)), g[0]);
}
TEST(AgradFwdLog2,FvarFvarDouble) {
  using stan::agrad::fvar;
  using std::log;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = log2(x);

  EXPECT_FLOAT_EQ(log2(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(2)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = log2(y);
  EXPECT_FLOAT_EQ(log2(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(2)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdLog2,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;

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
  test_nan(log2_,false);
}
