#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/square.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdSquare, Fvar) {
  using stan::agrad::fvar;
  using stan::math::square;

  fvar<double> x(0.5,1.0);
  fvar<double> a = square(x);

  EXPECT_FLOAT_EQ(square(0.5), a.val_);
  EXPECT_FLOAT_EQ(2 * 0.5, a.d_);

  fvar<double> b = 3 * square(x) + x;
  EXPECT_FLOAT_EQ(3 * square(0.5) + 0.5, b.val_);
  EXPECT_FLOAT_EQ(3 * 2 * 0.5 + 1, b.d_);

  fvar<double> c = -square(x) + 5;
  EXPECT_FLOAT_EQ(-square(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-2 * 0.5, c.d_);

  fvar<double> d = -3 * square(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * square(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 * 2 * 0.5 + 5, d.d_);

  fvar<double> e = -3 * square(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * square(-0.5) + 5 * 0.5, e.val_);
  EXPECT_FLOAT_EQ(-3 * 2 * 0.5 + 5, e.d_);

  fvar<double> y(-0.5,1.0);
  fvar<double> f = square(y);
  EXPECT_FLOAT_EQ(square(-0.5), f.val_);
  EXPECT_FLOAT_EQ(2 * -0.5, f.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> g = square(z);
  EXPECT_FLOAT_EQ(square(0.0), g.val_);
  EXPECT_FLOAT_EQ(2 * 0.0, g.d_);
}   

TEST(AgradFwdSquare, FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::square;

  fvar<var> x(1.5,1.3);
  fvar<var> a = square(x);

  EXPECT_FLOAT_EQ(square(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 2.0 * (1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(2.0 * (1.5), g[0]);
}
TEST(AgradFwdSquare, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::square;

  fvar<var> x(1.5,1.3);
  fvar<var> a = square(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * 2.0, g[0]);
}

TEST(AgradFwdSquare, FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::square;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = square(x);

  EXPECT_FLOAT_EQ(square(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 * 2.0 * (1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = square(y);
  EXPECT_FLOAT_EQ(square(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 * 2.0 * (1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdSquare, FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::square;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = square(x);

  EXPECT_FLOAT_EQ(square(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 * 2.0 * (1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * 1.5, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = square(y);
  EXPECT_FLOAT_EQ(square(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 * 2.0 * (1.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * 1.5, r[0]);
}
TEST(AgradFwdSquare, FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::square;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = square(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * 2.0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = square(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * 2.0, r[0]);
}
TEST(AgradFwdSquare, FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::square;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = square(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

struct square_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return square(arg1);
  }
};

TEST(AgradFwdSquare,square_NaN) {
  square_fun square_;
  test_nan(square_,false);
}
