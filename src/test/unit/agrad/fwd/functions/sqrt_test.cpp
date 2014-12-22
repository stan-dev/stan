#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdSqrt, Fvar) {
  using stan::agrad::fvar;
  using std::sqrt;
  using std::isnan;

  fvar<double> x(0.5,1.0);
  fvar<double> a = sqrt(x);

  EXPECT_FLOAT_EQ(sqrt(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (2 * sqrt(0.5)), a.d_);

  fvar<double> b = 3 * sqrt(x) + x;
  EXPECT_FLOAT_EQ(3 * sqrt(0.5) + 0.5, b.val_);
  EXPECT_FLOAT_EQ(3 / (2 * sqrt(0.5)) + 1, b.d_);

  fvar<double> c = -sqrt(x) + 5;
  EXPECT_FLOAT_EQ(-sqrt(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (2 * sqrt(0.5)), c.d_);

  fvar<double> d = -3 * sqrt(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * sqrt(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (2 * sqrt(0.5)) + 5, d.d_);

  fvar<double> y(-0.5,1.0);
  fvar<double> e = sqrt(-y);
  EXPECT_FLOAT_EQ(sqrt(0.5), e.val_);
  EXPECT_FLOAT_EQ(-1 / (2 * sqrt(0.5)), e.d_); 

  fvar<double> f = sqrt(y);
  isnan(f.val_);
  isnan(f.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> g = sqrt(z);
  EXPECT_FLOAT_EQ(sqrt(0.0), g.val_);
  isnan(g.d_); 
}

TEST(AgradFwdSqrt, FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::sqrt;

  fvar<var> x(1.5,1.3);
  fvar<var> a = sqrt(x);

  EXPECT_FLOAT_EQ(sqrt(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 0.5 / sqrt(1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0.5 / sqrt(1.5), g[0]);
}
TEST(AgradFwdSqrt, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::sqrt;

  fvar<var> x(1.5,1.3);
  fvar<var> a = sqrt(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.5 * -0.5 * 1.3 / sqrt(1.5) / 1.5, g[0]);
}

TEST(AgradFwdSqrt, FvarFvarDouble) {
  using stan::agrad::fvar;
  using std::sqrt;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = sqrt(x);

  EXPECT_FLOAT_EQ(sqrt(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 * 0.5 / sqrt(1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = sqrt(y);
  EXPECT_FLOAT_EQ(sqrt(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 * 0.5 / sqrt(1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

TEST(AgradFwdSqrt, FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = sqrt(x);

  EXPECT_FLOAT_EQ(sqrt(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 * 0.5 / sqrt(1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.5 / sqrt(1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = sqrt(y);
  EXPECT_FLOAT_EQ(sqrt(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 * 0.5 / sqrt(1.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.5 / sqrt(1.5), r[0]);
}
TEST(AgradFwdSqrt, FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = sqrt(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.5 * 2.0 * -0.5 / 1.5 / sqrt(1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = sqrt(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.5 * -2.0 * 0.5 / 1.5 / sqrt(1.5), r[0]);
}
TEST(AgradFwdSqrt, FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = sqrt(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.1360827634879543387887380041503, g[0]);
}

struct sqrt_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return sqrt(arg1);
  }
};

TEST(AgradFwdSqrt,sqrt_NaN) {
  sqrt_fun sqrt_;
  test_nan(sqrt_,false);
}
