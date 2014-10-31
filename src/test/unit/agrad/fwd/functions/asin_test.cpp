#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <stan/math/constants.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdAsin,Fvar) {
  using stan::agrad::fvar;
  using std::asin;
  using std::isnan;
  using std::sqrt;
  using stan::math::INFTY;

  fvar<double> x(0.5,1.0);
  
  fvar<double> a = asin(x);
  EXPECT_FLOAT_EQ(asin(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 - 0.5 * 0.5), a.d_);

  fvar<double> b = 2 * asin(x) + 4;
  EXPECT_FLOAT_EQ(2 * asin(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / sqrt(1 - 0.5 * 0.5), b.d_);

  fvar<double> c = -asin(x) + 5;
  EXPECT_FLOAT_EQ(-asin(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / sqrt(1 - 0.5 * 0.5), c.d_);

  fvar<double> d = -3 * asin(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * asin(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / sqrt(1 - 0.5 * 0.5) + 5, d.d_);

  fvar<double> y(3.4,1.0);
  fvar<double> e = asin(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(1.0,1.0);
  fvar<double> f = asin(z);
  EXPECT_FLOAT_EQ(asin(1.0), f.val_);
  EXPECT_FLOAT_EQ(INFTY, f.d_);

  fvar<double> z2(1.0+stan::math::EPSILON,1.0);
  fvar<double> f2 = asin(z2);
  EXPECT_TRUE(boost::math::isnan(f2.val_));
  EXPECT_TRUE(boost::math::isnan(f2.d_));

  fvar<double> z3(-1.0-stan::math::EPSILON,1.0);
  fvar<double> f3 = asin(z3);
  EXPECT_TRUE(boost::math::isnan(f3.val_));
  EXPECT_TRUE(boost::math::isnan(f3.d_));
}

TEST(AgradFwdAsin,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::asin;

  fvar<var> x(0.5,0.3);
  fvar<var> a = asin(x);

  EXPECT_FLOAT_EQ(asin(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(0.3 / sqrt(1.0 - 0.5 * 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 - 0.5 * 0.5), g[0]);
}

TEST(AgradFwdAsin,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::asin;

  fvar<var> x(0.5,0.3);
  fvar<var> a = asin(x);

  EXPECT_FLOAT_EQ(asin(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(0.3 / sqrt(1.0 - 0.5 * 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.3 * 0.76980033, g[0]);
}

TEST(AgradFwdAsin,FvarFvarDouble) {
  using stan::agrad::fvar;
  using std::asin;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = asin(x);

  EXPECT_FLOAT_EQ(asin(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 - 0.5 * 0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 2.0;

  a = asin(y);
  EXPECT_FLOAT_EQ(asin(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 - 0.5 * 0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

TEST(AgradFwdAsin,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::asin;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = asin(x);

  EXPECT_FLOAT_EQ(asin(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 - 0.5 * 0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 - 0.5 * 0.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = asin(y);
  EXPECT_FLOAT_EQ(asin(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 - 0.5 * 0.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 - 0.5 * 0.5), r[0]);
}

TEST(AgradFwdAsin,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::asin;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = asin(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * 0.76980033, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = asin(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * 0.76980033, r[0]);
}
TEST(AgradFwdAsin,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::asin;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = asin(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(3.07920143567800, g[0]);
}
struct asin_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return asin(arg1);
  }
};

TEST(AgradFwdAsin,asin_NaN) {
  asin_fun asin_;
  test_nan(asin_,false);
}
