#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/fma.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, fma) { 
  using stan::agrad::fvar;
  using stan::math::fma;
  fvar<double> x(0.5);
  fvar<double> y(1.2);
  fvar<double> z(1.8);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  double p = 1.4;
  double q = 2.3;

  fvar<double> a = fma(x, y, z);
  EXPECT_FLOAT_EQ(fma(0.5, 1.2, 1.8), a.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.2 + 2.0 * 0.5 + 3.0, a.d_);

  fvar<double> b = fma(p, y, z);
  EXPECT_FLOAT_EQ(fma(1.4, 1.2, 1.8), b.val_);
  EXPECT_FLOAT_EQ(2.0 * 1.4 + 3.0, b.d_);

  fvar<double> c = fma(x, p, z);
  EXPECT_FLOAT_EQ(fma(0.5, 1.4, 1.8), c.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.4 + 3.0, c.d_);

  fvar<double> d = fma(x, y, p);
  EXPECT_FLOAT_EQ(fma(0.5, 1.2, 1.4), d.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.2 + 2.0 * 0.5, d.d_);

  fvar<double> e = fma(p, q, z);
  EXPECT_FLOAT_EQ(fma(1.4, 2.3, 1.8), e.val_);
  EXPECT_FLOAT_EQ(3.0, e.d_);

  fvar<double> f = fma(x, p, q);
  EXPECT_FLOAT_EQ(fma(0.5, 1.4, 2.3), f.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.4, f.d_);

  fvar<double> g = fma(q, y, p);
  EXPECT_FLOAT_EQ(fma(2.3, 1.2, 1.4), g.val_);
  EXPECT_FLOAT_EQ(2.0 * 2.3, g.d_);
}

TEST(AgradFvarVar, fma) {
  using stan::agrad::fvar;
  using stan::agrad::var;  
  using stan::math::fma;

  fvar<var> x(2.5,1.3);
  fvar<var> y(1.7,1.5);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.7,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(2.5 * 1.5 + 1.3 * 1.7 + 1.0, a.d_.val());

  AVEC w = createAVEC(x.val_,y.val_,z.val_);
  VEC g;
  a.val_.grad(w,g);
  EXPECT_FLOAT_EQ(1.7, g[0]);
  EXPECT_FLOAT_EQ(2.5,g[1]);
  EXPECT_FLOAT_EQ(1.0,g[2]);
}

TEST(AgradFvarFvar, fma) {
  using stan::agrad::fvar;
  using stan::math::fma;

  fvar<fvar<double> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > z;
  z.val_.val_ = 1.7;

  fvar<fvar<double> > a = fma(x,y,z);

  EXPECT_FLOAT_EQ(fma(2.5,1.5,1.7), a.val_.val_);
  EXPECT_FLOAT_EQ(1.5, a.val_.d_);
  EXPECT_FLOAT_EQ(2.5, a.d_.val_);
  EXPECT_FLOAT_EQ(1, a.d_.d_);
}
