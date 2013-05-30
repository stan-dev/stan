#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/constants.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, fabs) {
  using stan::agrad::fvar;
  using std::fabs;
  using std::isnan;

  fvar<double> x(2.0,1.0);
  fvar<double> y(-3.0,2.0);

  fvar<double> a = fabs(x);
  EXPECT_FLOAT_EQ(fabs(2), a.val_);
  EXPECT_FLOAT_EQ(1.0, a.d_);

  fvar<double> b = fabs(-x);
  EXPECT_FLOAT_EQ(fabs(-2), b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<double> c = fabs(y);
  EXPECT_FLOAT_EQ(fabs(-3), c.val_);
  EXPECT_FLOAT_EQ(-2.0, c.d_);

  fvar<double> d = fabs(x * 2);
  EXPECT_FLOAT_EQ(fabs(2 * 2), d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, d.d_);

  fvar<double> e = fabs(y + 4);
  EXPECT_FLOAT_EQ(fabs(-3.0 + 4), e.val_);
  EXPECT_FLOAT_EQ(2.0, e.d_);

  fvar<double> f = fabs(x - 2);
  EXPECT_FLOAT_EQ(fabs(2.0 - 2), f.val_);
  isnan(f.d_);
 }

TEST(AgradFvarVar, fabs) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fabs;

  fvar<var> x(1.5,1.3);
  fvar<var> a = fabs(x);

  EXPECT_FLOAT_EQ(fabs(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1, g[0]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1, g[0]);
}

TEST(AgradFvarFvar, fabs) {
  using stan::agrad::fvar;
  using std::fabs;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = fabs(x);

  EXPECT_FLOAT_EQ(fabs(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;  

  a = fabs(y);
  EXPECT_FLOAT_EQ(fabs(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
