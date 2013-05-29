#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, sinh) {
  using stan::agrad::fvar;
  using std::sinh;
  using std::cosh;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = sinh(x);
  EXPECT_FLOAT_EQ(sinh(0.5), a.val_);
  EXPECT_FLOAT_EQ(cosh(0.5), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = sinh(y);
  EXPECT_FLOAT_EQ(sinh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(cosh(-1.2), b.d_);

  fvar<double> c = sinh(-x);
  EXPECT_FLOAT_EQ(sinh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-cosh(-0.5), c.d_);
}

TEST(AgradFvarVar, sinh) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::sinh;
  using std::cosh;

  fvar<var> x;
  x.val_ = 1.5;
  x.d_ = 1.3;
  fvar<var> a = sinh(x);

  EXPECT_FLOAT_EQ(sinh(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * cosh(1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(cosh(1.5), g[0]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFvarFvar, sinh) {
  using stan::agrad::fvar;
  using std::sinh;
  using std::cosh;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;
  x.d_.val_ = 0.0;
  x.d_.d_ = 0.0;

  fvar<fvar<double> > a = sinh(x);

  EXPECT_FLOAT_EQ(sinh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 * cosh(1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.val_.d_ = 0.0;
  y.d_.val_ = 2.0;
  y.d_.d_ = 0.0;

  a = sinh(y);
  EXPECT_FLOAT_EQ(sinh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 * cosh(1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
