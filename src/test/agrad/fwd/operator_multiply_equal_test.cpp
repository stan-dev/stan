#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, operatorMultiplyEqual) {
  using stan::agrad::fvar;

  fvar<double> a(0.5,1.0);
  fvar<double> x1(0.4,2.0);
  a *= x1;
  EXPECT_FLOAT_EQ(0.5 * 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 * 0.4 + 2.0 * 0.5, a.d_);

  fvar<double> b(0.5,1.0);
  fvar<double> x2(0.4,2.0);
  b *= -x2;
  EXPECT_FLOAT_EQ(0.5 * -0.4, b.val_);
  EXPECT_FLOAT_EQ(1.0 * -0.4 + -2.0 * 0.5, b.d_);

  fvar<double> c(0.6,3.0);
  double x3(0.3);
  c *= x3;
  EXPECT_FLOAT_EQ(0.6 * 0.3, c.val_);
  EXPECT_FLOAT_EQ(3.0, c.d_);

  fvar<double> d(0.5,1.0);
  fvar<double> x4(-0.4,2.0);
  d *= x4;
  EXPECT_FLOAT_EQ(0.5 * -0.4, d.val_);
  EXPECT_FLOAT_EQ(1.0 * -0.4 + 2.0 * 0.5, d.d_);
}

TEST(AgradFvarVar, operatorMultiplyEqual) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> z(0.5,1.3);
  x *= z;

  EXPECT_FLOAT_EQ(0.25, x.val_.val());
  EXPECT_FLOAT_EQ(1.3, x.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  x.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1, g[0]);
  EXPECT_FLOAT_EQ(0.5, g[1]);
}

TEST(AgradFvarFvar, operatorMultiplyEqual) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  x *= y;
  EXPECT_FLOAT_EQ(0.25, x.val_.val_);
  EXPECT_FLOAT_EQ(0.5, x.val_.d_);
  EXPECT_FLOAT_EQ(0.5, x.d_.val_);
  EXPECT_FLOAT_EQ(1, x.d_.d_);
}
