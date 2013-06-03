#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, fdim) {
  using stan::agrad::fvar;
  using stan::math::fdim;
  using std::isnan;
  using std::floor;

  fvar<double> x(2.0,1.0);
  fvar<double> y(-3.0,2.0);

  fvar<double> a = fdim(x, y);
  EXPECT_FLOAT_EQ(fdim(2.0, -3.0), a.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.0 + 2.0 * -floor(2.0 / -3.0), a.d_);

  fvar<double> b = fdim(2 * x, y);
  EXPECT_FLOAT_EQ(fdim(2 * 2.0, -3.0), b.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 * 1.0 + 2.0 * -floor(4.0 / -3.0), b.d_);

  fvar<double> c = fdim(y, x);
  EXPECT_FLOAT_EQ(fdim(-3.0, 2.0), c.val_);
  EXPECT_FLOAT_EQ(0.0, c.d_);

  fvar<double> d = fdim(x, x);
  EXPECT_FLOAT_EQ(fdim(2.0, 2.0), d.val_);
  EXPECT_FLOAT_EQ(0.0, d.d_);

  double z = 1.0;

  fvar<double> e = fdim(x, z);
  EXPECT_FLOAT_EQ(fdim(2.0, 1.0), e.val_);
  EXPECT_FLOAT_EQ(1.0, e.d_);

  fvar<double> f = fdim(z, x);
  EXPECT_FLOAT_EQ(fdim(1.0, 2.0), f.val_);
  EXPECT_FLOAT_EQ(0.0, f.d_);
 }

TEST(AgradFvarVar, fdim) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::fdim;
  using std::floor;
  using std::isnan;

  fvar<var> x(2.5,1.3);
  fvar<var> z(1.5,1.0);
  fvar<var> a = fdim(x,z);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val());
  isnan(a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  isnan(g[0]);
  isnan(g[1]);
}

TEST(AgradFvarFvar, fdim) {
  using stan::agrad::fvar;
  using stan::math::fdim;
  using std::floor;

  fvar<fvar<double> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = fdim(x,y);

  EXPECT_FLOAT_EQ(fdim(2.5,1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.val_.d_);
  EXPECT_FLOAT_EQ(-1, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
