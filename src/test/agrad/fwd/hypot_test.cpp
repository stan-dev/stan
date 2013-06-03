#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/hypot.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, hypot) {
  using stan::agrad::fvar;
  using boost::math::hypot;
  using std::isnan;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.3,2.0);

  fvar<double> a = hypot(x, y);
  EXPECT_FLOAT_EQ(hypot(0.5, 2.3), a.val_);
  EXPECT_FLOAT_EQ((0.5 * 1.0 + 2.3 * 2.0) / hypot(0.5, 2.3), a.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> w(-2.3,2.0);
  fvar<double> b = hypot(x, z);

  EXPECT_FLOAT_EQ(0.5, b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<double> c = hypot(x, w);
  isnan(c.val_);
  isnan(c.d_);

  fvar<double> d = hypot(z, x);
  EXPECT_FLOAT_EQ(0.5, d.val_);
  EXPECT_FLOAT_EQ(1.0, d.d_);
}

TEST(AgradFvarVar, hypot) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::hypot;

  fvar<var> x(3.0,1.3);

  fvar<var> z(6.0,1.0);
  fvar<var> a = hypot(x,z);

  EXPECT_FLOAT_EQ(hypot(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * 3.0 + 6.0 * 1.0) / hypot(3.0, 6.0), a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(3.0 / hypot(3.0,6.0),g[0]);
  std::isnan(g[1]);
}

TEST(AgradFvarFvar, hypot) {
  using stan::agrad::fvar;
  using boost::math::hypot;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = hypot(x,y);

  EXPECT_FLOAT_EQ(hypot(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(3.0 / hypot(3.0,6.0), a.val_.d_);
  EXPECT_FLOAT_EQ(6.0 / hypot(3.0,6.0), a.d_.val_);
  EXPECT_FLOAT_EQ(-0.059628479, a.d_.d_);
}
