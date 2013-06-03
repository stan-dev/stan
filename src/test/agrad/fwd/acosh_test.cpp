#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/acosh.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, acosh) {
  using stan::agrad::fvar;
  using boost::math::acosh;
  using std::sqrt;
  using std::isnan;

  fvar<double> x(1.5,1.0);

  fvar<double> a = acosh(x);
  EXPECT_FLOAT_EQ(acosh(1.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(-1 + (1.5) * (1.5)), a.d_);

  fvar<double> y(-1.2,1.0);

  fvar<double> b = acosh(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> z(0.5,1.0);

  fvar<double> c = acosh(z);
  isnan(c.val_);
  isnan(c.d_);
}

TEST(AgradFvarVar, acosh) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::acosh;

  fvar<var> x(1.5,1.3);
  fvar<var> a = acosh(x);

  EXPECT_FLOAT_EQ(acosh(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / sqrt(-1.0 + 1.5 * 1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / sqrt(-1.0 + 1.5 * 1.5), g[0]);
}

TEST(AgradFvarFvar, acosh) {
  using stan::agrad::fvar;
  using boost::math::acosh;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = acosh(x);

  EXPECT_FLOAT_EQ(acosh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(-1.0 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = acosh(y);
  EXPECT_FLOAT_EQ(acosh(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(-1.0 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
