#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar,multiply_log) {
  using stan::agrad::fvar;
  using std::isnan;
  using std::log;
  using stan::math::multiply_log;

  fvar<double> x(0.5);
  fvar<double> y(1.2);
  fvar<double> z(-0.4);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  double w = 0.0;
  double v = 1.3;

  fvar<double> a = multiply_log(x, y);
  EXPECT_FLOAT_EQ(multiply_log(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ(1.0 * log(1.2) + 0.5 * 2.0 / 1.2, a.d_);

  fvar<double> b = multiply_log(x,z);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = multiply_log(x, v);
  EXPECT_FLOAT_EQ(multiply_log(0.5, 1.3), c.val_);
  EXPECT_FLOAT_EQ(log(1.3), c.d_);

  fvar<double> d = multiply_log(v, x);
  EXPECT_FLOAT_EQ(multiply_log(1.3, 0.5), d.val_);
  EXPECT_FLOAT_EQ(1.3 * 1.0 / 0.5, d.d_);

  fvar<double> e = multiply_log(x, w);
  isnan(e.val_);
  isnan(e.d_);
}

// TEST(AgradFvarVar, multiply_log) {
//   using stan::agrad::fvar;
//   using stan::agrad::var;
//   using std::log;
//   using stan::math::multiply_log;

//   fvar<var> x;
//   x.val_ = 1.5;
//   x.d_ = 1.3;

//   fvar<var> z;
//   x.val_ = 1.8;
//   x.d_ = 1.1;
//   fvar<var> a = multiply_log(x,z);

//   EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val());
//   EXPECT_FLOAT_EQ(log(1.8) * 1.3 + 1.5 * 1.1 / 1.8, a.d_.val());

//   AVEC y = createAVEC(x.val_);
//   VEC g;
//   a.val_.grad(y,g);
//   EXPECT_FLOAT_EQ(log(1.8) / 1.5 + log(1.5) / 1.8, g[0]);

//   y = createAVEC(x.d_);
//   a.d_.grad(y,g);
//   EXPECT_FLOAT_EQ(0, g[0]);
// }

TEST(AgradFvarFvar, multiply_log) {
  using stan::agrad::fvar;
  using std::log;
  using stan::math::multiply_log;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.3;
  x.d_.val_ = 0.0;
  x.d_.d_ = 0.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.8;
  y.val_.d_ = 0;
  y.d_.val_ = 1.1;
  y.d_.d_ = 0.0;

  fvar<fvar<double> > a = multiply_log(x,y);

  EXPECT_FLOAT_EQ(multiply_log(1.5,1.8), a.val_.val_);
  EXPECT_FLOAT_EQ(log(1.8) * 1.3, a.val_.d_);
  EXPECT_FLOAT_EQ(1.5 / 1.8 * 1.1, a.d_.val_);
  EXPECT_FLOAT_EQ(143.0 / 180.0, a.d_.d_);
}
