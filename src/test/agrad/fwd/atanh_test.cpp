#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/atanh.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, atanh) {
  using stan::agrad::fvar;
  using boost::math::atanh;

  fvar<double> x(0.5,1.0);

  fvar<double> a = atanh(x);
  EXPECT_FLOAT_EQ(atanh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 - 0.5 * 0.5), a.d_);

  fvar<double> y(-0.9,1.0);

  fvar<double> b = atanh(y);
  EXPECT_FLOAT_EQ(atanh(-0.9), b.val_);
  EXPECT_FLOAT_EQ(1 / (1 - 0.9 * 0.9), b.d_);
}

TEST(AgradFvarVar, atanh) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::atanh;

  fvar<var> x(0.5,1.3);
  fvar<var> a = atanh(x);

  EXPECT_FLOAT_EQ(atanh(0.5), a.val_.val()); 
  EXPECT_FLOAT_EQ(1.3 / (1.0 - 0.5 * 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), g[0]);
}

TEST(AgradFvarFvar, atanh) {
  using stan::agrad::fvar;
  using boost::math::atanh;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<double> > a = atanh(x);

  EXPECT_FLOAT_EQ(atanh(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = atanh(y);

  EXPECT_FLOAT_EQ(atanh(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
