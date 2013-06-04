#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/binary_log_loss.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, binary_log_loss) {
  using stan::agrad::fvar;
  using stan::math::binary_log_loss;

  fvar<double> y(0.4,3.0);

  fvar<double> a = binary_log_loss(0.0, y);
  EXPECT_FLOAT_EQ(binary_log_loss(0.0, 0.4), a.val_);
  EXPECT_FLOAT_EQ(3.0 / 0.4, a.d_);

  fvar<double> b = binary_log_loss(1.0, y);
  EXPECT_FLOAT_EQ(binary_log_loss(1.0, 0.4), b.val_);
  EXPECT_FLOAT_EQ(-3.0 / 0.4, b.d_);
}

TEST(AgradFvarVar, binary_log_loss) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::binary_log_loss;

  fvar<var> z(0.4,3.0);
  fvar<var> a = binary_log_loss(0,z);

  EXPECT_FLOAT_EQ(binary_log_loss(0.0, 0.4), a.val_.val());
  EXPECT_FLOAT_EQ(3.0 / 0.4, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(2.5, g[0]);
}

TEST(AgradFvarFvar, binary_log_loss) {
  using stan::agrad::fvar;
  using stan::math::binary_log_loss;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.4;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.4;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = binary_log_loss(0.0,y);

  EXPECT_FLOAT_EQ(binary_log_loss(0.0,0.4), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.5, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > b = binary_log_loss(1.0,x);

  EXPECT_FLOAT_EQ(binary_log_loss(1.0,0.4), b.val_.val_);
  EXPECT_FLOAT_EQ(-2.5, b.val_.d_);
  EXPECT_FLOAT_EQ(0, b.d_.val_);
  EXPECT_FLOAT_EQ(0, b.d_.d_);
}
