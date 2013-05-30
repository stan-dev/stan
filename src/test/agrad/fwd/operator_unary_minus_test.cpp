#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, operatorUnaryMinus) {
  using stan::agrad::fvar;

  fvar<double> x1(0.5,1.0);
  fvar<double> a = -x1;
  EXPECT_FLOAT_EQ(-0.5, a.val_);
  EXPECT_FLOAT_EQ(-1.0, a.d_);
}

TEST(AgradFvarVar, operatorUnaryMinus) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> a = -x;

  EXPECT_FLOAT_EQ(-0.5, a.val_.val());
  EXPECT_FLOAT_EQ(-1.3, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-1, g[0]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFvarFvar, operatorUnaryMinus) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > z = -x;
  EXPECT_FLOAT_EQ(-0.5, z.val_.val_);
  EXPECT_FLOAT_EQ(-1.0, z.val_.d_);
  EXPECT_FLOAT_EQ(0, z.d_.val_);
  EXPECT_FLOAT_EQ(0, z.d_.d_);
}
