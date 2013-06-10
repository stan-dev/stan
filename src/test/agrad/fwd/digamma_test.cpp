#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, digamma) {
  using stan::agrad::fvar;
  using boost::math::digamma;
  using boost::math::zeta;

  fvar<double> x(0.5,1.0);
  fvar<double> a = digamma(x);
  EXPECT_FLOAT_EQ(digamma(0.5), a.val_);
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.d_);
}

TEST(AgradFvarVar, digamma) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::digamma;  
  using boost::math::zeta;
  
  fvar<var> x(0.5,1.3);
  fvar<var> a = digamma(x);

  EXPECT_FLOAT_EQ(digamma(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 4.9348022005446793094, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(4.9348022005446793094, g[0]);
}

TEST(AgradFvarFvar, digamma) {
  using stan::agrad::fvar;
  using boost::math::digamma;
  using boost::math::zeta;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = digamma(x);

  EXPECT_FLOAT_EQ(digamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = digamma(y);
  EXPECT_FLOAT_EQ(digamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
