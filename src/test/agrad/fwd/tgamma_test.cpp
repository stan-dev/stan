#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, tgamma) {
  using stan::agrad::fvar;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<double> x(0.5,1.0);
  fvar<double> a = tgamma(x);
  EXPECT_FLOAT_EQ(tgamma(0.5), a.val_);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), a.d_);
}

TEST(AgradFvarVar, tgamma) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<var> x(0.5,1.3);
  fvar<var> a = tgamma(x);

  EXPECT_FLOAT_EQ(tgamma(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * tgamma(0.5) * digamma(0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), g[0]);
}

TEST(AgradFvarFvar, tgamma) {
  using stan::agrad::fvar;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = tgamma(x);

  EXPECT_FLOAT_EQ(tgamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = tgamma(y);
  EXPECT_FLOAT_EQ(tgamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
