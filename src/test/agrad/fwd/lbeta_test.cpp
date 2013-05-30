#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/functions/lbeta.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, lbeta) {
  using stan::agrad::fvar;
  using boost::math::tgamma;
  using stan::math::lbeta;

  fvar<double> x(0.5,1.0);
  fvar<double> y(1.2,2.0);

  double w = 1.3;

  fvar<double> a = lbeta(x, y);
  EXPECT_FLOAT_EQ(lbeta(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(0.5) + 2.0 / tgamma(1.2) 
                  - (1.0 + 2.0) / tgamma(0.5 + 1.2), a.d_);

  fvar<double> b = lbeta(x, w);
  EXPECT_FLOAT_EQ(lbeta(0.5, 1.3), b.val_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(0.5) - 1.0 / tgamma(0.5 + 1.3), b.d_);

  fvar<double> c = lbeta(w, x);
  EXPECT_FLOAT_EQ(lbeta(1.3, 0.5), c.val_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(0.5) - 1.0 / tgamma(1.3 + 0.5), c.d_);
}

TEST(AgradFvarVar, lbeta) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::tgamma;
  using stan::math::lbeta;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = lbeta(x,z);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / tgamma(3.0) + 1.0 / tgamma(6.0) - (1.0 + 1.3) / tgamma(3.0 + 6.0), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.2178571,g[0]); //???
  std::isnan(g[1]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  std::isnan(g[1]);
}

TEST(AgradFvarFvar, lbeta) {
  using stan::agrad::fvar;
  using boost::math::tgamma;
  using stan::math::lbeta;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = lbeta(x,y);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(3.0) - 1.0 / tgamma(9.0), a.val_.d_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(6.0) - 1.0 / tgamma(9.0), a.d_.val_);
  EXPECT_FLOAT_EQ(0.000053091306, a.d_.d_);
}
