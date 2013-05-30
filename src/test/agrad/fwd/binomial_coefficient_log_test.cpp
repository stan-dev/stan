#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, binom_coeff_log) {
  using stan::agrad::fvar;
  using stan::math::binomial_coefficient_log;
  using boost::math::digamma;

  fvar<double> x(2004.0,1.0);
  fvar<double> y(1002.0,2.0);

  fvar<double> a = stan::agrad::binomial_coefficient_log(x, y);
  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0, 1002.0), a.val_);
  EXPECT_FLOAT_EQ(2.0 * log(2004.0 - 1002.0) + (1002.0 * (1.0 - 2.0))
                  / (2004.0 - 1002.0) + 1.0 * log(2004.0 / (2004.0 - 1002.0))
                  + (2004.0 + 0.5) / (2004.0 / (2004.0 - 1002.0))
                  * (1.0 * (2004.0 - 1002.0) - (1.0 - 2.0) * 2004.0) 
                  / ((2004.0 - 1002.0) * (2004.0 - 1002.0)) + 1.0
                  / (12 * 2004.0 * 2004.0) - 2.0 + (1.0 - 2.0) 
                  / (12 * (2004.0 - 1002.0) * (2004.0 - 1002.0))
                  - digamma(1002.0 + 1) * 2.0, a.d_);
}

TEST(AgradFvarVar, binomial_coefficient_log) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::binomial_coefficient_log;

  fvar<var> x(2004.0,1.0);

  fvar<var> z(1002.0,2.0);
  fvar<var> a = binomial_coefficient_log(x,z);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val());
  EXPECT_FLOAT_EQ(2.0 * log(2004.0 - 1002.0) + (1002.0 * (1.0 - 2.0))
                  / (2004.0 - 1002.0) + 1.0 * log(2004.0 / (2004.0 - 1002.0))
                  + (2004.0 + 0.5) / (2004.0 / (2004.0 - 1002.0))
                  * (1.0 * (2004.0 - 1002.0) - (1.0 - 2.0) * 2004.0) 
                  / ((2004.0 - 1002.0) * (2004.0 - 1002.0)) + 1.0
                  / (12 * 2004.0 * 2004.0) - 2.0 + (1.0 - 2.0) 
                  / (12 * (2004.0 - 1002.0) * (2004.0 - 1002.0))
                  - digamma(1002.0 + 1) * 2.0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}

TEST(AgradFvarFvar, binomial_coefficient_log) {
  using stan::agrad::fvar;
  using stan::math::binomial_coefficient_log;
  using stan::agrad::binomial_coefficient_log;

  fvar<fvar<double> > x;
  x.val_.val_ = 2004.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1002.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(-1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ((1.5 * 1.5 - 1.5 * 1.5) / ((1.5 * 1.5 + 1.5 * 1.5) * (1.5 * 1.5 + 1.5 * 1.5)), a.d_.d_);
}
