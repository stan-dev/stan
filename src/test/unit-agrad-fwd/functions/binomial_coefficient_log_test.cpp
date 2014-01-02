#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdBinomialCoefficientLog,Fvar) {
  using stan::agrad::fvar;
  using stan::math::binomial_coefficient_log;
  using boost::math::digamma;

  fvar<double> x(2004.0,1.0);
  fvar<double> y(1002.0,2.0);

  fvar<double> a = stan::agrad::binomial_coefficient_log(x, y);
  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0, 1002.0), a.val_);
  EXPECT_FLOAT_EQ(0.69289774, a.d_);
}

// TEST(AgradFwdBinomialCoefficientLog,FvarVar_1stDeriv) {
//   using stan::agrad::fvar;
//   using stan::agrad::var;
//   using stan::math::binomial_coefficient_log;
//   using boost::math::digamma;

//   fvar<var> x(2004.0,1.0);
//   fvar<var> z(1002.0,2.0);
//   fvar<var> a = binomial_coefficient_log(x,z);

//   EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val());
//   EXPECT_FLOAT_EQ(0.69289774, a.d_.val());

//   AVEC y = createAVEC(x.val_,z.val_);
//   VEC g;
//   a.val_.grad(y,g);
//   EXPECT_FLOAT_EQ(0.69289774, g[0]);
//   EXPECT_FLOAT_EQ(-7.9936058e-15, g[1]);
// }

// TEST(AgradFwdBinomialCoefficientLog,FvarFvarDouble) {
//   using stan::agrad::fvar;
//   using stan::math::binomial_coefficient_log;
//   using stan::agrad::binomial_coefficient_log;

//   fvar<fvar<double> > x;
//   x.val_.val_ = 2004.0;
//   x.val_.d_ = 1.0;

//   fvar<fvar<double> > y;
//   y.val_.val_ = 1002.0;
//   y.d_.val_ = 1.0;

//   fvar<fvar<double> > a = binomial_coefficient_log(x,y);

//   EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_);
//   EXPECT_FLOAT_EQ(0.69289774, a.val_.d_);
//   EXPECT_FLOAT_EQ(-8.8817842e-15, a.d_.val_);
//   EXPECT_FLOAT_EQ(0.0009975062, a.d_.d_);
// }
