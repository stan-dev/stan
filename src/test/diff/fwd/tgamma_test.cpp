#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradFvar, tgamma){
  using stan::agrad::fvar;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = tgamma(x);
  EXPECT_FLOAT_EQ(tgamma(0.5), a.val_);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), a.d_);
}
