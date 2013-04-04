#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradFvar, lgamma){
  using stan::agrad::fvar;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = lgamma(x);
  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_);
  EXPECT_FLOAT_EQ(digamma(0.5), a.d_);
}
