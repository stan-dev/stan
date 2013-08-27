#include <gtest/gtest.h>
#include <stan/diff/fwd.hpp>
#include <boost/math/special_functions/digamma.hpp>

TEST(DiffFvar, lgamma){
  using stan::diff::fvar;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = lgamma(x);
  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_);
  EXPECT_FLOAT_EQ(digamma(0.5), a.d_);
}
