#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/erf.hpp>

TEST(AgradFvar, erfc){
  using stan::agrad::fvar;
  using std::exp;
  using std::sqrt;
  using boost::math::erfc;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = erfc(x);
  EXPECT_FLOAT_EQ(erfc(0.5), a.val_);
  EXPECT_FLOAT_EQ(-2 * exp(-0.5 * 0.5) 
                  / sqrt(boost::math::constants::pi<double>()), a.d_);

 fvar<double> b = erfc(-x);
  EXPECT_FLOAT_EQ(erfc(-0.5), b.val_);
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) 
                  / sqrt(boost::math::constants::pi<double>()), b.d_);
}
