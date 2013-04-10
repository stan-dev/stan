#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>

TEST(AgradFvar, binom_coeff_log) {
  using stan::agrad::fvar;
  using stan::math::binomial_coefficient_log;
  using boost::math::digamma;

  fvar<double> x(2004.0);
  x.d_ = 1.0;
  fvar<double> y(1002.0);
  y.d_ = 2.0;

  fvar<double> a = binomial_coefficient_log(x, y);
  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0, 1002.0), a.val_);
  EXPECT_FLOAT_EQ(2.0 * log(2004.0 - 1002.0) + (1002.0 * (1.0 - 2.0))
                  / (2004.0 - 1002.0) + 1.0 * log(2004.0 / (2004.0 - 1002.0))
                  + (2004.0 + 0.5) / (2004.0 / (2004.0 - 1002.0))
                  * (1.0 * (2004.0 - 1002.0) - (1.0 - 2.0) * 2004.0) 
                  / ((2004.0 - 1002.0) * (2004.0 - 1002.0)) + 1.0
                  / (12 * 2004.0 * 2004.0) - 2.0 + (1.0 - 2.0) 
                  / (12 * (2004.0 - 1002.0) * (2004.0 - 1002.0))
                  - digamma(1002.0 + 1) * 2.0, a.d_);


  double z = 1003.0;

  fvar<double> b = binomial_coefficient_log(x, z);
  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0, 1003.0), b.val_);
  EXPECT_FLOAT_EQ(0 * log(2004.0 - 1003.0) + (1003.0
                  * (1.0 - 0)) / (2004.0 - 1003.0) + 1.0 * log(2004.0 
                  / (2004.0 - 1003.0)) + (2004.0 + 0.5) / (2004.0
                  / (2004.0 - 1003.0)) * (1.0 * (2004.0 - 1003.0)
                  - (1.0 - 0) * 2004.0) / ((2004.0 - 1003.0) 
                  * (2004.0 - 1003.0)) + 1.0 / (12 * 2004.0 * 2004.0) - 0 
                  + (1.0 - 0) / (12 * (2004.0 - 1003.0) * (2004.0 - 1003.0)) 
                  - digamma(1003.0 + 1) * 0, b.d_);

  double w = 2006.0;

  fvar<double> c = binomial_coefficient_log(w, y);

  // EXPECT_FLOAT_EQ(binomial_coefficient_log(2006.0, 1002.0), c.val_);
  // EXPECT_FLOAT_EQ(2.0 * log(2006.0 - 1002.0) + (1002.0 * (0 - 2.0)) 
  //                 / (2006.0 - 1002.0) + 0 * log(2006.0 / (2006.0 - 1002.0))
  //                 + (2006.0 + 0.5) / (2006.0 / (2006.0 - 1002.0)) * (0
  //                 * (2006.0 - 1002.0) - (0 - 2.0) * 2006.0) / ((2006.0 - 1002.0)
  //                 * (2006.0 - 1002.0)) + 0 / (12 * 2006.0 * 2006.0) - 2.0
  //                 + (0 - 2.0) / (12 * (2006.0 - 1002.0) * (2006.0 - 1002.0)) 
  //                 - digamma(1002.0 + 1) * 2.0, c.d_);
}
