#include <gtest/gtest.h>

#include <stdexcept>
#include <iostream>

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <stan/math/functions/digamma.hpp>

TEST(MathsSpecialFunctions, digamma) {
  EXPECT_FLOAT_EQ(boost::math::digamma(0.5), stan::math::digamma(0.5));
  EXPECT_FLOAT_EQ(boost::math::digamma(-1.5), stan::math::digamma(-1.5));
  EXPECT_THROW(stan::math::digamma(-1.0), std::domain_error);
}  

