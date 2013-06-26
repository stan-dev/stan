#include "stan/math/functions/modified_bessel_second_kind.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, modified_bessel_second_kind) {
  using stan::math::modified_bessel_second_kind;
  
  EXPECT_FLOAT_EQ(0.011159676085853024269745195979833489225, 
                  modified_bessel_second_kind(0,4.0));
  EXPECT_THROW(modified_bessel_second_kind(1,-3.0), std::domain_error);
  EXPECT_THROW(modified_bessel_second_kind(-1,-3.0), std::domain_error);
}
