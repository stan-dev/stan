#include "stan/math/functions/modified_bessel_first_kind.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, modified_bessel_first_kind) {
  using stan::math::modified_bessel_first_kind;
  
  EXPECT_FLOAT_EQ(11.301921952136330496356270183217102497412616594, 
                  modified_bessel_first_kind(0,4.0));
  EXPECT_FLOAT_EQ(-3.953370217402609396478635740580581287584221595, 
                  modified_bessel_first_kind(1,-3.0));
  EXPECT_FLOAT_EQ(-3.953370217402609396478635740580581287584221595, 
                  modified_bessel_first_kind(-1,-3.0));
}
