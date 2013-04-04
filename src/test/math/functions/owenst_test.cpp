#include "stan/math/functions/owenst.hpp"
#include <gtest/gtest.h>
#include <boost/math/special_functions/owens_t.hpp>

TEST(MathSpecialFunctions,owenst) {
  double a = 1.0;
  double b = 2.0;
  EXPECT_FLOAT_EQ(stan::math::owenst(a,b), boost::math::owens_t(a,b));
}
