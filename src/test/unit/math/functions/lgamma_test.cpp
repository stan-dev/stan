#include <gtest/gtest.h>

#include <stdexcept>
#include <iostream>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <stan/math/functions/lgamma.hpp>

TEST(MathsSpecialFunctions, lgamma) {
  // only brought in for Posix and ANSI C99
  // no need to include anything on thse compilers
  // see:  http://www.johndcook.com/cpp_gamma.html
  EXPECT_TRUE(boost::math::isinf(lgamma(0.0)));
}  
TEST(MathsSpecialFunctions, lgammaStanMath) {
  EXPECT_THROW(stan::math::lgamma(0.0), std::domain_error);
}  
TEST(MathsSpecialFunctions, lgammaStanMathUsing) {
  using stan::math::lgamma;
  EXPECT_THROW(lgamma(0.0), std::domain_error);
}  
TEST(MathsSpecialFunctions, lgammaUsingBoost) {
  using boost::math::lgamma;
  EXPECT_THROW(lgamma(0.0),std::domain_error);
}  
TEST(MathsSpecialFunctions, lgammaExplicitBoost) {
   EXPECT_THROW(boost::math::lgamma(0.0), std::domain_error);
}


// C++ 11 now
// TEST(MathsSpecialFunctions, lgammaExplicitStd) {
//    EXPECT_NO_THROW(std::lgamma(0.0));
// } 


