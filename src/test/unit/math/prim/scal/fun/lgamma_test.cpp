#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, lgamma) {
  // only brought in for Posix and ANSI C99
  // no need to include anything on thse compilers
  // see:  http://www.johndcook.com/cpp_gamma.html
  EXPECT_TRUE(boost::math::isinf(lgamma(0.0)));
}  

TEST(MathFunctions, lgammaStanMath) {
  EXPECT_THROW(stan::math::lgamma(0.0), std::domain_error);
}  

TEST(MathFunctions, lgammaStanMathUsing) {
  using stan::math::lgamma;
  EXPECT_THROW(lgamma(0.0), std::domain_error);
}  

TEST(MathFunctions, lgammaUsingBoost) {
  using boost::math::lgamma;
  EXPECT_THROW(lgamma(0.0),std::domain_error);
}  

TEST(MathFunctions, lgammaExplicitBoost) {
   EXPECT_THROW(boost::math::lgamma(0.0), std::domain_error);
}

// C++ 11 now
// TEST(MathFunctions, lgammaExplicitStd) {
//    EXPECT_NO_THROW(std::lgamma(0.0));
// } 

TEST(MathFunctions, lgamma_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::lgamma(nan));
}
