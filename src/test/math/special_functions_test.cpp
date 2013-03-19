#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>
#include "stan/math/special_functions.hpp"

TEST(MathSpecialFunctions,BoostUseTr1Def) {
 bool BOOST_RESULT_OF_USE_TR1_is_defined = false;

#ifdef BOOST_RESULT_OF_USE_TR1
 BOOST_RESULT_OF_USE_TR1_is_defined = true;
#endif

 EXPECT_TRUE(BOOST_RESULT_OF_USE_TR1_is_defined);
}
TEST(MathSpecialFunctions,BoostNoDeclTypeDef) {
 bool BOOST_NO_DECLTYPE_is_defined = false;
#ifdef BOOST_NO_DECLTYPE
 BOOST_NO_DECLTYPE_is_defined = true;
#endif
 EXPECT_TRUE(BOOST_NO_DECLTYPE_is_defined);
}


TEST(MathsSpecialFunctions, pi_fun) {
  EXPECT_FLOAT_EQ(4.0 * std::atan(1.0), stan::math::pi());
}
TEST(MathsSpecialFunctions, e_fun) {
  EXPECT_FLOAT_EQ(std::exp(1.0), stan::math::e());
}
TEST(MathsSpecialFunctions, sqrt2_fun) {
  EXPECT_FLOAT_EQ(std::sqrt(2.0), stan::math::sqrt2());
}
TEST(MathsSpecialFunctions, log10_fun) {
  EXPECT_FLOAT_EQ(std::log(10.0), stan::math::log10());
}

TEST(MathsSpecialFunctions, infty) {
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), stan::math::positive_infinity());
}
TEST(MathsSpecialFunctions, neg_infty) {
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  stan::math::negative_infinity());
}
TEST(MathsSpecialFunctions, nan) {
  EXPECT_TRUE(std::isnan(stan::math::not_a_number()));
}
TEST(MathsSpecialFunctions, epsilon) {
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::epsilon(),
                  stan::math::epsilon());
}
TEST(MathsSpecialFunctions, negative_epsilon) {
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::epsilon(),
                  stan::math::negative_epsilon());
}
