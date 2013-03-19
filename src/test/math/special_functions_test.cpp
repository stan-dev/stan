#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>
#include "stan/math/special_functions.hpp"

TEST(MathSpecialFunctions,logicalOr) {
  using stan::math::logical_or;
  EXPECT_TRUE(logical_or(1,0));
  EXPECT_TRUE(logical_or(0,1));
  EXPECT_TRUE(logical_or(1,1));
  EXPECT_TRUE(logical_or(5.7,-1.9));
  EXPECT_TRUE(logical_or(5.7,-1));

  EXPECT_FALSE(logical_or(0,0));
  EXPECT_FALSE(logical_or(0.0, 0.0));
  EXPECT_FALSE(logical_or(0.0, 0));
  EXPECT_FALSE(logical_or(0, 0.0));
}
TEST(MathSpecialFunctions,logicalAnd) {
  using stan::math::logical_and;
  EXPECT_TRUE(logical_and(1,1));
  EXPECT_TRUE(logical_and(5.7,-1.9));

  EXPECT_FALSE(logical_and(0,0));
  EXPECT_FALSE(logical_and(0,1));
  EXPECT_FALSE(logical_and(1,0));
  EXPECT_FALSE(logical_and(0.0, 0.0));
  EXPECT_FALSE(logical_and(0.0, 1.0));
  EXPECT_FALSE(logical_and(1, 0.0));
}
TEST(MathSpecialFunctions,logicalEq) {
  using stan::math::logical_eq;
  EXPECT_TRUE(logical_eq(1,1));
  EXPECT_TRUE(logical_eq(5.7,5.7));
  EXPECT_TRUE(logical_eq(0,0.0));

  EXPECT_FALSE(logical_eq(0,1));
  EXPECT_FALSE(logical_eq(1.0,0));
  EXPECT_FALSE(logical_eq(1, 2));
  EXPECT_FALSE(logical_eq(2.0, -1.0));
}
TEST(MathSpecialFunctions,logicalNeq) {
  using stan::math::logical_neq;
  EXPECT_TRUE(logical_neq(0,1));
  EXPECT_TRUE(logical_neq(1.0,0));
  EXPECT_TRUE(logical_neq(1, 2));
  EXPECT_TRUE(logical_neq(2.0, -1.0));

  EXPECT_FALSE(logical_neq(1,1));
  EXPECT_FALSE(logical_neq(5.7,5.7));
  EXPECT_FALSE(logical_neq(0,0.0));
}
TEST(MathSpecialFunctions,logicalLt) {
  using stan::math::logical_lt;
  EXPECT_TRUE(logical_lt(0,1));
  EXPECT_TRUE(logical_lt(1.0,2.0));
  EXPECT_TRUE(logical_lt(1, 2.0));
  EXPECT_TRUE(logical_lt(-1, 0));

  EXPECT_FALSE(logical_lt(1,1));
  EXPECT_FALSE(logical_lt(5.7,5.7));
  EXPECT_FALSE(logical_lt(5.7,-9.0));
  EXPECT_FALSE(logical_lt(0,0.0));
}
TEST(MathSpecialFunctions,logicalLte) {
  using stan::math::logical_lte;
  EXPECT_TRUE(logical_lte(0,1));
  EXPECT_TRUE(logical_lte(1.0,2.0));
  EXPECT_TRUE(logical_lte(1, 2.0));
  EXPECT_TRUE(logical_lte(-1, 0));
  EXPECT_TRUE(logical_lte(1,1));
  EXPECT_TRUE(logical_lte(5.7,5.7));

  EXPECT_FALSE(logical_lte(5.7,-9.0));
  EXPECT_FALSE(logical_lte(-1,-2));
}
TEST(MathSpecialFunctions,logicalGt) {
  using stan::math::logical_gt;
  EXPECT_TRUE(logical_gt(1,0));
  EXPECT_TRUE(logical_gt(2,1.00));
  EXPECT_TRUE(logical_gt(2.0,1));
  EXPECT_TRUE(logical_gt(0,-1));

  EXPECT_FALSE(logical_gt(1,1));
  EXPECT_FALSE(logical_gt(5.7,5.7));
  EXPECT_FALSE(logical_gt(-5.7,9.0));
  EXPECT_FALSE(logical_gt(0,0.0));
}
TEST(MathSpecialFunctions,logicalGte) {
  using stan::math::logical_gte;
  EXPECT_TRUE(logical_gte(1,0));
  EXPECT_TRUE(logical_gte(2.0,1.0));
  EXPECT_TRUE(logical_gte(2.0, 1));
  EXPECT_TRUE(logical_gte(0, -1));
  EXPECT_TRUE(logical_gte(1,1));
  EXPECT_TRUE(logical_gte(5.7,5.7));

  EXPECT_FALSE(logical_gte(-9.0, 5.7));
  EXPECT_FALSE(logical_gte(-2,-1));
}
TEST(MathSpecialFunctions,asBool) {
  using stan::math::as_bool;
  EXPECT_TRUE(as_bool(1));
  EXPECT_TRUE(as_bool(1.7));
  EXPECT_TRUE(as_bool(-1.7));
  EXPECT_TRUE(as_bool(std::numeric_limits<double>::infinity()));
  EXPECT_TRUE(as_bool(-std::numeric_limits<double>::infinity()));
  // don't like this behavior, but it's what C++ does
  EXPECT_TRUE(as_bool(std::numeric_limits<double>::quiet_NaN()));

  EXPECT_FALSE(as_bool(0));
  EXPECT_FALSE(as_bool(0.0));
  EXPECT_FALSE(as_bool(0.0f));
}
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
