#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>
#include "stan/math/special_functions.hpp"

TEST(MathsSpecialFunctions, int_step) {
  EXPECT_EQ(0U, stan::math::int_step(-1.0));
  EXPECT_EQ(0U, stan::math::int_step(0.0));
  EXPECT_EQ(1U, stan::math::int_step(0.00000000001));
  EXPECT_EQ(1U, stan::math::int_step(100.0));
}

TEST(MathsSpecialFunctions, binomial_coefficient_log) {
  EXPECT_FLOAT_EQ(1.0, exp(stan::math::binomial_coefficient_log(2.0,2.0)));
  EXPECT_FLOAT_EQ(2.0, exp(stan::math::binomial_coefficient_log(2.0,1.0)));
  EXPECT_FLOAT_EQ(3.0, exp(stan::math::binomial_coefficient_log(3.0,1.0)));
  EXPECT_NEAR(3.0, exp(stan::math::binomial_coefficient_log(3.0,2.0)),0.0001);
}

TEST(MathsSpecialFunctions, fma) {
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, stan::math::fma(3.0,5.0,7.0));
}

TEST(MathsSpecialFunctions, lbeta) {
  EXPECT_FLOAT_EQ(0.0, stan::math::lbeta(1.0,1.0));
  EXPECT_FLOAT_EQ(2.981361, stan::math::lbeta(0.1,0.1));
  EXPECT_FLOAT_EQ(-4.094345, stan::math::lbeta(3.0,4.0));
  EXPECT_FLOAT_EQ(-4.094345, stan::math::lbeta(4.0,3.0));
}

TEST(MathsSpecialFunctions, inv_logit) {
  EXPECT_FLOAT_EQ(0.5, stan::math::inv_logit(0.0));
  EXPECT_FLOAT_EQ(1.0/(1.0 + exp(-5.0)), stan::math::inv_logit(5.0));
}

TEST(MathsSpecialFunctions, log1m) {
  EXPECT_FLOAT_EQ(log1p(-0.1),stan::math::log1m(0.1));
}

TEST(MathsSpecialFunctions, log_loss) {
  EXPECT_FLOAT_EQ(0.0, stan::math::binary_log_loss(0,0.0));
  EXPECT_FLOAT_EQ(0.0, stan::math::binary_log_loss(1,1.0));
  EXPECT_FLOAT_EQ(-log(0.5), stan::math::binary_log_loss(0,0.5));
  EXPECT_FLOAT_EQ(-log(0.5), stan::math::binary_log_loss(1,0.5));
  EXPECT_FLOAT_EQ(-log(0.75), stan::math::binary_log_loss(0,0.25));
  EXPECT_FLOAT_EQ(-log(0.75), stan::math::binary_log_loss(1,0.75));
}

TEST(MathsSpecialFunctions, exp2) {
  EXPECT_FLOAT_EQ(std::pow(2.0,3.0), stan::math::exp2(3.0));
  EXPECT_FLOAT_EQ(1, stan::math::exp2(0.0));
}

TEST(MathsSpecialFunctions, log2) {
  EXPECT_FLOAT_EQ(0.0, stan::math::log2(1.0));
  EXPECT_FLOAT_EQ(3.0, stan::math::log2(8.0));
  EXPECT_FLOAT_EQ(std::log(5.0)/std::log(2.0), stan::math::log2(5.0));
}

TEST(MathsSpecialFunctions, fdim) {
  EXPECT_FLOAT_EQ(1.0, stan::math::fdim(3.0,2.0));
  EXPECT_FLOAT_EQ(0.0, stan::math::fdim(2.0,3.0));
}

TEST(MathsSpecialFunctions, step) {
  EXPECT_EQ(1, stan::math::step(3.7));
  EXPECT_EQ(1, stan::math::step(0.0));
  EXPECT_EQ(0, stan::math::step(-2.93));
}

TEST(MathsSpecialFunctions, inv_cloglog) {
  EXPECT_EQ(std::exp(-std::exp(3.7)), stan::math::inv_cloglog(3.7));
  EXPECT_EQ(std::exp(-std::exp(0.0)), stan::math::inv_cloglog(0.0));
  EXPECT_EQ(std::exp(-std::exp(-2.93)), stan::math::inv_cloglog(-2.93));
}

TEST(MathsSpecialFunctions, Phi) {
  EXPECT_EQ(0.5 + 0.5 * boost::math::erf(0.0), stan::math::Phi(0.0));
  EXPECT_FLOAT_EQ(0.5 + 0.5 * boost::math::erf(0.9/std::sqrt(2.0)), stan::math::Phi(0.9));
  EXPECT_EQ(0.5 + 0.5 * boost::math::erf(-5.0/std::sqrt(2.0)), stan::math::Phi(-5.0));
}

TEST(MathsSpecialFunctions, softmax) {
  std::vector<double> x;
  x.push_back(1.0);
  x.push_back(-1.0);
  x.push_back(2.0);
  std::vector<double> simplex(3);
  stan::math::softmax<std::vector<double>,double>(x,simplex);
  EXPECT_FLOAT_EQ(1.0, simplex[0] + simplex[1] + simplex[2]);
  double sum = exp(1.0) + exp(-1.0) + exp(2.0);
  EXPECT_FLOAT_EQ(exp(1.0)/sum, simplex[0]);
  EXPECT_FLOAT_EQ(exp(-1.0)/sum, simplex[1]);
  EXPECT_FLOAT_EQ(exp(2.0)/sum, simplex[2]);
}
TEST(MathsSpecialFunctions, softmax_exception) {
  std::vector<double> x;
  x.push_back(1.0);
  x.push_back(-1.0);
  x.push_back(2.0);
  std::vector<double> simplex(2);
  
  // dl: EXPECT_THROW is not able to handle this templating
  //EXPECT_THROW(stan::math::softmax<std::vector<double>,double>(x,simplex), std::invalid_argument);
  try{
    stan::math::softmax<std::vector<double>,double>(x,simplex);
    FAIL();
  } catch (std::invalid_argument e) {
    SUCCEED();
  }
}
TEST(MathsSpecialFunctions, inverse_softmax_exception) {
  std::vector<double> simplex(2);
  std::vector<double> y(3);
  EXPECT_THROW(stan::math::inverse_softmax< std::vector<double> >(simplex, y), std::invalid_argument);
}
TEST(MathsSpecialFunctions, log1p) {
  double x;

  x = 0;
  EXPECT_FLOAT_EQ (0.0, stan::math::log1p(x));
  x = 0.0000001;
  EXPECT_FLOAT_EQ (0.0000001, stan::math::log1p(x));
  x = 0.001;
  EXPECT_FLOAT_EQ (0.0009995003, stan::math::log1p(x));
  x = 0.1;
  EXPECT_FLOAT_EQ (0.09531018, stan::math::log1p(x));
  x = 1;
  EXPECT_FLOAT_EQ (0.6931472, stan::math::log1p(x));
  x = 10;
  EXPECT_FLOAT_EQ (2.397895, stan::math::log1p(x));

  x = -0.0000001;
  EXPECT_FLOAT_EQ (-0.0000001, stan::math::log1p(x));
  x = -0.001;
  EXPECT_FLOAT_EQ (-0.0010005, stan::math::log1p(x));
  x = -0.1;
  EXPECT_FLOAT_EQ (-0.1053605, stan::math::log1p(x));
  x = -0.999;
  EXPECT_FLOAT_EQ (-6.907755, stan::math::log1p(x));
  x = -1;
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), stan::math::log1p(x));
}
TEST(MathsSpecialFunctions, log1p_exception) {
  double x;

  x = -2;
  EXPECT_THROW (stan::math::log1p(x), std::domain_error);
}
TEST(MathsSpecialFunctions, lmgamma) {
  unsigned int k = 1;
  double x = 2.5;
  double result = k * (k - 1) * log(boost::math::constants::pi<double>()) / 4.0;
  result += lgamma(x); // j = 1
  EXPECT_FLOAT_EQ(result, stan::math::lmgamma(k,x));

  k = 2;
  x = 3.0;
  result = k * (k - 1) * log(boost::math::constants::pi<double>()) / 4.0;
  result += lgamma(x); // j = 1
  result += lgamma(x + (1.0 - 2.0)/2.0); // j = 2
  EXPECT_FLOAT_EQ(result, stan::math::lmgamma(k,x));
}

TEST(MathsSpecialFunctions, if_else) {
  unsigned int c = 5;
  double x = 1.0;
  double y = -1.0;
  EXPECT_FLOAT_EQ(x, stan::math::if_else(c,x,y));
  c = 0;
  EXPECT_FLOAT_EQ(y, stan::math::if_else(c,x,y));

  bool d = true;
  int u = 1;
  int v = -1;
  EXPECT_EQ(1, stan::math::if_else(d,u,v));
  d = false;
  EXPECT_EQ(-1, stan::math::if_else(d,u,v));
}

TEST(MathsSpecialFunctions, square) {
  double y = 2.0;
  EXPECT_FLOAT_EQ(y * y, stan::math::square(y));

  y = 0.0;
  EXPECT_FLOAT_EQ(y * y, stan::math::square(y));

  y = -32.7;
  EXPECT_FLOAT_EQ(y * y, stan::math::square(y));
}

TEST(MathsSpecialFunctions, multiply_log) {
  double a = 2.0;
  double b = 3.0;
  EXPECT_FLOAT_EQ(a * log(b), stan::math::multiply_log(a,b));

  a = 0.0;
  b = 0.0;
  EXPECT_FLOAT_EQ(0.0, stan::math::multiply_log(a,b)) << 
    "when a and b are both 0, the result should be 0";

  a = 1.0;
  b = -1.0;
  EXPECT_TRUE(std::isnan(stan::math::multiply_log(a,b))) << 
    "log(b) with b < 0 should result in NaN";
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
TEST(MathsSpecialFunctions, log2_fun) {
  EXPECT_FLOAT_EQ(std::log(2.0), stan::math::log2());
}
TEST(MathsSpecialFunctions, log10_fun) {
  EXPECT_FLOAT_EQ(std::log(10.0), stan::math::log10());
}

TEST(MathsSpecialFunctions, infty) {
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), stan::math::infinity());
}
TEST(MathsSpecialFunctions, neg_infty) {
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  stan::math::negative_infinity());
}
TEST(MathsSpecialFunctions, nan) {
  EXPECT_TRUE(std::isnan(stan::math::nan()));
}
TEST(MathsSpecialFunctions, epsilon) {
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::epsilon(),
                  stan::math::epsilon());
}
TEST(MathsSpecialFunctions, negative_epsilon) {
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::epsilon(),
                  stan::math::negative_epsilon());
}
