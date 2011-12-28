#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>
#include "stan/maths/special_functions.hpp"

TEST(MathsSpecialFunctions, int_step) {
  EXPECT_EQ(0, stan::maths::int_step(-1.0));
  EXPECT_EQ(0, stan::maths::int_step(0.0));
  EXPECT_EQ(1, stan::maths::int_step(0.00000000001));
  EXPECT_EQ(1, stan::maths::int_step(100.0));
}

TEST(MathsSpecialFunctions, binomial_coefficient_log) {
  EXPECT_FLOAT_EQ(1.0, exp(stan::maths::binomial_coefficient_log(2.0,2.0)));
  EXPECT_FLOAT_EQ(2.0, exp(stan::maths::binomial_coefficient_log(2.0,1.0)));
  EXPECT_FLOAT_EQ(3.0, exp(stan::maths::binomial_coefficient_log(3.0,1.0)));
  EXPECT_NEAR(3.0, exp(stan::maths::binomial_coefficient_log(3.0,2.0)),0.0001);
}

TEST(MathsSpecialFunctions, fma) {
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, stan::maths::fma(3.0,5.0,7.0));
}

TEST(MathsSpecialFunctions, beta_log) {
  EXPECT_FLOAT_EQ(0.0, stan::maths::beta_log(1.0,1.0));
  EXPECT_FLOAT_EQ(2.981361, stan::maths::beta_log(0.1,0.1));
  EXPECT_FLOAT_EQ(-4.094345, stan::maths::beta_log(3.0,4.0));
  EXPECT_FLOAT_EQ(-4.094345, stan::maths::beta_log(4.0,3.0));
}

TEST(MathsSpecialFunctions, inv_logit) {
  EXPECT_FLOAT_EQ(0.5, stan::maths::inv_logit(0.0));
  EXPECT_FLOAT_EQ(1.0/(1.0 + exp(-5.0)), stan::maths::inv_logit(5.0));
}

TEST(MathsSpecialFunctions, log1m) {
  EXPECT_FLOAT_EQ(log1p(-0.1),stan::maths::log1m(0.1));
}

TEST(MathsSpecialFunctions, log_loss) {
  EXPECT_FLOAT_EQ(0.0, stan::maths::binary_log_loss(0,0.0));
  EXPECT_FLOAT_EQ(0.0, stan::maths::binary_log_loss(1,1.0));
  EXPECT_FLOAT_EQ(-log(0.5), stan::maths::binary_log_loss(0,0.5));
  EXPECT_FLOAT_EQ(-log(0.5), stan::maths::binary_log_loss(1,0.5));
  EXPECT_FLOAT_EQ(-log(0.75), stan::maths::binary_log_loss(0,0.25));
  EXPECT_FLOAT_EQ(-log(0.75), stan::maths::binary_log_loss(1,0.75));
}

TEST(MathsSpecialFunctions, exp2) {
  EXPECT_FLOAT_EQ(std::pow(2.0,3.0), stan::maths::exp2(3.0));
  EXPECT_FLOAT_EQ(1, stan::maths::exp2(0.0));
}

TEST(MathsSpecialFunctions, log2) {
  EXPECT_FLOAT_EQ(0.0, stan::maths::log2(1.0));
  EXPECT_FLOAT_EQ(3.0, stan::maths::log2(8.0));
  EXPECT_FLOAT_EQ(std::log(5.0)/std::log(2.0), stan::maths::log2(5.0));
}

TEST(MathsSpecialFunctions, fdim) {
  EXPECT_FLOAT_EQ(1.0, stan::maths::fdim(3.0,2.0));
  EXPECT_FLOAT_EQ(0.0, stan::maths::fdim(2.0,3.0));
}

TEST(MathsSpecialFunctions, step) {
  EXPECT_EQ(1, stan::maths::step(3.7));
  EXPECT_EQ(1, stan::maths::step(0.0));
  EXPECT_EQ(0, stan::maths::step(-2.93));
}

TEST(MathsSpecialFunctions, inv_cloglog) {
  EXPECT_EQ(std::exp(-std::exp(3.7)), stan::maths::inv_cloglog(3.7));
  EXPECT_EQ(std::exp(-std::exp(0.0)), stan::maths::inv_cloglog(0.0));
  EXPECT_EQ(std::exp(-std::exp(-2.93)), stan::maths::inv_cloglog(-2.93));
}

TEST(MathsSpecialFunctions, Phi) {
  EXPECT_EQ(0.5 + 0.5 * boost::math::erf(0.0), stan::maths::Phi(0.0));
  EXPECT_FLOAT_EQ(0.5 + 0.5 * boost::math::erf(0.9/std::sqrt(2.0)), stan::maths::Phi(0.9));
  EXPECT_EQ(0.5 + 0.5 * boost::math::erf(-5.0/std::sqrt(2.0)), stan::maths::Phi(-5.0));
}

TEST(MathsSpecialFunctions, softmax) {
  std::vector<double> x;
  x.push_back(1.0);
  x.push_back(-1.0);
  x.push_back(2.0);
  std::vector<double> simplex(3);
  stan::maths::softmax<std::vector<double>,double>(x,simplex);
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
  //EXPECT_THROW(stan::maths::softmax<std::vector<double>,double>(x,simplex), std::invalid_argument);
  try{
    stan::maths::softmax<std::vector<double>,double>(x,simplex);
    FAIL();
  } catch (std::invalid_argument e) {
    SUCCEED();
  }
}
TEST(MathsSpecialFunctions, inverse_softmax_exception) {
  std::vector<double> simplex(2);
  std::vector<double> y(3);
  EXPECT_THROW(stan::maths::inverse_softmax< std::vector<double> >(simplex, y), std::invalid_argument);
}
TEST(MathsSpecialFunctions, log1p) {
  double x;

  x = 0;
  EXPECT_FLOAT_EQ (0.0, stan::maths::log1p(x));
  x = 0.0000001;
  EXPECT_FLOAT_EQ (0.0000001, stan::maths::log1p(x));
  x = 0.001;
  EXPECT_FLOAT_EQ (0.0009995003, stan::maths::log1p(x));
  x = 0.1;
  EXPECT_FLOAT_EQ (0.09531018, stan::maths::log1p(x));
  x = 1;
  EXPECT_FLOAT_EQ (0.6931472, stan::maths::log1p(x));
  x = 10;
  EXPECT_FLOAT_EQ (2.397895, stan::maths::log1p(x));

  x = -0.0000001;
  EXPECT_FLOAT_EQ (-0.0000001, stan::maths::log1p(x));
  x = -0.001;
  EXPECT_FLOAT_EQ (-0.0010005, stan::maths::log1p(x));
  x = -0.1;
  EXPECT_FLOAT_EQ (-0.1053605, stan::maths::log1p(x));
  x = -0.999;
  EXPECT_FLOAT_EQ (-6.907755, stan::maths::log1p(x));
  x = -1;
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), stan::maths::log1p(x));
}
TEST(MathsSpecialFunctions, log1p_exception) {
  double x;

  x = -2;
  EXPECT_THROW (stan::maths::log1p(x), std::domain_error);
}
TEST(MathsSpecialFunctions, lmgamma) {
  unsigned int k = 1;
  double x = 2.5;
  double result = k * (k - 1) * log(boost::math::constants::pi<double>()) / 4.0;
  result += lgamma(x); // j = 1
  EXPECT_FLOAT_EQ(result, stan::maths::lmgamma(k,x));

  k = 2;
  x = 3.0;
  result = k * (k - 1) * log(boost::math::constants::pi<double>()) / 4.0;
  result += lgamma(x); // j = 1
  result += lgamma(x + (1.0 - 2.0)/2.0); // j = 2
  EXPECT_FLOAT_EQ(result, stan::maths::lmgamma(k,x));
}

TEST(MathsSpecialFunctions, if_else) {
  unsigned int c = 5;
  double x = 1.0;
  double y = -1.0;
  EXPECT_FLOAT_EQ(x, stan::maths::if_else(c,x,y));
  c = 0;
  EXPECT_FLOAT_EQ(y, stan::maths::if_else(c,x,y));

  bool d = true;
  int u = 1;
  int v = -1;
  EXPECT_EQ(1, stan::maths::if_else(d,u,v));
  d = false;
  EXPECT_EQ(-1, stan::maths::if_else(d,u,v));
}

TEST(MathsSpecialFunctions, square) {
  double y = 2.0;
  EXPECT_FLOAT_EQ(y * y, stan::maths::square(y));

  y = 0.0;
  EXPECT_FLOAT_EQ(y * y, stan::maths::square(y));

  y = -32.7;
  EXPECT_FLOAT_EQ(y * y, stan::maths::square(y));
}

TEST(MathsSpecialFunctions, multiply_log) {
  double a = 2.0;
  double b = 3.0;
  EXPECT_FLOAT_EQ(a * log(b), stan::maths::multiply_log(a,b));

  a = 0.0;
  b = 0.0;
  EXPECT_FLOAT_EQ(0.0, stan::maths::multiply_log(a,b)) << 
    "when a and b are both 0, the result should be 0";

  a = 1.0;
  b = -1.0;
  EXPECT_TRUE(std::isnan(stan::maths::multiply_log(a,b))) << 
    "log(b) with b < 0 should result in NaN";
}


