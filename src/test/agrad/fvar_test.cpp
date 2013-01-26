#include <gtest/gtest.h>

#include <iostream>

#include <cmath>

#include <boost/math/special_functions/fpclassify.hpp>

#include <stan/agrad/fvar.hpp>

#include <boost/math/special_functions/cbrt.hpp>
#include <stan/math/constants.hpp>
#include <math.h>


TEST(AgradFvar,fvar) {
  using stan::agrad::fvar;
  typedef stan::agrad::fvar<double> fvd;

  fvar<double> a;
  EXPECT_FLOAT_EQ(0.0, a.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_);

  fvar<double> b(1.9);
  EXPECT_FLOAT_EQ(1.9, b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c(1.93, -27.832);
  EXPECT_FLOAT_EQ(1.93, c.val_);
  EXPECT_FLOAT_EQ(-27.832, c.d_);

  fvar<double> d = -c;
  EXPECT_FLOAT_EQ(-1.93, d.val_);
  EXPECT_FLOAT_EQ(27.832, d.d_);

  fvar<double> e(5.0);
  d += e;
  EXPECT_FLOAT_EQ(3.07, d.val_);

  EXPECT_FLOAT_EQ(3.07, (d++).val_);
  EXPECT_FLOAT_EQ(4.07, d.val_);

  EXPECT_FLOAT_EQ(5.07, (++d).val_);
  EXPECT_FLOAT_EQ(5.07, d.val_);
}

TEST(AgradFvar, operatorPlusEqual){
  using stan::agrad::fvar;

  fvar<double> a(0.5);
  fvar<double> x1(0.4);
  a.d_ = 1.0;
  x1.d_ = 2.0;
  a += x1;
  EXPECT_FLOAT_EQ(0.5 + 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 + 2.0, a.d_);

  fvar<double> b(0.5);
  fvar<double> x2(0.4);
  b.d_ = 1.0;
  x2.d_ = 2.0;
  b += -x2;
  EXPECT_FLOAT_EQ(0.5 - 0.4, b.val_);
  EXPECT_FLOAT_EQ(1.0 - 2.0, b.d_);

  fvar<double> c(0.6);
  double x3(0.3);
  c.d_ = 3.0;
  c += x3;
  EXPECT_FLOAT_EQ(0.6 + 0.3, c.val_);
  EXPECT_FLOAT_EQ(3.0, c.d_);

  fvar<double> d(0.5);
  fvar<double> x4(-0.4);
  d.d_ = 1.0;
  x4.d_ = 2.0;
  d += x4;
  EXPECT_FLOAT_EQ(0.5 - 0.4, d.val_);
  EXPECT_FLOAT_EQ(1.0 + 2.0, d.d_);
}

TEST(AgradFvar, operatorMinusEqual){
  using stan::agrad::fvar;

  fvar<double> a(0.5);
  fvar<double> x1(0.4);
  a.d_ = 1.0;
  x1.d_ = 2.0;
  a -= x1;
  EXPECT_FLOAT_EQ(0.5 - 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 - 2.0, a.d_);

  fvar<double> b(0.5);
  fvar<double> x2(0.4);
  b.d_ = 1.0;
  x2.d_ = 2.0;
  b -= -x2;
  EXPECT_FLOAT_EQ(0.5 + 0.4, b.val_);
  EXPECT_FLOAT_EQ(1.0 + 2.0, b.d_);

  fvar<double> c(0.6);
  double x3(0.3);
  c.d_ = 3.0;
  c -= x3;
  EXPECT_FLOAT_EQ(0.6 - 0.3, c.val_);
  EXPECT_FLOAT_EQ(3.0, c.d_);

  fvar<double> d(0.5);
  fvar<double> x4(-0.4);
  d.d_ = 1.0;
  x4.d_ = 2.0;
  d -= x4;
  EXPECT_FLOAT_EQ(0.5 + 0.4, d.val_);
  EXPECT_FLOAT_EQ(1.0 - 2.0, d.d_);
}

TEST(AgradFvar, operatorMultiplyEqual){
  using stan::agrad::fvar;

  fvar<double> a(0.5);
  fvar<double> x1(0.4);
  a.d_ = 1.0;
  x1.d_ = 2.0;
  a *= x1;
  EXPECT_FLOAT_EQ(0.5 * 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 * 0.4 + 2.0 * 0.5, a.d_);

  fvar<double> b(0.5);
  fvar<double> x2(0.4);
  b.d_ = 1.0;
  x2.d_ = 2.0;
  b *= -x2;
  EXPECT_FLOAT_EQ(0.5 * -0.4, b.val_);
  EXPECT_FLOAT_EQ(1.0 * -0.4 + -2.0 * 0.5, b.d_);

  fvar<double> c(0.6);
  double x3(0.3);
  c.d_ = 3.0;
  c *= x3;
  EXPECT_FLOAT_EQ(0.6 * 0.3, c.val_);
  EXPECT_FLOAT_EQ(3.0, c.d_);

  fvar<double> d(0.5);
  fvar<double> x4(-0.4);
  d.d_ = 1.0;
  x4.d_ = 2.0;
  d *= x4;
  EXPECT_FLOAT_EQ(0.5 * -0.4, d.val_);
  EXPECT_FLOAT_EQ(1.0 * -0.4 + 2.0 * 0.5, d.d_);
}

TEST(AgradFvar, operatorDivideEqual){
  using stan::agrad::fvar;

  fvar<double> a(0.5);
  fvar<double> x1(0.4);
  a.d_ = 1.0;
  x1.d_ = 2.0;
  a /= x1;
  EXPECT_FLOAT_EQ(0.5 / 0.4, a.val_);
  EXPECT_FLOAT_EQ((1.0 * 0.4 - 2.0 * 0.5) / (0.4 * 0.4), a.d_);

  fvar<double> b(0.5);
  fvar<double> x2(0.4);
  b.d_ = 1.0;
  x2.d_ = 2.0;
  b /= -x2;
  EXPECT_FLOAT_EQ(0.5 / -0.4, b.val_);
  EXPECT_FLOAT_EQ((1.0 * -0.4 - -2.0 * 0.5) / (-0.4 * -0.4), b.d_);

  fvar<double> c(0.6);
  double x3(0.3);
  c.d_ = 3.0;
  c /= x3;
  EXPECT_FLOAT_EQ(0.6 / 0.3, c.val_);
  EXPECT_FLOAT_EQ(3.0, c.d_);

  fvar<double> d(0.5);
  fvar<double> x4(-0.4);
  d.d_ = 1.0;
  x4.d_ = 2.0;
  d /= x4;
  EXPECT_FLOAT_EQ(0.5 / -0.4, d.val_);
  EXPECT_FLOAT_EQ((1.0 * -0.4 - 2.0 * 0.5) / (-0.4 * -0.4), d.d_);
}

TEST(AgradFvar, operatorPlus){
  using stan::agrad::fvar;

  fvar<double> x1(0.5);
  fvar<double> x2(0.4);
  x1.d_ = 1.0;
  x2.d_ = 2.0;
  fvar<double> a = x1 + x2;

  EXPECT_FLOAT_EQ(0.5 + 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 + 2.0, a.d_);

  fvar<double> b = -x1 + x2;
  EXPECT_FLOAT_EQ(-0.5 + 0.4, b.val_);
  EXPECT_FLOAT_EQ(-1 * 1.0 + 2.0, b.d_);

  fvar<double> c = 2 * x1 + -3 * x2;
  EXPECT_FLOAT_EQ(2 * 0.5 + -3 * 0.4, c.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 + -3 * 2.0, c.d_);

  fvar<double> x3(0.5);
  fvar<double> x4(1.0);
  x3.d_ = 1.0;
  x4.d_ = 2.0;

  fvar<double> d = 2 * x3 + x4;
  EXPECT_FLOAT_EQ(2 * 0.5 + 1 * 1.0, d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 + 1 * 2.0, d.d_);

  fvar<double> e = 2 * x3 + 4;
  EXPECT_FLOAT_EQ(2 * 0.5 + 4, e.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, e.d_);

  fvar<double> f = 5 + 2 * x3;
  EXPECT_FLOAT_EQ(5 + 2 * 0.5, f.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, f.d_);
}

TEST(AgradFvar, operatorMinus){
  using stan::agrad::fvar;

  fvar<double> x1(0.5);
  fvar<double> x2(0.4);
  x1.d_ = 1.0;
  x2.d_ = 2.0;
  fvar<double> a = x1 - x2;
  EXPECT_FLOAT_EQ(0.5 - 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 - 2.0, a.d_);

  fvar<double> b = -x1 - x2;
  EXPECT_FLOAT_EQ(-0.5 - 0.4, b.val_);
  EXPECT_FLOAT_EQ(-1 * 1.0 - 2.0, b.d_);

  fvar<double> c = 2 * x1 - -3 * x2;
  EXPECT_FLOAT_EQ(2 * 0.5 - -3 * 0.4, c.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 - -3 * 2.0, c.d_);

  fvar<double> x3(0.5);
  fvar<double> x4(1.0);
  x3.d_ = 1.0;
  x4.d_ = 2.0;

  fvar<double> d = 2 * x3 - x4;
  EXPECT_FLOAT_EQ(2 * 0.5 - 1 * 1.0, d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 - 1 * 2.0, d.d_);

  fvar<double> e = 2 * x3 - 4;
  EXPECT_FLOAT_EQ(2 * 0.5 - 4, e.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, e.d_);

  fvar<double> f = 5 - 2 * x3;
  EXPECT_FLOAT_EQ(5 - 2 * 0.5, f.val_);
  EXPECT_FLOAT_EQ(-2 * 1.0, f.d_);
}

TEST(AgradFvar, operatorMultiply){
  using stan::agrad::fvar;

  fvar<double> x1(0.5);
  fvar<double> x2(0.4);
  x1.d_ = 1.0;
  x2.d_ = 2.0;
  fvar<double> a = x1 * x2;

  EXPECT_FLOAT_EQ(0.5 * 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 * 0.4 + 2.0 * 0.5, a.d_);

  fvar<double> b = -x1 * x2;
  EXPECT_FLOAT_EQ(-0.5 * 0.4, b.val_);
  EXPECT_FLOAT_EQ(-1 * 0.4 - 2.0 * 0.5, b.d_);

  fvar<double> c = -3 * x1 * x2;
  EXPECT_FLOAT_EQ(-3 * 0.5 * 0.4, c.val_);
  EXPECT_FLOAT_EQ(3 * (-1 * 0.4 - 2.0 * 0.5), c.d_);

  fvar<double> x3(0.5);
  x3.d_ = 1.0;

  fvar<double> e = 2 * x3;
  EXPECT_FLOAT_EQ(2 * 0.5, e.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, e.d_);

  fvar<double> f = x3 * -2;
  EXPECT_FLOAT_EQ(0.5 * -2, f.val_);
  EXPECT_FLOAT_EQ(1.0 * -2, f.d_);
}

TEST(AgradFvar, operatorPlusPlus){
  using stan::agrad::fvar;

  fvar<double> x(0.5);
  x.d_ = 1.0;
  x++;

  EXPECT_FLOAT_EQ(0.5 + 1.0, x.val_);
  EXPECT_FLOAT_EQ(1.0, x.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  y++;

  EXPECT_FLOAT_EQ(-0.5 + 1.0, y.val_);
  EXPECT_FLOAT_EQ(1.0, y.d_);
}

TEST(AgradFvar, operatorMinusMinus){
  using stan::agrad::fvar;

  fvar<double> x(0.5);
  x.d_ = 1.0;
  x--;

  EXPECT_FLOAT_EQ(0.5 - 1.0, x.val_);
  EXPECT_FLOAT_EQ(1.0, x.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  y--;

  EXPECT_FLOAT_EQ(-0.5 - 1.0, y.val_);
  EXPECT_FLOAT_EQ(1.0, y.d_);
}

TEST(AgradFvar, sqrt) {
  using stan::agrad::fvar;
  using std::sqrt;
  using std::isnan;
  using stan::math::INFTY;
  using stan::math::NOT_A_NUMBER;

  fvar<double> x(0.5);
  x.d_ = 1.0; //derivatives w.r.t. x
  fvar<double> a = sqrt(x);

  EXPECT_FLOAT_EQ(sqrt(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (2 * sqrt(0.5)), a.d_);

  fvar<double> b = 3 * sqrt(x) + x;
  EXPECT_FLOAT_EQ(3 * sqrt(0.5) + 0.5, b.val_);
  EXPECT_FLOAT_EQ(3 / (2 * sqrt(0.5)) + 1, b.d_);

  fvar<double> c = -sqrt(x) + 5;
  EXPECT_FLOAT_EQ(-sqrt(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (2 * sqrt(0.5)), c.d_);

  fvar<double> d = -3 * sqrt(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * sqrt(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (2 * sqrt(0.5)) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = sqrt(-y);
  EXPECT_FLOAT_EQ(sqrt(0.5), e.val_);
  EXPECT_FLOAT_EQ(-1 / (2 * sqrt(0.5)), e.d_); 

  fvar<double> f = sqrt(y);
  isnan(f.val_);
  isnan(f.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> g = sqrt(z);
  EXPECT_FLOAT_EQ(sqrt(0.0), g.val_);
  EXPECT_FLOAT_EQ(INFTY, g.d_); 
}

TEST(AgradFvar, cbrt) {
  using stan::agrad::fvar;
  using boost::math::cbrt;
  using stan::math::INFTY;

  fvar<double> x(0.5);
  x.d_ = 1.0; //derivatives w.r.t. x
  fvar<double> a = cbrt(x);

  EXPECT_FLOAT_EQ(cbrt(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (3 * pow(0.5, 2.0 / 3.0)), a.d_);

  fvar<double> b = 3 * cbrt(x) + x;
  EXPECT_FLOAT_EQ(3 * cbrt(0.5) + 0.5, b.val_);
  EXPECT_FLOAT_EQ(3 / (3 * pow(0.5, 2.0 / 3.0)) + 1, b.d_);

  fvar<double> c = -cbrt(x) + 5;
  EXPECT_FLOAT_EQ(-cbrt(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (3 * pow(0.5, 2.0 / 3.0)), c.d_);

  fvar<double> d = -3 * cbrt(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * cbrt(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (3 * pow(0.5, 2.0 / 3.0)) + 5, d.d_);

  fvar<double> e = -3 * cbrt(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * cbrt(-0.5) + 5 * 0.5, e.val_);
  EXPECT_FLOAT_EQ(3 / (3 * cbrt(-0.5) * cbrt(-0.5)) + 5, e.d_);

  fvar<double> y(0.0);
  y.d_ = 1.0;
  fvar<double> f = cbrt(y);
  EXPECT_FLOAT_EQ(cbrt(0.0), f.val_);
  EXPECT_FLOAT_EQ(INFTY, f.d_);
}

TEST(AgradFvar, square) {
  using stan::agrad::fvar;
  using stan::math::square;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  fvar<double> a = square(x);

  EXPECT_FLOAT_EQ(square(0.5), a.val_);
  EXPECT_FLOAT_EQ(2 * 0.5, a.d_);

  fvar<double> b = 3 * square(x) + x;
  EXPECT_FLOAT_EQ(3 * square(0.5) + 0.5, b.val_);
  EXPECT_FLOAT_EQ(3 * 2 * 0.5 + 1, b.d_);

  fvar<double> c = -square(x) + 5;
  EXPECT_FLOAT_EQ(-square(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-2 * 0.5, c.d_);

  fvar<double> d = -3 * square(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * square(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 * 2 * 0.5 + 5, d.d_);

  fvar<double> e = -3 * square(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * square(-0.5) + 5 * 0.5, e.val_);
  EXPECT_FLOAT_EQ(-3 * 2 * 0.5 + 5, e.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> f = square(y);
  EXPECT_FLOAT_EQ(square(-0.5), f.val_);
  EXPECT_FLOAT_EQ(2 * -0.5, f.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> g = square(z);
  EXPECT_FLOAT_EQ(square(0.0), g.val_);
  EXPECT_FLOAT_EQ(2 * 0.0, g.d_);
}   

TEST(AgradFvar, exp) {
  using stan::agrad::fvar;
  using std::exp;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = exp(x);
  EXPECT_FLOAT_EQ(exp(0.5), a.val_);
  EXPECT_FLOAT_EQ(exp(0.5), a.d_);

  fvar<double> b = 2 * exp(x) + 4;
  EXPECT_FLOAT_EQ(2 * exp(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 * exp(0.5), b.d_);

  fvar<double> c = -exp(x) + 5;
  EXPECT_FLOAT_EQ(-exp(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-exp(0.5), c.d_);

  fvar<double> d = -3 * exp(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * exp(-0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(3 * exp(-0.5) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = exp(y);
  EXPECT_FLOAT_EQ(exp(-0.5), e.val_);
  EXPECT_FLOAT_EQ(exp(-0.5), e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = exp(z);
  EXPECT_FLOAT_EQ(exp(0.0), f.val_);
  EXPECT_FLOAT_EQ(exp(0.0), f.d_);
}

TEST(AgradFvar, exp2){
  using stan::agrad::fvar;
  using stan::math::exp2;
  using std::log;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = exp2(x);
  EXPECT_FLOAT_EQ(exp2(0.5), a.val_);
  EXPECT_FLOAT_EQ(exp2(0.5) * log(2), a.d_);

  fvar<double> b = 2 * exp2(x) + 4;
  EXPECT_FLOAT_EQ(2 * exp2(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 * exp2(0.5) * log(2), b.d_);

  fvar<double> c = -exp2(x) + 5;
  EXPECT_FLOAT_EQ(-exp2(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-exp2(0.5) * log(2), c.d_);

  fvar<double> d = -3 * exp2(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * exp2(-0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(3 * exp2(-0.5) * log(2) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = exp2(y);
  EXPECT_FLOAT_EQ(exp2(-0.5), e.val_);
  EXPECT_FLOAT_EQ(exp2(-0.5) * log(2), e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = exp2(z);
  EXPECT_FLOAT_EQ(exp2(0.0), f.val_);
  EXPECT_FLOAT_EQ(exp2(0.0) * log(2), f.d_);
}

TEST(AgradFvar, log) {
  using stan::agrad::fvar;
  using std::log;
  using std::isnan;
  using stan::math::NOT_A_NUMBER;
  using stan::math::NEGATIVE_INFTY;
  using stan::math::INFTY;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = log(x);
  EXPECT_FLOAT_EQ(log(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / 0.5, a.d_);

  fvar<double> b = 2 * log(x) + 4;
  EXPECT_FLOAT_EQ(2 * log(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / 0.5, b.d_);

  fvar<double> c = -log(x) + 5;
  EXPECT_FLOAT_EQ(-log(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / 0.5, c.d_);

  fvar<double> d = -3 * log(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * log(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / 0.5 + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = log(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = log(z);
  EXPECT_FLOAT_EQ(NEGATIVE_INFTY, f.val_);
  EXPECT_FLOAT_EQ(INFTY, f.d_);
}

TEST(AgradFvar, log2) {
  using stan::agrad::fvar;
  using std::log;
  using std::isnan;
  using stan::math::log2;
  using stan::math::NOT_A_NUMBER;
  using stan::math::NEGATIVE_INFTY;
  using stan::math::INFTY;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = log2(x);
  EXPECT_FLOAT_EQ(log2(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(2)), a.d_);

  fvar<double> b = 2 * log2(x) + 4;
  EXPECT_FLOAT_EQ(2 * log2(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (0.5 * log(2)), b.d_);

  fvar<double> c = -log2(x) + 5;
  EXPECT_FLOAT_EQ(-log2(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (0.5 * log(2)), c.d_);

  fvar<double> d = -3 * log2(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * log2(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (0.5 * log(2)) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = log2(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = log2(z);
  EXPECT_FLOAT_EQ(NEGATIVE_INFTY, f.val_);
  EXPECT_FLOAT_EQ(INFTY, f.d_);
}

TEST(AgradFvar, log10) {
  using stan::agrad::fvar;
  using std::log;
  using std::log10;
  using std::isnan;
  using stan::math::NOT_A_NUMBER;
  using stan::math::NEGATIVE_INFTY;
  using stan::math::INFTY;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = log10(x);
  EXPECT_FLOAT_EQ(log10(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(10)), a.d_);

  fvar<double> b = 2 * log10(x) + 4;
  EXPECT_FLOAT_EQ(2 * log10(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (0.5 * log(10)), b.d_);

  fvar<double> c = -log10(x) + 5;
  EXPECT_FLOAT_EQ(-log10(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (0.5 * log(10)), c.d_);

  fvar<double> d = -3 * log10(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * log10(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (0.5 * log(10)) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = log10(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = log10(z);
  EXPECT_FLOAT_EQ(NEGATIVE_INFTY, f.val_);
  EXPECT_FLOAT_EQ(INFTY, f.d_);
}

TEST(AgradFvar, pow) {
  using stan::agrad::fvar;
  using std::pow;
  using std::log;

  fvar<double> x(0.5);
  x.d_ = 1.0;
  double y = 5.0;

  fvar<double> a = pow(x, y);
  EXPECT_FLOAT_EQ(pow(0.5, 5.0), a.val_);
  EXPECT_FLOAT_EQ(5.0 * pow(0.5, 5.0 - 1.0), a.d_);

  fvar<double> b = pow(y, x);
  EXPECT_FLOAT_EQ(pow(5.0, 0.5), b.val_);
  EXPECT_FLOAT_EQ(log(5.0) * pow(5.0, 0.5), b.d_);
}

TEST(AgradFvar, sin) {
  using stan::agrad::fvar;
  using std::sin;
  using std::cos;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = sin(x);
  EXPECT_FLOAT_EQ(sin(0.5), a.val_);
  EXPECT_FLOAT_EQ(cos(0.5), a.d_);

  fvar<double> b = 2 * sin(x) + 4;
  EXPECT_FLOAT_EQ(2 * sin(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 * cos(0.5), b.d_);

  fvar<double> c = -sin(x) + 5;
  EXPECT_FLOAT_EQ(-sin(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-cos(0.5), c.d_);

  fvar<double> d = -3 * sin(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * sin(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 * cos(0.5) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = sin(y);
  EXPECT_FLOAT_EQ(sin(-0.5), e.val_);
  EXPECT_FLOAT_EQ(cos(-0.5), e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = sin(z);
  EXPECT_FLOAT_EQ(sin(0.0), f.val_);
  EXPECT_FLOAT_EQ(cos(0.0), f.d_);
}

TEST(AgradFvar, cos) {
  using stan::agrad::fvar;
  using std::sin;
  using std::cos;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = cos(x);
  EXPECT_FLOAT_EQ(cos(0.5), a.val_);
  EXPECT_FLOAT_EQ(-sin(0.5), a.d_);

  fvar<double> b = 2 * cos(x) + 4;
  EXPECT_FLOAT_EQ(2 * cos(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 * -sin(0.5), b.d_);

  fvar<double> c = -cos(x) + 5;
  EXPECT_FLOAT_EQ(-cos(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(sin(0.5), c.d_);

  fvar<double> d = -3 * cos(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * cos(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 * -sin(0.5) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = cos(y);
  EXPECT_FLOAT_EQ(cos(-0.5), e.val_);
  EXPECT_FLOAT_EQ(-sin(-0.5), e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = cos(z);
  EXPECT_FLOAT_EQ(cos(0.0), f.val_);
  EXPECT_FLOAT_EQ(-sin(0.0), f.d_);
}

TEST(AgradFvar, tan) {
  using stan::agrad::fvar;
  using std::tan;
  using std::cos;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = tan(x);
  EXPECT_FLOAT_EQ(tan(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (cos(0.5) * cos(0.5)), a.d_);

  fvar<double> b = 2 * tan(x) + 4;
  EXPECT_FLOAT_EQ(2 * tan(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (cos(0.5) * cos(0.5)), b.d_);

  fvar<double> c = -tan(x) + 5;
  EXPECT_FLOAT_EQ(-tan(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (cos(0.5) * cos(0.5)), c.d_);

  fvar<double> d = -3 * tan(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * tan(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (cos(0.5) * cos(0.5)) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = tan(y);
  EXPECT_FLOAT_EQ(tan(-0.5), e.val_);
  EXPECT_FLOAT_EQ(1 / (cos(-0.5) * cos(-0.5)), e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = tan(z);
  EXPECT_FLOAT_EQ(tan(0.0), f.val_);
  EXPECT_FLOAT_EQ(1 / (cos(0.0) * cos(0.0)), f.d_);
}

TEST(AgradFvar, asin) {
  using stan::agrad::fvar;
  using std::asin;
  using std::isnan;
  using std::sqrt;
  using stan::math::INFTY;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = asin(x);
  EXPECT_FLOAT_EQ(asin(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 - 0.5 * 0.5), a.d_);

  fvar<double> b = 2 * asin(x) + 4;
  EXPECT_FLOAT_EQ(2 * asin(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / sqrt(1 - 0.5 * 0.5), b.d_);

  fvar<double> c = -asin(x) + 5;
  EXPECT_FLOAT_EQ(-asin(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / sqrt(1 - 0.5 * 0.5), c.d_);

  fvar<double> d = -3 * asin(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * asin(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / sqrt(1 - 0.5 * 0.5) + 5, d.d_);

  fvar<double> y(3.4);
  y.d_ = 1.0;
  fvar<double> e = asin(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(1.0);
  z.d_ = 1.0;
  fvar<double> f = asin(z);
  EXPECT_FLOAT_EQ(asin(1.0), f.val_);
  EXPECT_FLOAT_EQ(INFTY, f.d_);
}

TEST(AgradFvar, acos) {
  using stan::agrad::fvar;
  using std::acos;
  using std::sqrt;
  using std::isnan;
  using stan::math::NEGATIVE_INFTY;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = acos(x);
  EXPECT_FLOAT_EQ(acos(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / -sqrt(1 - 0.5 * 0.5), a.d_);

  fvar<double> b = 2 * acos(x) + 4;
  EXPECT_FLOAT_EQ(2 * acos(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / -sqrt(1 - 0.5 * 0.5), b.d_);

  fvar<double> c = -acos(x) + 5;
  EXPECT_FLOAT_EQ(-acos(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / -sqrt(1 - 0.5 * 0.5), c.d_);

  fvar<double> d = -3 * acos(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * acos(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / -sqrt(1 - 0.5 * 0.5) + 5, d.d_);

  fvar<double> y(3.4);
  y.d_ = 1.0;
  fvar<double> e = acos(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(1.0);
  z.d_ = 1.0;
  fvar<double> f = acos(z);
  EXPECT_FLOAT_EQ(acos(1.0), f.val_);
  EXPECT_FLOAT_EQ(NEGATIVE_INFTY, f.d_);
}

TEST(AgradFvar, atan) {
  using stan::agrad::fvar;
  using std::atan;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = atan(x);
  EXPECT_FLOAT_EQ(atan(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 + 0.5 * 0.5), a.d_);

  fvar<double> b = 2 * atan(x) + 4;
  EXPECT_FLOAT_EQ(2 * atan(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (1 + 0.5 * 0.5), b.d_);

  fvar<double> c = -atan(x) + 5;
  EXPECT_FLOAT_EQ(-atan(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (1 + 0.5 * 0.5), c.d_);

  fvar<double> d = -3 * atan(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * atan(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (1 + 0.5 * 0.5) + 5, d.d_);
}

