#include <gtest/gtest.h>

#include <iostream>

#include <cmath>

#include <boost/math/special_functions/fpclassify.hpp>

#include <stan/agrad/fvar.hpp>

#include <boost/math/special_functions/cbrt.hpp>


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

  fvar<double> d = -3 * exp(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * exp(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 * exp(0.5) + 5, d.d_);
}

TEST(AgradFvar, log) {
  using stan::agrad::fvar;
  using std::log;

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
}

TEST(AgradFvar, asin) {
  using stan::agrad::fvar;
  using std::asin;
  using std::sqrt;

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
}

TEST(AgradFvar, acos) {
  using stan::agrad::fvar;
  using std::acos;
  using std::sqrt;

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

TEST(AgradFvar, sinh){
  using stan::agrad::fvar;
  using std::sinh;
  using std::cosh;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = sinh(x);
  EXPECT_FLOAT_EQ(sinh(0.5), a.val_);
  EXPECT_FLOAT_EQ(cosh(0.5), a.d_);

  fvar<double> b = 3 * sinh(x);
  EXPECT_FLOAT_EQ(3 * sinh(0.5), b.val_);
  EXPECT_FLOAT_EQ(3 * cosh(0.5), b.d_);

  fvar<double> c = -sinh(x) + 4;
  EXPECT_FLOAT_EQ(-sinh(0.5) + 4, c.val_);
  EXPECT_FLOAT_EQ(-cosh(0.5), c.d_);
}

TEST(AgradFvar, cosh){
  using stan::agrad::fvar;
  using std::sinh;
  using std::cosh;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = cosh(x);
  EXPECT_FLOAT_EQ(cosh(0.5), a.val_);
  EXPECT_FLOAT_EQ(sinh(0.5), a.d_);

  fvar<double> b = 3 * cosh(x);
  EXPECT_FLOAT_EQ(3 * cosh(0.5), b.val_);
  EXPECT_FLOAT_EQ(3 * sinh(0.5), b.d_);

  fvar<double> c = -cosh(x) + 4;
  EXPECT_FLOAT_EQ(-cosh(0.5) + 4, c.val_);
  EXPECT_FLOAT_EQ(-sinh(0.5), c.d_);
}

TEST(AgradFvar, tanh){
  using stan::agrad::fvar;
  using std::tanh;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = tanh(x);
  EXPECT_FLOAT_EQ(tanh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 - tanh(0.5) * tanh(0.5), a.d_);

  fvar<double> b = 3 * tanh(x);
  EXPECT_FLOAT_EQ(3 * tanh(0.5), b.val_);
  EXPECT_FLOAT_EQ(3 * (1 - tanh(0.5) * tanh(0.5)), b.d_);

  fvar<double> c = -tanh(x) + 4;
  EXPECT_FLOAT_EQ(-tanh(0.5) + 4, c.val_);
  EXPECT_FLOAT_EQ(-(1 - tanh(0.5) * tanh(0.5)), c.d_);
}
