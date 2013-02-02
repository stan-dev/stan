#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include <stan/math/constants.hpp>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/math/special_functions.hpp>
#include <boost/math/special_functions/cbrt.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/hypot.hpp>
#include <boost/math/special_functions/asinh.hpp>
#include <boost/math/special_functions/acosh.hpp>
#include <boost/math/special_functions/atanh.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <boost/math/special_functions/digamma.hpp>

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

TEST(AgradFvar, operatorDivide){
  using stan::agrad::fvar;
  using std::isnan;

  fvar<double> x1(0.5);
  fvar<double> x2(0.4);
  x1.d_ = 1.0;
  x2.d_ = 2.0;
  fvar<double> a = x1 / x2;

  EXPECT_FLOAT_EQ(0.5 / 0.4, a.val_);
  EXPECT_FLOAT_EQ((1.0 * 0.4 - 2.0 * 0.5) / (0.4 * 0.4), a.d_);

  fvar<double> b = -x1 / x2;
  EXPECT_FLOAT_EQ(-0.5 / 0.4, b.val_);
  EXPECT_FLOAT_EQ((-1 * 0.4 + 2.0 * 0.5) / (0.4 * 0.4), b.d_);

  fvar<double> c = -3 * x1 / x2;
  EXPECT_FLOAT_EQ(-3 * 0.5 / 0.4, c.val_);
  EXPECT_FLOAT_EQ(3 * (-1 * 0.4 + 2.0 * 0.5) / (0.4 * 0.4), c.d_);

  fvar<double> x3(0.5);
  x3.d_ = 1.0;
   double x4 = 2.0;

  fvar<double> e = x4 / x3;
  EXPECT_FLOAT_EQ(2 / 0.5, e.val_);
  EXPECT_FLOAT_EQ(-2 * 1.0 / (0.5 * 0.5), e.d_);

  fvar<double> f = x3 / -2;
  EXPECT_FLOAT_EQ(0.5 / -2, f.val_);
  EXPECT_FLOAT_EQ(1.0 / -2, f.d_);

  fvar<double> x5(0.0);
  x5.d_ = 1.0;
  fvar<double> g = x3/x5;
  isnan(g.val_);
  isnan(g.d_);
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

TEST(AgradFvar, abs) {
  using stan::agrad::fvar;
  using std::abs;
  using std::isnan;

  fvar<int> x(2);
  fvar<int> y(-3);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<int> a = abs(x);
  EXPECT_FLOAT_EQ(abs(2), a.val_);
  EXPECT_FLOAT_EQ(1.0, a.d_);

  fvar<int> b = abs(-x);
  EXPECT_FLOAT_EQ(abs(-2), b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<int> c = abs(y);
  EXPECT_FLOAT_EQ(abs(-3), c.val_);
  EXPECT_FLOAT_EQ(-2.0, c.d_);

  fvar<double> d = abs(2 * x);
  EXPECT_FLOAT_EQ(abs(2 * 2), d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, d.d_);

  fvar<double> e = abs(y + 4);
  EXPECT_FLOAT_EQ(abs(-3 + 4), e.val_);
  EXPECT_FLOAT_EQ(2.0, e.d_);

  fvar<double> f = abs(x - 2);
  EXPECT_FLOAT_EQ(abs(2 - 2), f.val_);
  isnan(f.d_);
 }

TEST(AgradFvar, fabs) {
  using stan::agrad::fvar;
  using std::fabs;
  using std::isnan;

  fvar<double> x(2.0);
  fvar<double> y(-3.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = fabs(x);
  EXPECT_FLOAT_EQ(fabs(2), a.val_);
  EXPECT_FLOAT_EQ(1.0, a.d_);

  fvar<double> b = fabs(-x);
  EXPECT_FLOAT_EQ(fabs(-2), b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<double> c = fabs(y);
  EXPECT_FLOAT_EQ(fabs(-3), c.val_);
  EXPECT_FLOAT_EQ(-2.0, c.d_);

  fvar<double> d = fabs(x * 2);
  EXPECT_FLOAT_EQ(fabs(2 * 2), d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, d.d_);

  fvar<double> e = fabs(y + 4);
  EXPECT_FLOAT_EQ(fabs(-3.0 + 4), e.val_);
  EXPECT_FLOAT_EQ(2.0, e.d_);

  fvar<double> f = fabs(x - 2);
  EXPECT_FLOAT_EQ(fabs(2.0 - 2), f.val_);
  isnan(f.d_);
 }

TEST(AgradFvar, fdim) {
  using stan::agrad::fvar;
  using stan::math::fdim;
  using std::isnan;
  using std::floor;

  fvar<double> x(2.0);
  fvar<double> y(-3.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = fdim(x, y);
  EXPECT_FLOAT_EQ(fdim(2.0, -3.0), a.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.0 + 2.0 * -floor(2.0 / -3.0), a.d_);

  fvar<double> b = fdim(2 * x, y);
  EXPECT_FLOAT_EQ(fdim(2 * 2.0, -3.0), b.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 * 1.0 + 2.0 * -floor(4.0 / -3.0), b.d_);

  fvar<double> c = fdim(y, x);
  EXPECT_FLOAT_EQ(fdim(-3.0, 2.0), c.val_);
  EXPECT_FLOAT_EQ(0.0, c.d_);

  fvar<double> d = fdim(x, x);
  EXPECT_FLOAT_EQ(fdim(2.0, 2.0), d.val_);
  EXPECT_FLOAT_EQ(0.0, d.d_);

  double z = 1.0;

  fvar<double> e = fdim(x, z);
  EXPECT_FLOAT_EQ(fdim(2.0, 1.0), e.val_);
  EXPECT_FLOAT_EQ(1.0, e.d_);

  fvar<double> f = fdim(z, x);
  EXPECT_FLOAT_EQ(fdim(1.0, 2.0), f.val_);
  EXPECT_FLOAT_EQ(0.0, f.d_);
 }

TEST(AgradFvar, floor) {
  using stan::agrad::fvar;
  using std::floor;

  fvar<double> x(0.5);
  fvar<double> y(2.0);
  y.d_ = 2.0;
  x.d_ = 1.0;

  fvar<double> a = floor(x);
  EXPECT_FLOAT_EQ(floor(0.5), a.val_);
  EXPECT_FLOAT_EQ(0, a.d_);

  fvar<double> b = floor(y);
  EXPECT_FLOAT_EQ(floor(2.0), b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = floor(2 * x);
  EXPECT_FLOAT_EQ(floor(2 * 0.5), c.val_);
   EXPECT_FLOAT_EQ(0.0, c.d_);
}

TEST(AgradFvar, trunc) {
  using stan::agrad::fvar;
  using boost::math::trunc;

  fvar<double> x(0.5);
  fvar<double> y(2.4);
  y.d_ = 2.0;
  x.d_ = 1.0;

  fvar<double> a = trunc(x);
  EXPECT_FLOAT_EQ(trunc(0.5), a.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_);

  fvar<double> b = trunc(y);
  EXPECT_FLOAT_EQ(trunc(2.4), b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = trunc(2 * x);
  EXPECT_FLOAT_EQ(trunc(2 * 0.5), c.val_);
  EXPECT_FLOAT_EQ(0.0, c.d_);
}

TEST(AgradFvar, round) {
  using stan::agrad::fvar;
  using boost::math::round;

  fvar<double> x(0.5);
  fvar<double> y(2.4);
  y.d_ = 2.0;
  x.d_ = 1.0;

  fvar<double> a = round(x);
  EXPECT_FLOAT_EQ(round(0.5), a.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_);

  fvar<double> b = round(y);
  EXPECT_FLOAT_EQ(round(2.4), b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = round(2 * x);
  EXPECT_FLOAT_EQ(round(2 * 0.5), c.val_);
  EXPECT_FLOAT_EQ(0.0, c.d_);

  fvar<double> z(1.25);
  z.d_ = 1.0;

  fvar<double> d = round(2 * z);
  EXPECT_FLOAT_EQ(round(2 * 1.25), d.val_);
   EXPECT_FLOAT_EQ(0.0, d.d_);
}

TEST(AgradFvar, ceil) {
  using stan::agrad::fvar;
  using std::ceil;

  fvar<double> x(0.5);
  fvar<double> y(2.0);
  y.d_ = 2.0;
  x.d_ = 1.0;

  fvar<double> a = ceil(x);
  EXPECT_FLOAT_EQ(ceil(0.5), a.val_);
  EXPECT_FLOAT_EQ(0, a.d_);

  fvar<double> b = ceil(y);
  EXPECT_FLOAT_EQ(ceil(2.0), b.val_);
   EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = ceil(2 * x);
  EXPECT_FLOAT_EQ(ceil(2 * 0.5), c.val_);
   EXPECT_FLOAT_EQ(0.0, c.d_);
}

TEST(AgradFvar, fmod) {
  using stan::agrad::fvar;
  using std::fmod;
  using std::floor;

  fvar<double> x(2.0);
  fvar<double> y(3.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = fmod(x, y);
  EXPECT_FLOAT_EQ(fmod(2.0, 3.0), a.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.0 + 2.0 * -floor(2.0 / 3.0), a.d_);

  double z = 4.0;
  double w = 3.0;

  fvar<double> e = fmod(x, z);
  EXPECT_FLOAT_EQ(fmod(2.0, 4.0), e.val_);
  EXPECT_FLOAT_EQ(1.0 / 4.0, e.d_);

  fvar<double> f = fmod(w, x);
  EXPECT_FLOAT_EQ(fmod(3.0, 2.0), f.val_);
  EXPECT_FLOAT_EQ(1.0 * -floor(3.0 / 2.0), f.d_);
 }

TEST(AgradFvar, fmin) {
  using stan::agrad::fvar;
  using stan::agrad::fmin;
  using std::isnan;

  fvar<double> x(2.0);
  fvar<double> y(-3.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = fmin(x, y);
  EXPECT_FLOAT_EQ(-3.0, a.val_);
  EXPECT_FLOAT_EQ(2.0, a.d_);

  fvar<double> b = fmin(2 * x, y);
  EXPECT_FLOAT_EQ(-3.0, b.val_);
  EXPECT_FLOAT_EQ(2.0, b.d_);

  fvar<double> c = fmin(y, x);
  EXPECT_FLOAT_EQ(-3.0, c.val_);
  EXPECT_FLOAT_EQ(2.0, c.d_);

  fvar<double> d = fmin(x, x);
  EXPECT_FLOAT_EQ(2.0, d.val_);
  isnan(d.d_);

  double z = 1.0;

  fvar<double> e = fmin(x, z);
  EXPECT_FLOAT_EQ(1.0, e.val_);
  EXPECT_FLOAT_EQ(0.0, e.d_);

  fvar<double> f = fmin(z, x);
  EXPECT_FLOAT_EQ(1.0, f.val_);
  EXPECT_FLOAT_EQ(0.0, f.d_);
 }

TEST(AgradFvar, fmax) {
  using stan::agrad::fvar;
  using stan::agrad::fmax;
  using std::isnan;

  fvar<double> x(2.0);
  fvar<double> y(-3.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = fmax(x, y);
  EXPECT_FLOAT_EQ(2.0, a.val_);
  EXPECT_FLOAT_EQ(1.0, a.d_);

  fvar<double> b = fmax(2 * x, y);
  EXPECT_FLOAT_EQ(4.0, b.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, b.d_);

  fvar<double> c = fmax(y, x);
  EXPECT_FLOAT_EQ(2.0, c.val_);
  EXPECT_FLOAT_EQ(1.0, c.d_);

  fvar<double> d = fmax(x, x);
  EXPECT_FLOAT_EQ(2.0, d.val_);
  isnan(d.d_);

  double z = 1.0;

  fvar<double> e = fmax(x, z);
  EXPECT_FLOAT_EQ(2.0, e.val_);
  EXPECT_FLOAT_EQ(1.0, e.d_);

  fvar<double> f = fmax(z, x);
  EXPECT_FLOAT_EQ(2.0, f.val_);
  EXPECT_FLOAT_EQ(1.0, f.d_);
 }

TEST(AgradFvar, sqrt) {
  using stan::agrad::fvar;
  using std::sqrt;
  using std::isnan;

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
  isnan(g.d_); 
}

TEST(AgradFvar, cbrt) {
  using stan::agrad::fvar;
  using boost::math::cbrt;
  using std::isnan;

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
  isnan(f.d_);
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
  isnan(f.val_);
  isnan(f.d_);
}

TEST(AgradFvar, log2) {
  using stan::agrad::fvar;
  using std::log;
  using std::isnan;
  using stan::math::log2;

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
  isnan(f.val_);
  isnan(f.d_);
}

TEST(AgradFvar, log10) {
  using stan::agrad::fvar;
  using std::log;
  using std::log10;
  using std::isnan;

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
  isnan(f.val_);
  isnan(f.d_);
}

TEST(AgradFvar, pow) {
  using stan::agrad::fvar;
  using std::pow;
  using std::log;
  using std::isnan;

  fvar<double> x(0.5);
  x.d_ = 1.0;
  double y = 5.0;

  fvar<double> a = pow(x, y);
  EXPECT_FLOAT_EQ(pow(0.5, 5.0), a.val_);
  EXPECT_FLOAT_EQ(5.0 * pow(0.5, 5.0 - 1.0), a.d_);

  fvar<double> b = pow(y, x);
  EXPECT_FLOAT_EQ(pow(5.0, 0.5), b.val_);
  EXPECT_FLOAT_EQ(log(5.0) * pow(5.0, 0.5), b.d_);

  fvar<double> z(1.2);
  z.d_ = 2.0;
  fvar<double> c = pow(x, z);
  EXPECT_FLOAT_EQ(pow(0.5, 1.2), c.val_);
  EXPECT_FLOAT_EQ((2.0 * log(0.5) + 1.2 * 1.0 / 0.5) * pow(0.5, 1.2), c.d_);

  fvar<double> w(-0.4);
  w.d_ = 1.0;
  fvar<double> d = pow(w, x);
  isnan(d.val_);
  isnan(d.d_);
}

TEST(AgradFvar, hypot) {
  using stan::agrad::fvar;
  using boost::math::hypot;
  using std::isnan;

  fvar<double> x(0.5);
  fvar<double> y(2.3);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = hypot(x, y);
  EXPECT_FLOAT_EQ(hypot(0.5, 2.3), a.val_);
  EXPECT_FLOAT_EQ((0.5 * 1.0 + 2.3 * 2.0) / hypot(0.5, 2.3), a.d_);

  fvar<double> z(0.0);
  fvar<double> w(-2.3);
  z.d_ = 1.0;
  w.d_ = 2.0;
  fvar<double> b = hypot(x, z);
  EXPECT_FLOAT_EQ(0.5, b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<double> c = hypot(x, w);
  isnan(c.val_);
  isnan(c.d_);
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

TEST(AgradFvar, atan2) {
  using stan::agrad::fvar;
  using std::atan2;

  fvar<double> x(0.5);
  fvar<double> y(2.3);
  x.d_ = 1.0;
  y.d_ = 2.0;
  double w = 2.1;

  fvar<double> a = atan2(x, y);
  EXPECT_FLOAT_EQ(atan2(0.5, 2.3), a.val_);
  EXPECT_FLOAT_EQ((1.0 * 2.3 - 0.5 * 2.0) / (0.5 * 0.5 + 2.3 * 2.3), a.d_);

  fvar<double> b = atan2(w, x);
  EXPECT_FLOAT_EQ(atan2(2.1, 0.5), b.val_);
  EXPECT_FLOAT_EQ((-2.1 * 1.0) / (2.1 * 2.1 + 0.5 * 0.5), b.d_);

  fvar<double> c = atan2(x, w);
  EXPECT_FLOAT_EQ(atan2(0.5, 2.1), c.val_);
  EXPECT_FLOAT_EQ((1.0 * 2.1) / (0.5 * 0.5 + 2.1 * 2.1), c.d_);
}

TEST(AgradFvar, sinh) {
  using stan::agrad::fvar;
  using std::sinh;
  using std::cosh;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = sinh(x);
  EXPECT_FLOAT_EQ(sinh(0.5), a.val_);
  EXPECT_FLOAT_EQ(cosh(0.5), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = sinh(y);
  EXPECT_FLOAT_EQ(sinh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(cosh(-1.2), b.d_);

  fvar<double> c = sinh(-x);
  EXPECT_FLOAT_EQ(sinh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-cosh(-0.5), c.d_);
}

TEST(AgradFvar, cosh) {
  using stan::agrad::fvar;
  using std::sinh;
  using std::cosh;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = cosh(x);
  EXPECT_FLOAT_EQ(cosh(0.5), a.val_);
  EXPECT_FLOAT_EQ(sinh(0.5), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = cosh(y);
  EXPECT_FLOAT_EQ(cosh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(sinh(-1.2), b.d_);

  fvar<double> c = cosh(-x);
  EXPECT_FLOAT_EQ(cosh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-sinh(-0.5), c.d_);
}

TEST(AgradFvar, tanh) {
  using stan::agrad::fvar;
  using std::tanh;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = tanh(x);
  EXPECT_FLOAT_EQ(tanh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 - tanh(0.5) * tanh(0.5), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = tanh(y);
  EXPECT_FLOAT_EQ(tanh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(1 - tanh(-1.2) * tanh(-1.2), b.d_);

  fvar<double> c = tanh(-x);
  EXPECT_FLOAT_EQ(tanh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-1 * (1 - tanh(-0.5) * tanh(-0.5)), c.d_);
}

TEST(AgradFvar, asinh) {
  using stan::agrad::fvar;
  using boost::math::asinh;
  using std::sqrt;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = asinh(x);
  EXPECT_FLOAT_EQ(asinh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 + (0.5) * (0.5)), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = asinh(y);
  EXPECT_FLOAT_EQ(asinh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 + (-1.2) * (-1.2)), b.d_);

  fvar<double> c = asinh(-x);
  EXPECT_FLOAT_EQ(asinh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-1 / sqrt(1 + (-0.5) * (-0.5)), c.d_);
}

TEST(AgradFvar, acosh) {
  using stan::agrad::fvar;
  using boost::math::acosh;
  using std::sqrt;
  using std::isnan;

  fvar<double> x(1.5);
  x.d_ = 1.0;

  fvar<double> a = acosh(x);
  EXPECT_FLOAT_EQ(acosh(1.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(-1 + (1.5) * (1.5)), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = acosh(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> z(0.5);
  z.d_ = 1.0;

  fvar<double> c = acosh(z);
  isnan(c.val_);
  isnan(c.d_);
}

TEST(AgradFvar, atanh) {
  using stan::agrad::fvar;
  using boost::math::atanh;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = atanh(x);
  EXPECT_FLOAT_EQ(atanh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 - 0.5 * 0.5), a.d_);

  fvar<double> y(-0.9);
  y.d_ = 1.0;

  fvar<double> b = atanh(y);
  EXPECT_FLOAT_EQ(atanh(-0.9), b.val_);
  EXPECT_FLOAT_EQ(1 / (1 - 0.9 * 0.9), b.d_);
}

TEST(AgradFvar, logit) {
  using stan::agrad::fvar;
  using stan::math::logit;
  using std::isnan;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = logit(x);
  EXPECT_FLOAT_EQ(logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.5 * 0.5), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = logit(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> z(1.5);
  z.d_ = 1.0;

  fvar<double> c = logit(z);
  isnan(c.val_);
  isnan(c.d_);
}

TEST(AgradFvar, invLogit) {
  using stan::agrad::fvar;
  using stan::math::inv_logit;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = inv_logit(x);
  EXPECT_FLOAT_EQ(inv_logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(inv_logit(0.5) * (1 - inv_logit(0.5)), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = inv_logit(y);
  EXPECT_FLOAT_EQ(inv_logit(-1.2), b.val_);
  EXPECT_FLOAT_EQ(inv_logit(-1.2) * (1 - inv_logit(-1.2)), b.d_);

  fvar<double> z(1.5);
  z.d_ = 1.0;

  fvar<double> c = inv_logit(z);
  EXPECT_FLOAT_EQ(inv_logit(1.5), c.val_);
  EXPECT_FLOAT_EQ(inv_logit(1.5) * (1 - inv_logit(1.5)), c.d_);
}

TEST(AgradFvar, invCLogLog) {
  using stan::agrad::fvar;
  using stan::math::inv_cloglog;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = inv_cloglog(x);
  EXPECT_FLOAT_EQ(inv_cloglog(0.5), a.val_);
  EXPECT_FLOAT_EQ(-exp(0.5 - exp(0.5)), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = inv_cloglog(y);
  EXPECT_FLOAT_EQ(inv_cloglog(-1.2), b.val_);
  EXPECT_FLOAT_EQ(-exp(-1.2 - exp(-1.2)), b.d_);

  fvar<double> z(1.5);
  z.d_ = 1.0;

  fvar<double> c = inv_cloglog(z);
  EXPECT_FLOAT_EQ(inv_cloglog(1.5), c.val_);
  EXPECT_FLOAT_EQ(-exp(1.5 - exp(1.5)), c.d_);
}

TEST(AgradFvar, erf){
  using stan::agrad::fvar;
  using std::exp;
  using std::sqrt;
  using boost::math::erf;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = erf(x);
  EXPECT_FLOAT_EQ(erf(0.5), a.val_);
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / sqrt(boost::math::constants::pi<double>()), a.d_);

 fvar<double> b = erf(-x);
  EXPECT_FLOAT_EQ(erf(-0.5), b.val_);
  EXPECT_FLOAT_EQ(-2 * exp(-0.5 * 0.5) / sqrt(boost::math::constants::pi<double>()), b.d_);
}

TEST(AgradFvar, erfc){
  using stan::agrad::fvar;
  using std::exp;
  using std::sqrt;
  using boost::math::erfc;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = erfc(x);
  EXPECT_FLOAT_EQ(erfc(0.5), a.val_);
  EXPECT_FLOAT_EQ(-2 * exp(-0.5 * 0.5) / sqrt(boost::math::constants::pi<double>()), a.d_);

 fvar<double> b = erfc(-x);
  EXPECT_FLOAT_EQ(erfc(-0.5), b.val_);
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / sqrt(boost::math::constants::pi<double>()), b.d_);
}


TEST(AgradFvar, expm1) {
  using stan::agrad::fvar;
  using boost::math::expm1;
  using std::exp;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = expm1(x);
  EXPECT_FLOAT_EQ(expm1(0.5), a.val_);
  EXPECT_FLOAT_EQ(exp(0.5), a.d_);

  fvar<double> b = 2 * expm1(x) + 4;
  EXPECT_FLOAT_EQ(2 * expm1(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 * exp(0.5), b.d_);

  fvar<double> c = -expm1(x) + 5;
  EXPECT_FLOAT_EQ(-expm1(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-exp(0.5), c.d_);

  fvar<double> d = -3 * expm1(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * expm1(-0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(3 * exp(-0.5) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = expm1(y);
  EXPECT_FLOAT_EQ(expm1(-0.5), e.val_);
  EXPECT_FLOAT_EQ(exp(-0.5), e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = expm1(z);
  EXPECT_FLOAT_EQ(expm1(0.0), f.val_);
  EXPECT_FLOAT_EQ(exp(0.0), f.d_);
}

TEST(AgradFvar, fma) { 
  using stan::agrad::fvar;
  using stan::math::fma;
  fvar<double> x(0.5);
  fvar<double> y(1.2);
  fvar<double> z(1.8);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  double p = 1.4;
  double q = 2.3;

  fvar<double> a = fma(x, y, z);
  EXPECT_FLOAT_EQ(fma(0.5, 1.2, 1.8), a.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.2 + 2.0 * 0.5 + 3.0, a.d_);

  fvar<double> b = fma(p, y, z);
  EXPECT_FLOAT_EQ(fma(1.4, 1.2, 1.8), b.val_);
  EXPECT_FLOAT_EQ(2.0 * 1.4 + 3.0, b.d_);

  fvar<double> c = fma(x, p, z);
  EXPECT_FLOAT_EQ(fma(0.5, 1.4, 1.8), c.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.4 + 3.0, c.d_);

  fvar<double> d = fma(x, y, p);
  EXPECT_FLOAT_EQ(fma(0.5, 1.2, 1.4), d.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.2 + 2.0 * 0.5, d.d_);

  fvar<double> e = fma(p, q, z);
  EXPECT_FLOAT_EQ(fma(1.4, 2.3, 1.8), e.val_);
  EXPECT_FLOAT_EQ(3.0, e.d_);

  fvar<double> f = fma(x, p, q);
  EXPECT_FLOAT_EQ(fma(0.5, 1.4, 2.3), f.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.4, f.d_);

  fvar<double> g = fma(q, y, p);
  EXPECT_FLOAT_EQ(fma(2.3, 1.2, 1.4), g.val_);
  EXPECT_FLOAT_EQ(2.0 * 2.3, g.d_);
}

TEST(AgradFvar,multiply_log) {
  using stan::agrad::fvar;
  using std::isnan;
  using std::log;
  using stan::math::multiply_log;

  fvar<double> x(0.5);
  fvar<double> y(1.2);
  fvar<double> z(-0.4);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  double w = 0.0;
  double v = 1.3;

  fvar<double> a = multiply_log(x, y);
  EXPECT_FLOAT_EQ(multiply_log(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ(1.0 * log(1.2) + 0.5 * 2.0 / 1.2, a.d_);

  fvar<double> b = multiply_log(x,z);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = multiply_log(x, v);
  EXPECT_FLOAT_EQ(multiply_log(0.5, 1.3), c.val_);
  EXPECT_FLOAT_EQ(log(1.3), c.d_);

  fvar<double> d = multiply_log(v, x);
  EXPECT_FLOAT_EQ(multiply_log(1.3, 0.5), d.val_);
  EXPECT_FLOAT_EQ(1.3 * 1.0 / 0.5, d.d_);

  fvar<double> e = multiply_log(x, w);
  isnan(e.val_);
  isnan(e.d_);
}

TEST(AgradFvar, log1p){
  using stan::agrad::fvar;
  using stan::math::log1p;
  using std::isnan;

  fvar<double> x(0.5);
  fvar<double> y(-1.0);
  fvar<double> z(-2.0);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  fvar<double> a = log1p(x);
  EXPECT_FLOAT_EQ(log1p(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 + 0.5), a.d_);

  fvar<double> b = log1p(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = log1p(z);
  isnan(c.val_);
  isnan(c.d_);
}

TEST(AgradFvar, log1m){
  using stan::agrad::fvar;
  using stan::math::log1m;
  using std::isnan;

  fvar<double> x(0.5);
  fvar<double> y(1.0);
  fvar<double> z(2.0);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  fvar<double> a = log1m(x);
  EXPECT_FLOAT_EQ(log1m(0.5), a.val_);
  EXPECT_FLOAT_EQ(-1 / (1 - 0.5), a.d_);

  fvar<double> b = log1m(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = log1m(z);
  isnan(c.val_);
  isnan(c.d_);
}

TEST(AgradFvar, log1p_exp){
  using stan::agrad::fvar;
  using stan::math::log1p_exp;
  using std::exp;

  fvar<double> x(0.5);
  fvar<double> y(1.0);
  fvar<double> z(2.0);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  fvar<double> a = log1p_exp(x);
  EXPECT_FLOAT_EQ(log1p_exp(0.5), a.val_);
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)), a.d_);

  fvar<double> b = log1p_exp(y);
  EXPECT_FLOAT_EQ(log1p_exp(1.0), b.val_);
  EXPECT_FLOAT_EQ(2.0 * exp(1.0) / (1 + exp(1.0)), b.d_);
}

TEST(AgradFvar, log_sum_exp) {
  using stan::agrad::fvar;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<double> x(0.5);
  fvar<double> y(1.2);
  x.d_ = 1.0;
  y.d_ = 2.0;

  double z = 1.4;

  fvar<double> a = log_sum_exp(x, y);
  EXPECT_FLOAT_EQ(log_sum_exp(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ((1.0 * exp(0.5) + 2.0 * exp(1.2)) / (exp(0.5) + exp(1.2)), a.d_);

  fvar<double> b = log_sum_exp(x, z);
  EXPECT_FLOAT_EQ(log_sum_exp(0.5, 1.4), b.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(0.5) / (exp(0.5) + exp(1.4)), b.d_);

  fvar<double> c = log_sum_exp(z, x);
  EXPECT_FLOAT_EQ(log_sum_exp(1.4, 0.5), c.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(0.5) / (exp(0.5) + exp(1.4)), c.d_);
}

TEST(AgradFvar, log_inv_logit){
  using stan::agrad::fvar;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<double> x(0.5);
  fvar<double> y(-1.0);
  fvar<double> z(0.0);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  fvar<double> a = log_inv_logit(x);
  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(-0.5) / (1 + exp(-0.5)), a.d_);

  fvar<double> b = log_inv_logit(y);
  EXPECT_FLOAT_EQ(log_inv_logit(-1.0), b.val_);
  EXPECT_FLOAT_EQ(2.0 * exp(1.0) / (1 + exp(1.0)), b.d_);

  fvar<double> c = log_inv_logit(z);
  EXPECT_FLOAT_EQ(log_inv_logit(0.0), c.val_);
  EXPECT_FLOAT_EQ(3.0 * exp(0.0) / (1 + exp(0.0)), c.d_);
}

TEST(AgradFvar, log1m_inv_logit){
  using stan::agrad::fvar;
  using stan::math::log1m_inv_logit;
  using std::exp;

  fvar<double> x(0.5);
  fvar<double> y(-1.0);
  fvar<double> z(0.0);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  fvar<double> a = log1m_inv_logit(x);
  EXPECT_FLOAT_EQ(log1m_inv_logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(-1.0 * exp(0.5) / (1 + exp(0.5)), a.d_);

  fvar<double> b = log1m_inv_logit(y);
  EXPECT_FLOAT_EQ(log1m_inv_logit(-1.0), b.val_);
  EXPECT_FLOAT_EQ(-2.0 * exp(-1.0) / (1 + exp(-1.0)), b.d_);

  fvar<double> c = log1m_inv_logit(z);
  EXPECT_FLOAT_EQ(log1m_inv_logit(0.0), c.val_);
  EXPECT_FLOAT_EQ(-3.0 * exp(0.0) / (1 + exp(0.0)), c.d_);
}

TEST(AgradFvar, tgamma){
  using stan::agrad::fvar;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = tgamma(x);
  EXPECT_FLOAT_EQ(tgamma(0.5), a.val_);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), a.d_);
}

TEST(AgradFvar, lgamma){
  using stan::agrad::fvar;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = lgamma(x);
  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_);
  EXPECT_FLOAT_EQ(digamma(0.5), a.d_);
}

TEST(AgradFvar, lbeta) {
  using stan::agrad::fvar;
  using boost::math::tgamma;
  using stan::math::lbeta;

  fvar<double> x(0.5);
  fvar<double> y(1.2);
  x.d_ = 1.0;
  y.d_ = 2.0;

  double w = 1.3;

  fvar<double> a = lbeta(x, y);
  EXPECT_FLOAT_EQ(lbeta(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(0.5) + 2.0 / tgamma(1.2) - (1.0 + 2.0) / tgamma(0.5 + 1.2), a.d_);

  fvar<double> b = lbeta(x, w);
  EXPECT_FLOAT_EQ(lbeta(0.5, 1.3), b.val_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(0.5) - 1.0 / tgamma(0.5 + 1.3), b.d_);

  fvar<double> c = lbeta(w, x);
  EXPECT_FLOAT_EQ(lbeta(1.3, 0.5), c.val_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(0.5) - 1.0 / tgamma(1.3 + 0.5), c.d_);
}

TEST(AgradFvar, binom_coeff_log) {
  using stan::agrad::fvar;
  using stan::math::binomial_coefficient_log;
  using boost::math::digamma;
  using boost::math::tgamma;

  fvar<double> x(2004.0);
  x.d_ = 1.0;
  fvar<double> y(1002.0);
  y.d_ = 2.0;

  fvar<double> a = binomial_coefficient_log(x, y);
  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0, 1002.0), a.val_);
  EXPECT_FLOAT_EQ(1.0 * digamma(2004.0 + 1) / gamma(2004.0 + 1) 
        - 2.0 * digamma(1002.0 + 1) / gamma(1002.0 + 1) 
         + (1.0 - 2.0) * digamma(2004.0 - 1002.0 + 1) / gamma(2004.0 - 1002.0 + 1), a.d_);

  double z = 1003.0;

  fvar<double> b = binomial_coefficient_log(x, z);
  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0, 1003.0), b.val_);
  EXPECT_FLOAT_EQ(1.0 * digamma(2004.0 + 1) / gamma(2004.0 + 1) + 1.0 * digamma(2004.0 - 1003.0 + 1) / gamma(2004.0 - 1003.0 + 1), b.d_);

  double w = 2006.0;

  fvar<double> c = binomial_coefficient_log(w, y);
  EXPECT_FLOAT_EQ(binomial_coefficient_log(2006.0, 1002.0), c.val_);
  EXPECT_FLOAT_EQ( -1.0 * digamma(1002.0 + 1) / gamma(1002.0 + 1) - 1.0 * digamma(2006.0 - 1002.0 + 1) / gamma(2006.0 - 1002.0 + 1), c.d_);

  double q = 1001.3;

  fvar<double> d = binomial_coefficient_log(x, q);
  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0, 1001.3), d.val_);
  EXPECT_FLOAT_EQ(1.0 * digamma(2004.0 + 1) / gamma(2004.0 + 1) + 1.0 * digamma(2004.0 - 1001.3 + 1) / gamma(2004.0 - 1001.3 + 1), d.d_);
}
