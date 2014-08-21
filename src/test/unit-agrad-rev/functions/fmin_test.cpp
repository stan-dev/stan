#include <stan/agrad/rev/functions/fmin.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(AgradRev,fmin_vv) {
  AVAR a = 1.3;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(1.3,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
  EXPECT_FLOAT_EQ(0.0,grad_f[1]);
}  

TEST(AgradRev,fmin_vv_2) {
  AVAR a = 2.3;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}  

TEST(AgradRev,fmin_vv_3) {
  AVAR a = 2.0;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but documented this way
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}  

TEST(AgradRev,fmin_vd) {
  AVAR a = 1.3;
  double b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(1.3,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(AgradRev,fmin_vd_2) {
  AVAR a = 2.3;
  double b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(AgradRev,fmin_vd_3) {
  AVAR a = 2.0;
  double b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(AgradRev,fmin_dv) {
  double a = 1.3;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(1.3,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(AgradRev,fmin_dv_2) {
  double a = 2.3;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(AgradRev,fmin_dv_3) {
  double a = 2.0;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(AgradRev,fmin_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  stan::agrad::var nan_v = std::numeric_limits<double>::quiet_NaN();
  stan::agrad::var a_v = 0.0;
  double a = 0.0;

  EXPECT_FLOAT_EQ(0.0, stan::agrad::fmin(nan_v, a).val());
  EXPECT_FLOAT_EQ(0.0, stan::agrad::fmin(nan_v, a_v).val());
  EXPECT_FLOAT_EQ(0.0, stan::agrad::fmin(nan, a_v).val());
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::agrad::fmin(a, nan_v).val());
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::agrad::fmin(a_v, nan_v).val());
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::agrad::fmin(a_v, nan).val());
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::agrad::fmin(nan, nan_v).val());
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::agrad::fmin(nan_v, nan).val());
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::agrad::fmin(nan_v, nan_v).val());
}
