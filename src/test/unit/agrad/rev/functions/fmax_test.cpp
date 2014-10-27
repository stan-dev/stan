#include <stan/agrad/rev/functions/fmax.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/meta/traits.hpp>
#include <test/unit/agrad/rev/nan_util.hpp>

TEST(AgradRev,fmax_vv) {
  AVAR a = 1.3;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}  

TEST(AgradRev,fmax_vv_2) {
  AVAR a = 2.3;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.3,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
  EXPECT_FLOAT_EQ(0.0,grad_f[1]);
}  

TEST(AgradRev,fmax_vv_3) {
  AVAR a = 2.0;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but documented this way
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}  

TEST(AgradRev,fmax_vd) {
  AVAR a = 1.3;
  double b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(AgradRev,fmax_vd_2) {
  AVAR a = 2.3;
  double b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.3,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(AgradRev,fmax_vd_3) {
  AVAR a = 2.0;
  double b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(AgradRev,fmax_dv) {
  double a = 1.3;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(AgradRev,fmax_dv_2) {
  double a = 2.3;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.3,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(AgradRev,fmax_dv_3) {
  double a = 2.0;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

struct fmax_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return fmax(arg1,arg2);
  }
};

TEST(AgradRev, fmax_nan) {
  fmax_fun fmax_;
  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_vv(fmax_,nan,nan,false, true);
  test_nan_dv(fmax_,nan,nan,false, true);
  test_nan_vd(fmax_,nan,nan,false, true);
}
