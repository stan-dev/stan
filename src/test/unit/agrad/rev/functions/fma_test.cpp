#include <cmath>
#include <stan/agrad/rev/functions/fma.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/rev/nan_util.hpp>
#include <stan/meta/traits.hpp>

TEST(AgradRev,fma_vvv) {
  AVAR a = 3.0;
  AVAR b = 5.0;
  AVAR c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(a,b,c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(5.0,grad_f[0]);
  EXPECT_FLOAT_EQ(3.0,grad_f[1]);
  EXPECT_FLOAT_EQ(1.0,grad_f[2]);
}
TEST(AgradRev,fma_vvd) {
  AVAR a = 3.0;
  AVAR b = 5.0;
  double c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(5.0,grad_f[0]);
  EXPECT_FLOAT_EQ(3.0,grad_f[1]);
}  
TEST(AgradRev,fma_vdv) {
  AVAR a = 3.0;
  double b = 5.0;
  AVAR c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(a,c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(5.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
} 
TEST(AgradRev,fma_vdd) {
  AVAR a = 3.0;
  double b = 5.0;
  double c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(5.0,grad_f[0]);
}  
TEST(AgradRev,fma_dvv) {
  double a = 3.0;
  AVAR b = 5.0;
  AVAR c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(b,c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(3.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}
TEST(AgradRev,fma_dvd) {
  double a = 3.0;
  AVAR b = 5.0;
  double c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(3.0,grad_f[0]);
}  
TEST(AgradRev,fma_ddv) {
  double a = 3.0;
  double b = 5.0;
  AVAR c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

struct fma_fun {
  template <typename T0, typename T1, typename T2>
  inline
  typename stan::return_type<T0,T1,T2>::type
  operator()(const T0& arg1,
             const T1& arg2,
             const T2& arg3) const {
    return fma(arg1,arg2,arg3);
  }
};

TEST(AgradRev,fma_NaN) {
  fma_fun fma_;
  test_nan(fma_,0.6,0.3,0.5,false,true);
}
