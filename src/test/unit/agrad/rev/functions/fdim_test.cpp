#include <stan/agrad/rev/functions/fdim.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev.hpp>
#include <stan/math/functions/fdim.hpp>
#include <test/unit/agrad/rev/nan_util.hpp>
#include <stan/meta/traits.hpp>

TEST(AgradRev,fdim_vv) {
  using stan::agrad::fdim;
  AVAR a = 3.0;
  AVAR b = 4.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(0.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
  EXPECT_FLOAT_EQ(0.0,grad_f[1]);

  a = std::numeric_limits<AVAR>::infinity();
  EXPECT_FLOAT_EQ(0.0, fdim(a,a).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), fdim(a,b).val());
  EXPECT_FLOAT_EQ(0.0, fdim(b,a).val());
}  
TEST(AgradRev,fdim_vv_2) {
  using stan::agrad::fdim;
  AVAR a = 7.0;
  AVAR b = 2.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
  EXPECT_FLOAT_EQ(-1.0,grad_f[1]);

  a = std::numeric_limits<AVAR>::infinity();
  EXPECT_FLOAT_EQ(0.0, fdim(a,a).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), fdim(a,b).val());
  EXPECT_FLOAT_EQ(0.0, fdim(b,a).val());
}  
TEST(AgradRev,fdim_vd) {
  using stan::agrad::fdim;
  AVAR a = 3.0;
  double b = 4.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(0.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);

  AVAR infinityavar = std::numeric_limits<AVAR>::infinity();
  double infinitydouble = std::numeric_limits<double>::infinity();
  EXPECT_FLOAT_EQ(0.0, fdim(infinityavar,infinitydouble).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), fdim(infinityavar,b).val());
  EXPECT_FLOAT_EQ(0.0, fdim(a,infinitydouble).val());
}  

TEST(AgradRev,fdim_vd_2) {
  using stan::agrad::fdim;
  AVAR a = 7.0;
  double b = 2.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);

  AVAR infinityavar = std::numeric_limits<AVAR>::infinity();
  double infinitydouble = std::numeric_limits<double>::infinity();
  EXPECT_FLOAT_EQ(0.0, fdim(infinityavar,infinitydouble).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), fdim(infinityavar,b).val());
  EXPECT_FLOAT_EQ(0.0, fdim(a,infinitydouble).val());
}  
TEST(AgradRev,fdim_dv) {
  using stan::agrad::fdim;
  double a = 3.0;
  AVAR b = 4.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(0.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
 
  AVAR infinityavar = std::numeric_limits<AVAR>::infinity();
  double infinitydouble = std::numeric_limits<double>::infinity();
  EXPECT_FLOAT_EQ(0.0, fdim(infinitydouble,infinityavar).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), fdim(infinitydouble,b).val());
  EXPECT_FLOAT_EQ(0.0, fdim(a,infinityavar).val());
}
TEST(AgradRev,fdim_dv_2) {
  using stan::agrad::fdim;
  double a = 7.0;
  AVAR b = 2.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-1.0,grad_f[0]);

  AVAR infinityavar = std::numeric_limits<AVAR>::infinity();
  double infinitydouble = std::numeric_limits<double>::infinity();
  EXPECT_FLOAT_EQ(0.0, fdim(infinitydouble,infinityavar).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), fdim(infinitydouble,b).val());
  EXPECT_FLOAT_EQ(0.0, fdim(a,infinityavar).val());
}  

struct fdim_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return fdim(arg1,arg2);
  }
};

TEST(AgradRev, fdim_nan) {
  fdim_fun fdim_;
  test_nan(fdim_,3.0,5.0,false, true);
}
