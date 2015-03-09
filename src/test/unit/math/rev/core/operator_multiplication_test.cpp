#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

TEST(AgradRev,a_times_b) {
  AVAR a = 2.0;
  AVAR b = -3.0;
  AVAR f = a * b;
  EXPECT_FLOAT_EQ(-6.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-3.0,grad_f[0]);
  EXPECT_FLOAT_EQ(2.0,grad_f[1]);
}

TEST(AgradRev,a_times_a) {
  AVAR a = 2.0;
  AVAR f = a * a;
  EXPECT_FLOAT_EQ(4.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(4.0,grad_f[0]);
}

TEST(AgradRev,a_times_y) {
  AVAR a = 2.0;
  double y = -3.0;
  AVAR f = a * y;
  EXPECT_FLOAT_EQ(-6.0,f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-3.0,g[0]);
}
 
TEST(AgradRev,x_times_b) {
  double x = 2.0;
  AVAR b = -3.0;
  AVAR f = x * b;
  EXPECT_FLOAT_EQ(-6.0,f.val());

  AVEC v = createAVEC(b);
  VEC g;
  f.grad(v,g);
  EXPECT_FLOAT_EQ(2.0,g[0]);
}

struct multiply_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return arg1*arg2;
  }
};

TEST(AgradRev, multiply_nan) {
  multiply_fun multiply_;
  test_nan(multiply_,3.0,5.0,false, true);
}
