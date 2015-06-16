#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

TEST(AgradRev,a_div_b) {
  AVAR a = 6.0;
  AVAR b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/3.0,g[0]);
  EXPECT_FLOAT_EQ(-6.0/(3.0*3.0),g[1]);
}

TEST(AgradRev,a_divide_bd) {
  AVAR a = 6.0;
  double b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/3.0,g[0]);
}

TEST(AgradRev,ad_divide_b) {
  double a = 6.0;
  AVAR b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-6.0/(3.0*3.0),g[0]);
}

struct divide_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return arg1/arg2;
  }
};

TEST(AgradRev, divide_nan) {
  divide_fun divide_;
  test_nan(divide_,3.0,5.0,false, true);
}
