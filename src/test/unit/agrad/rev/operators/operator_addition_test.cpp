#include <stan/agrad/rev/operators/operator_addition.hpp>
#include <stan/agrad/rev/operators/operator_unary_negative.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/rev/nan_util.hpp>
#include <stan/meta/traits.hpp>

TEST(AgradRev,a_plus_b) {
  AVAR a = 5.0;
  AVAR b = -1.0;
  AVAR f = a + b;
  EXPECT_FLOAT_EQ(4.0,f.val());
  AVEC x = createAVEC(a,b);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
  EXPECT_FLOAT_EQ(1.0,dx[1]);
}

TEST(AgradRev,a_plus_a) {
  AVAR a = 5.0;
  AVAR f = a + a;
  EXPECT_FLOAT_EQ(10.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(2.0,dx[0]);
}

TEST(AgradRev,a_plus_neg_b) {
  AVAR a = 5.0;
  AVAR b = -1.0;
  AVAR f = a + -b;
  EXPECT_FLOAT_EQ(6.0,f.val());
  AVEC x = createAVEC(a,b);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
  EXPECT_FLOAT_EQ(-1.0,dx[1]);
}

TEST(AgradRev,a_plus_x) {
  AVAR a = 5.0;
  double z = 3.0;
  AVAR f = a + z;
  EXPECT_FLOAT_EQ(8.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}

TEST(AgradRev,x_plus_a) {
  AVAR a = 5.0;
  double z = 3.0;
  AVAR f = z + a;
  EXPECT_FLOAT_EQ(8.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}

struct add_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return arg1+arg2;
  }
};

TEST(AgradRev, add_nan) {
  add_fun add_;
  test_nan(add_,3.0,5.0,false, true);
}
