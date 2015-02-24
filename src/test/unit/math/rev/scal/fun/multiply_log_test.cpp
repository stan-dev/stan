#include <stan/math/rev/scal/fun/multiply_log.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradRev,multiplyLogChainVV) {
  AVAR a = 19.7;
  AVAR b = 1299.1;
  AVAR f = 2.0 * multiply_log(a,b);

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);

  EXPECT_FLOAT_EQ(2.0 * std::log(b.val()), grad_f[0]);
  EXPECT_FLOAT_EQ(2.0 * a.val() / b.val(), grad_f[1]);
}
TEST(AgradRev,multiplyLogChainDV) {
  double a = 19.7;
  AVAR b = 1299.1;
  AVAR f = 2.0 * multiply_log(a,b);

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);

  EXPECT_FLOAT_EQ(2.0 * a / b.val(), grad_f[0]);
}
TEST(AgradRev,multiplyLogChainVD) {
  AVAR a = 19.7;
  double b = 1299.1;
  AVAR f = 2.0 * multiply_log(a,b);

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);

  EXPECT_FLOAT_EQ(2.0 * std::log(b), grad_f[0]);
}
TEST(AgradRev,multiply_log_var_var) {
  AVAR a = 2.2;
  AVAR b = 3.3;
  AVAR f = multiply_log(a,b);
  EXPECT_FLOAT_EQ(2.2*std::log(3.3),f.val()) << "Reasonable values";

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(std::log(b.val()),g[0]);
  EXPECT_FLOAT_EQ(a.val()/b.val(),g[1]);

  a = 0.0;
  b = 0.0;
  f = multiply_log(a,b);
  EXPECT_FLOAT_EQ(0.0,f.val()) << "a and b both 0";

  x = createAVEC(a,b);
  g.resize(0);
  f.grad(x,g);
  EXPECT_FLOAT_EQ(std::log(b.val()),g[0]);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),g[1]);
}

TEST(AgradRev,multiply_log_var_double) {
  AVAR a = 2.2;
  double b = 3.3;
  AVAR f = multiply_log(a,b);
  EXPECT_FLOAT_EQ(2.2*std::log(3.3),f.val()) << "Reasonable values";

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(std::log(b),g[0]);

  a = 0.0;
  b = 0.0;
  f = multiply_log(a,b);
  EXPECT_FLOAT_EQ(0.0,f.val()) << "a and b both 0";

  x = createAVEC(a);
  g.resize(0);
  f.grad(x,g);
  EXPECT_FLOAT_EQ(std::log(b),g[0]);
}
TEST(AgradRev,multiply_log_double_var) {
  double a = 2.2;
  AVAR b = 3.3;
  AVAR f = multiply_log(a,b);
  EXPECT_FLOAT_EQ(2.2*std::log(3.3),f.val()) << "Reasonable values";

  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(a/b.val(),g[0]);

  a = 0.0;
  b = 0.0;
  f = multiply_log(a,b);
  EXPECT_FLOAT_EQ(0.0,f.val()) << "a and b both 0";

  x = createAVEC(b);
  g.resize(0);
  f.grad(x,g);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),g[0]);
}

struct multiply_log_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return multiply_log(arg1,arg2);
  }
};

TEST(AgradRev, multiply_log_nan) {
  multiply_log_fun multiply_log_;
  test_nan(multiply_log_,3.0,5.0,false, true);
}
