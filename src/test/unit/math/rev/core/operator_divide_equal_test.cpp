#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

TEST(AgradRev,a_divideeq_b) {
  AVAR a(6.0);
  AVAR b(-2.0);
  AVEC x = createAVEC(a,b);
  AVAR f = (a /= b);
  EXPECT_FLOAT_EQ(-3.0,f.val());
  EXPECT_FLOAT_EQ(-3.0,a.val());
  EXPECT_FLOAT_EQ(-2.0,b.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/-2.0,g[0]);
  EXPECT_FLOAT_EQ(-6.0/((-2.0)*(-2.0)),g[1]);
}

TEST(AgradRev,a_divideeq_bd) {
  AVAR a(6.0);
  double b = -2.0;
  AVEC x = createAVEC(a);
  AVAR f = (a /= b);
  EXPECT_FLOAT_EQ(-3.0,f.val());
  EXPECT_FLOAT_EQ(-3.0,a.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/-2.0,g[0]);
}

struct div_eq_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(T0 arg1,
             T1 arg2) const {
    return (arg1 /= arg2);
  }
};

TEST(AgradRev, div_eq_nan) {
  div_eq_fun div_eq_;
  double nan = std::numeric_limits<double>::quiet_NaN();

  test_nan_vv(div_eq_,3.0,nan,false, true);
  test_nan_vv(div_eq_,nan,5.0,false, true);
  test_nan_vv(div_eq_,nan,nan,false, true);
  test_nan_vd(div_eq_,3.0,nan,false, true);
  test_nan_vd(div_eq_,nan,5.0,false, true);
  test_nan_vd(div_eq_,nan,nan,false, true);
}
