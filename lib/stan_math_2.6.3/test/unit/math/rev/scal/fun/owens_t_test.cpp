#include <stan/math/rev/scal/fun/owens_t.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

TEST(AgradRev,owens_t_vv) {
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  var h = 1.0;
  var a = 2.0;
  var f = owens_t(h,a);
  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val());

  AVEC x = createAVEC(h,a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0026128467,grad_f[1]);
  EXPECT_FLOAT_EQ(-0.1154804963,grad_f[0]);
}
TEST(AgradRev,owens_t_vd) {
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  AVAR h = 1.0;
  double a = 2.0;
  AVAR f = owens_t(h,a);
  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val());

  AVEC x = createAVEC(h,a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0,grad_f[1]);
  EXPECT_FLOAT_EQ(-0.1154804963,grad_f[0]);
}
TEST(AgradRev,owens_t_dv) {
  using stan::math::var;
  using stan::math::owens_t;
  using boost::math::owens_t;

  double h = 1.0;
  AVAR a = 2.0;
  AVAR f = owens_t(h,a);
  EXPECT_FLOAT_EQ(owens_t(1.0, 2.0), f.val());

  AVEC x = createAVEC(h,a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0026128467,grad_f[1]);
  EXPECT_FLOAT_EQ(0,grad_f[0]);
}

struct owens_t_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return owens_t(arg1,arg2);
  }
};

TEST(AgradRev, owens_t_nan) {
  owens_t_fun owens_t_;
  test_nan(owens_t_,1.0,2.0,false, true);
}
