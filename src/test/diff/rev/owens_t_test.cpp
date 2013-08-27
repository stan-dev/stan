#include <stan/diff/rev/owens_t.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffSpecialFunctions,owens_t_vv) {
  using stan::diff::var;
  using stan::diff::owens_t;
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
TEST(DiffSpecialFunctions,owens_t_vd) {
  using stan::diff::var;
  using stan::diff::owens_t;
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
TEST(DiffSpecialFunctions,owens_t_dv) {
  using stan::diff::var;
  using stan::diff::owens_t;
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
