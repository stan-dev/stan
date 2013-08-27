#include <stan/diff/rev/hypot.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,hypot_vv) {
  AVAR a = 3.0;
  AVAR b = 4.0;
  AVAR f = hypot(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(3.0/5.0,grad_f[0]);
  EXPECT_FLOAT_EQ(4.0/5.0,grad_f[1]);
}  

TEST(DiffRev,hypot_vd) {
  AVAR a = 3.0;
  double b = 4.0;
  AVAR f = hypot(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(3.0/5.0,grad_f[0]);
}  

TEST(DiffRev,hypot_dv) {
  double a = 3.0;
  AVAR b = 4.0;
  AVAR f = hypot(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(4.0/5.0,grad_f[0]);
}  
