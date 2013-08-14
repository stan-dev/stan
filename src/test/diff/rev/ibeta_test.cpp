#include <stan/diff/rev/ibeta.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/ibeta.hpp>
#include <boost/math/special_functions/beta.hpp>

TEST(DiffRev,ibeta_vvv) {
  using stan::diff::var;
  using stan::math::ibeta;
  using stan::diff::ibeta;
  
  using boost::math::ibeta_derivative;

  AVAR a = 0.6;
  AVAR b = 0.3;
  AVAR c = 0.5;
  AVAR f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.3121373, f.val());
  
  AVEC x = createAVEC(a,b,c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.436993,grad_f[0]);
  EXPECT_FLOAT_EQ(0.7779751,grad_f[1]);
  EXPECT_FLOAT_EQ(ibeta_derivative(a.val(), b.val(), c.val()),grad_f[2]);

  a = 3;
  b = 2;
  c = 0.2;
  f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.0272, f.val());
  x = createAVEC(a,b,c);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.03737671,grad_f[0]);
  EXPECT_FLOAT_EQ(0.02507405,grad_f[1]);
  EXPECT_FLOAT_EQ(ibeta_derivative(a.val(), b.val(), c.val()),grad_f[2]);
}
TEST(DiffRev,ibeta_vvd) {
  using stan::diff::var;
  using stan::math::ibeta;
  using stan::diff::ibeta;
  
  using boost::math::ibeta_derivative;

  AVAR a = 0.6;
  AVAR b = 0.3;
  double c = 0.5;
  AVAR f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.3121373, f.val());
  
  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.436993,grad_f[0]);
  EXPECT_FLOAT_EQ(0.7779751,grad_f[1]);
  
  a = 3;
  b = 2;
  c = 0.2;
  f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.0272, f.val());
  x = createAVEC(a,b);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.03737671,grad_f[0]);
  EXPECT_FLOAT_EQ(0.02507405,grad_f[1]);
}
TEST(DiffRev,ibeta_vdv) {
  using stan::diff::var;
  using stan::math::ibeta;
  using stan::diff::ibeta;
  
  using boost::math::ibeta_derivative;

  AVAR a = 0.6;
  double b = 0.3;
  AVAR c = 0.5;
  AVAR f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.3121373, f.val());
  
  AVEC x = createAVEC(a,c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.436993,grad_f[0]);
  EXPECT_FLOAT_EQ(ibeta_derivative(a.val(), b, c.val()),grad_f[1]);

  a = 3;
  b = 2;
  c = 0.2;
  f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.0272, f.val());
  x = createAVEC(a,c);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.03737671,grad_f[0]);
  EXPECT_FLOAT_EQ(ibeta_derivative(a.val(), b, c.val()),grad_f[1]);
}
TEST(DiffRev,ibeta_vdd) {
  using stan::diff::var;
  using stan::math::ibeta;
  using stan::diff::ibeta;
  
  using boost::math::ibeta_derivative;

  AVAR a = 0.6;
  double b = 0.3;
  double c = 0.5;
  AVAR f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.3121373, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.436993,grad_f[0]);

  a = 3;
  b = 2;
  c = 0.2;
  f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.0272, f.val());
  x = createAVEC(a);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-0.03737671,grad_f[0]);
}
TEST(DiffRev,ibeta_dvv) {
  using stan::diff::var;
  using stan::math::ibeta;
  using stan::diff::ibeta;
  
  using boost::math::ibeta_derivative;

  double a = 0.6;
  AVAR b = 0.3;
  AVAR c = 0.5;
  AVAR f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.3121373, f.val());
  
  AVEC x = createAVEC(b,c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.7779751,grad_f[0]);
  EXPECT_FLOAT_EQ(ibeta_derivative(a, b.val(), c.val()),grad_f[1]);

  a = 3;
  b = 2;
  c = 0.2;
  f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.0272, f.val());
  x = createAVEC(b,c);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.02507405,grad_f[0]);
  EXPECT_FLOAT_EQ(ibeta_derivative(a, b.val(), c.val()),grad_f[1]);
}
TEST(DiffRev,ibeta_dvd) {
  using stan::diff::var;
  using stan::math::ibeta;
  using stan::diff::ibeta;
  
  using boost::math::ibeta_derivative;

  double a = 0.6;
  AVAR b = 0.3;
  double c = 0.5;
  AVAR f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.3121373, f.val());
  
  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.7779751,grad_f[0]);

  a = 3;
  b = 2;
  c = 0.2;
  f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.0272, f.val());
  x = createAVEC(b);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.02507405,grad_f[0]);
}
TEST(DiffRev,ibeta_ddv) {
  using stan::diff::var;
  using stan::math::ibeta;
  using stan::diff::ibeta;
  
  using boost::math::ibeta_derivative;

  double a = 0.6;
  double b = 0.3;
  AVAR c = 0.5;
  AVAR f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.3121373, f.val());
  
  AVEC x = createAVEC(c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(ibeta_derivative(a, b, c.val()),grad_f[0]);

  a = 3;
  b = 2;
  c = 0.2;
  f = ibeta(a,b,c);
  EXPECT_FLOAT_EQ(0.0272, f.val());
  x = createAVEC(c);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(ibeta_derivative(a, b, c.val()),grad_f[0]);
}
