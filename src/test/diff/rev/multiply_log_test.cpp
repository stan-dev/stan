#include <stan/diff/rev/multiply_log.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>
#include <stan/diff.hpp>

TEST(DiffRev,multiplyLogChainVV) {
  AVAR a = 19.7;
  AVAR b = 1299.1;
  AVAR f = 2.0 * multiply_log(a,b);

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);

  EXPECT_FLOAT_EQ(2.0 * std::log(b.val()), grad_f[0]);
  EXPECT_FLOAT_EQ(2.0 * a.val() / b.val(), grad_f[1]);
}
TEST(DiffRev,multiplyLogChainDV) {
  double a = 19.7;
  AVAR b = 1299.1;
  AVAR f = 2.0 * multiply_log(a,b);

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);

  EXPECT_FLOAT_EQ(2.0 * a / b.val(), grad_f[0]);
}
TEST(DiffRev,multiplyLogChainVD) {
  AVAR a = 19.7;
  double b = 1299.1;
  AVAR f = 2.0 * multiply_log(a,b);

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);

  EXPECT_FLOAT_EQ(2.0 * std::log(b), grad_f[0]);
}
TEST(DiffRev,multiply_log_var_var) {
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

TEST(DiffRev,multiply_log_var_double){
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
TEST(DiffRev,multiply_log_double_var){
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
