#include <stan/math/rev/scal/fun/log_sum_exp.hpp>
#include <stan/math/rev/arr/fun/log_sum_exp.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradRev,log_sum_exp_vv) {
  AVAR a = 5.0;
  AVAR b = 2.0;
  AVAR f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(5) + std::exp(2)), f.val());
  
  AVEC x = createAVEC(a, b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(5.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[0]);
  EXPECT_FLOAT_EQ(std::exp(2.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[1]);

  // underflow example
  a = 1000;
  b = 10;
  f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(0.0) + std::exp(-990.0)) + 1000.0, f.val());
  
  x = createAVEC(a, b);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (std::exp (1000.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)), grad_f[0]);
  EXPECT_FLOAT_EQ (std::exp (10.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)), grad_f[1]);
}
TEST(AgradRev,log_sum_exp_vd) {
  AVAR a = 5.0;
  double b = 2.0;
  AVAR f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(5) + std::exp(2)), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(5.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[0]);

  // underflow example
  a = 1000;
  b = 10;
  f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(0.0) + std::exp(-990.0)) + 1000.0, f.val());
  
  x = createAVEC(a);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (std::exp (1000.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)), grad_f[0]);
}
TEST(AgradRev,log_sum_exp_dv) {
  double a = 5.0;
  AVAR b = 2.0;
  AVAR f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(5) + std::exp(2)), f.val());
  
  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(2.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[0]);

  // underflow example
  a = 10;
  b = 1000;
  f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(0.0) + std::exp(-990.0)) + 1000.0, f.val());
  
  x = createAVEC(b);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (std::exp (1000.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)), grad_f[0]);
}
TEST(AgradRev,log_sum_exp_vector) {
  // simple test
  AVEC x;
  x.push_back (5.0);
  x.push_back (2.0);
  
  AVAR f = log_sum_exp(x);
  EXPECT_FLOAT_EQ (std::log(std::exp(5) + std::exp(2)), f.val());
  
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(5.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[0]);
  EXPECT_FLOAT_EQ(std::exp(2.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[1]);

  // longer test
  x.clear();
  x.push_back (1.0);
  x.push_back (2.0);
  x.push_back (3.0);
  x.push_back (4.0);
  x.push_back (5.0);
  f = log_sum_exp(x);
  double expected_log_sum_exp = std::log(std::exp(1) + std::exp(2) + std::exp(3) + std::exp(4) + std::exp(5));
  EXPECT_FLOAT_EQ (expected_log_sum_exp,
                   f.val());
  
  grad_f.clear();
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(1.0) / exp(expected_log_sum_exp), grad_f[0]);
  EXPECT_FLOAT_EQ(std::exp(2.0) / exp(expected_log_sum_exp), grad_f[1]);
  EXPECT_FLOAT_EQ(std::exp(3.0) / exp(expected_log_sum_exp), grad_f[2]);
  EXPECT_FLOAT_EQ(std::exp(4.0) / exp(expected_log_sum_exp), grad_f[3]);
  EXPECT_FLOAT_EQ(std::exp(5.0) / exp(expected_log_sum_exp), grad_f[4]);

  // underflow example
  x.clear();
  x.push_back(1000.0);
  x.push_back(10.0);
  f = log_sum_exp(x);
  EXPECT_FLOAT_EQ (std::log(std::exp(0.0) + std::exp(-990.0)) + 1000.0, f.val());
  
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (std::exp (1000.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)), grad_f[0]);
  EXPECT_FLOAT_EQ (std::exp (10.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)), grad_f[1]);

  // longer underflow example
  x.clear();
  x.push_back(800.0);
  x.push_back(900.0);
  x.push_back(10.0);
  x.push_back(0.0);
  x.push_back(-100.0);
  f = log_sum_exp(x);
  expected_log_sum_exp = std::log(std::exp(0.0) + std::exp(-100) + std::exp(-890.0) + std::exp(-900.0) + std::exp(-1000.0)) + 900.0;
  EXPECT_FLOAT_EQ (expected_log_sum_exp, 
                   f.val());
  
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (std::exp ( 800.0 - expected_log_sum_exp), grad_f[0]);
  EXPECT_FLOAT_EQ (std::exp ( 900.0 - expected_log_sum_exp), grad_f[1]);
  EXPECT_FLOAT_EQ (std::exp (  10.0 - expected_log_sum_exp), grad_f[2]);
  EXPECT_FLOAT_EQ (std::exp (   0.0 - expected_log_sum_exp), grad_f[3]);
  EXPECT_FLOAT_EQ (std::exp (-100.0 - expected_log_sum_exp), grad_f[4]);
}

void test_log_sum_exp_2_vv(double a_val, 
                           double b_val) {
  using std::exp;
  using std::log;
  using stan::math::log_sum_exp;

  AVAR a(a_val);
  AVAR b(b_val);

  AVEC x = createAVEC(a,b);
  AVAR f = log_sum_exp(a,b);
  VEC g;
  f.grad(x,g);
  
  double f_val = f.val();

  stan::math::var a2(a_val);
  stan::math::var b2(b_val);
  AVEC x2 = createAVEC(a2,b2);
  AVAR f2 = log(exp(a2) + exp(b2));
  VEC g2;
  f2.grad(x2,g2);

  EXPECT_FLOAT_EQ(f2.val(), f_val);
  EXPECT_EQ(2U,g.size());
  EXPECT_EQ(2U,g2.size());
  EXPECT_FLOAT_EQ(g2[0],g[0]);
  EXPECT_FLOAT_EQ(g2[1],g[1]);
}
void test_log_sum_exp_2_vd(double a_val,
                           double b) {
  using std::exp;
  using std::log;
  using stan::math::log_sum_exp;

  AVAR a(a_val);
  AVEC x = createAVEC(a);
  AVAR f = log_sum_exp(a,b);
  VEC g;
  f.grad(x,g);
  
  double f_val = f.val();

  stan::math::var a2(a_val);
  AVEC x2 = createAVEC(a2);
  AVAR f2 = log(exp(a2) + exp(b));
  VEC g2;
  f2.grad(x2,g2);

  EXPECT_FLOAT_EQ(f2.val(), f_val);
  EXPECT_EQ(1U,g.size());
  EXPECT_EQ(1U,g2.size());
  EXPECT_FLOAT_EQ(g2[0],g[0]);
  
}
void test_log_sum_exp_2_dv(double a,
                           double b_val) {
  using std::exp;
  using std::log;
  using stan::math::log_sum_exp;

  AVAR b(b_val);
  AVEC x = createAVEC(b);
  AVAR f = log_sum_exp(a,b);
  VEC g;
  f.grad(x,g);
  
  double f_val = f.val();

  AVAR b2(b_val);
  AVEC x2 = createAVEC(b2);
  AVAR f2 = log(exp(a) + exp(b2));
  VEC g2;
  f2.grad(x2,g2);

  EXPECT_FLOAT_EQ(f2.val(), f_val);
  EXPECT_EQ(1U,g.size());
  EXPECT_EQ(1U,g2.size());
  EXPECT_FLOAT_EQ(g2[0],g[0]);
  
}

void test_log_sum_exp_2(double a, double b) {
  test_log_sum_exp_2_vv(a,b);
  test_log_sum_exp_2_vd(a,b);
  test_log_sum_exp_2_dv(a,b);
}

TEST(AgradRev,log_sum_exp_2) {
  test_log_sum_exp_2(0.0,0.0);
  test_log_sum_exp_2(1.0,2.0);
  test_log_sum_exp_2(2.0,1.0);
  test_log_sum_exp_2(-2.0,15.0);
  test_log_sum_exp_2(2.0,-15.0);
}



TEST(AgradRev, log_sum_exp_vec_1) {
  using stan::math::log_sum_exp;
  using stan::math::log_sum_exp;
  AVAR a(5.0);
  AVEC as = createAVEC(a);
  AVEC x = createAVEC(a);
  AVAR f = log_sum_exp(as);
  EXPECT_FLOAT_EQ(5.0, f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_EQ(1U,g.size());
  EXPECT_FLOAT_EQ(1.0,g[0]);
}


TEST(AgradRev, log_sum_exp_vec_2) {
  using stan::math::log_sum_exp;
  using stan::math::log_sum_exp;
  AVAR a(5.0);
  AVAR b(-7.0);
  AVEC as = createAVEC(a,b);
  AVEC x = createAVEC(a,b);
  AVAR f = log_sum_exp(as);
  EXPECT_FLOAT_EQ(log(exp(5.0) + exp(-7.0)), f.val());
  VEC g;
  f.grad(x,g);

  double f_val = f.val();

  AVAR a2(5.0);
  AVAR b2(-7.0);
  AVEC as2 = createAVEC(a2,b2);
  AVEC x2 = createAVEC(a2,b2);
  AVAR f2 = stan::math::log(stan::math::exp(a2) + stan::math::exp(b2));
  VEC g2;
  f2.grad(x2,g2);

  EXPECT_FLOAT_EQ(f2.val(), f_val);
  EXPECT_EQ(2U,g.size());
  EXPECT_EQ(2U,g2.size());
  EXPECT_FLOAT_EQ(g[0],g2[0]);
  EXPECT_FLOAT_EQ(g[1],g2[1]);
}

TEST(AgradRev, log_sum_exp_vec_3) {
  using stan::math::log_sum_exp;
  using stan::math::log_sum_exp;
  AVAR a(5.0);
  AVAR b(-7.0);
  AVAR c(2.3);
  AVEC as = createAVEC(a,b,c);
  AVEC x = createAVEC(a,b,c);
  AVAR f = log_sum_exp(as);
  EXPECT_FLOAT_EQ(log(exp(5.0) + exp(-7.0) + exp(2.3)), f.val());
  VEC g;
  f.grad(x,g);

  double f_val = f.val();

  AVAR a2(5.0);
  AVAR b2(-7.0);
  AVAR c2(2.3);
  AVEC as2 = createAVEC(a2,b2,c2);
  AVEC x2 = createAVEC(a2,b2,c2);
  AVAR f2 = log(exp(a2) + exp(b2) + exp(c2));
  VEC g2;
  f2.grad(x2,g2);

  EXPECT_FLOAT_EQ(f2.val(), f_val);
  EXPECT_EQ(3U,g.size());
  EXPECT_EQ(3U,g2.size());
  EXPECT_FLOAT_EQ(g[0],g2[0]);
  EXPECT_FLOAT_EQ(g[1],g2[1]);
  EXPECT_FLOAT_EQ(g[2],g2[2]);
}

struct log_sum_exp_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return log_sum_exp(arg1,arg2);
  }
};

TEST(AgradRev, log_sum_exp_nan) {
  log_sum_exp_fun log_sum_exp_;
  test_nan(log_sum_exp_,3.0,5.0,false,true);
}
