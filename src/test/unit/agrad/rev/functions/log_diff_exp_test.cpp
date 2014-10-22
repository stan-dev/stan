#include <stan/agrad/rev/functions/log_diff_exp.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/rev/nan_util.hpp>
#include <stan/meta/traits.hpp>

TEST(AgradRev,log_diff_exp_vv) {
  AVAR a = 5.0;
  AVAR b = 2.0;
  AVAR f = log_diff_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(5) - std::exp(2)), f.val());
  
  AVEC x = createAVEC(a, b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(5.0) / (std::exp(5.0) - std::exp(2.0)), grad_f[0]);
  EXPECT_FLOAT_EQ(std::exp(2.0) / (std::exp(2.0) - std::exp(5.0)), grad_f[1]);

  // underflow example
  a = 1000;
  b = 10;
  f = log_diff_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(0.0) - std::exp(-990.0)) + 1000.0, f.val());
  
  x = createAVEC(a, b);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (std::exp (1000.0 - (std::log(std::exp(0.0) - std::exp(-999.0)) + 1000)), grad_f[0]);
  EXPECT_FLOAT_EQ (std::exp (10.0 - (std::log(std::exp(0.0) - std::exp(-999.0)) + 1000)), grad_f[1]);
}

TEST(AgradRev,log_diff_exp_vd) {
  AVAR a = 5.0;
  double b = 2.0;
  AVAR f = log_diff_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(5) - std::exp(2)), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(5.0) / (std::exp(5.0) - std::exp(2.0)), grad_f[0]);

  // underflow example
  a = 1000;
  b = 10;
  f = log_diff_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(0.0) - std::exp(-990.0)) + 1000.0, f.val());
  
  x = createAVEC(a);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (std::exp (1000.0 - (std::log(std::exp(0.0) - std::exp(-999.0)) + 1000)), grad_f[0]);
}

TEST(AgradRev,log_diff_exp_dv) {
  double a = 5.0;
  AVAR b = 2.0;
  AVAR f = log_diff_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(5) - std::exp(2)), f.val());
  
  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(2.0) / (std::exp(2.0) - std::exp(5.0)), grad_f[0]);

  // underflow example
  a = 1000;
  b = 10;
  f = log_diff_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(0.0) - std::exp(-990.0)) + 1000.0, f.val());
  
  x = createAVEC(b);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (0, grad_f[0]); //1/(1-exp(1000-10)) explodes to 1/-inf = 0
}

void test_log_diff_exp_2_vv(double a_val, 
                           double b_val) {
  using std::exp;
  using std::log;
  using stan::math::log_diff_exp;

  AVAR a(a_val);
  AVAR b(b_val);

  AVEC x = createAVEC(a,b);
  AVAR f = log_diff_exp(a,b);
  VEC g;
  f.grad(x,g);
  
  double f_val = f.val();

  stan::agrad::var a2(a_val);
  stan::agrad::var b2(b_val);
  AVEC x2 = createAVEC(a2,b2);
  AVAR f2 = log(exp(a2) - exp(b2));
  VEC g2;
  f2.grad(x2,g2);

  EXPECT_FLOAT_EQ(f2.val(), f_val);
  EXPECT_EQ(2U,g.size());
  EXPECT_EQ(2U,g2.size());
  EXPECT_FLOAT_EQ(g2[0],g[0]);
  EXPECT_FLOAT_EQ(g2[1],g[1]);
}
void test_log_diff_exp_2_vd(double a_val,
                           double b) {
  using std::exp;
  using std::log;
  using stan::math::log_diff_exp;

  AVAR a(a_val);
  AVEC x = createAVEC(a);
  AVAR f = log_diff_exp(a,b);
  VEC g;
  f.grad(x,g);
  
  double f_val = f.val();

  stan::agrad::var a2(a_val);
  AVEC x2 = createAVEC(a2);
  AVAR f2 = log(exp(a2) - exp(b));
  VEC g2;
  f2.grad(x2,g2);

  EXPECT_FLOAT_EQ(f2.val(), f_val);
  EXPECT_EQ(1U,g.size());
  EXPECT_EQ(1U,g2.size());
  EXPECT_FLOAT_EQ(g2[0],g[0]);
  
}
void test_log_diff_exp_2_dv(double a,
                           double b_val) {
  using std::exp;
  using std::log;
  using stan::math::log_diff_exp;

  AVAR b(b_val);
  AVEC x = createAVEC(b);
  AVAR f = log_diff_exp(a,b);
  VEC g;
  f.grad(x,g);
  
  double f_val = f.val();

  AVAR b2(b_val);
  AVEC x2 = createAVEC(b2);
  AVAR f2 = log(exp(a) - exp(b2));
  VEC g2;
  f2.grad(x2,g2);

  EXPECT_FLOAT_EQ(f2.val(), f_val);
  EXPECT_EQ(1U,g.size());
  EXPECT_EQ(1U,g2.size());
  EXPECT_FLOAT_EQ(g2[0],g[0]);
  
}

void test_log_diff_exp_2(double a, double b) {
  test_log_diff_exp_2_vv(a,b);
  test_log_diff_exp_2_vd(a,b);
  test_log_diff_exp_2_dv(a,b);
}

TEST(AgradRev,log_diff_exp_2) {
  test_log_diff_exp_2(0.1,0.0);
  test_log_diff_exp_2(3.01,2.0);
  test_log_diff_exp_2(2.0,1.0);
  test_log_diff_exp_2(-2.0,-15.0);
  test_log_diff_exp_2(2.0,-15.0);
}

TEST(AgradRev,log_diff_exp_exception) {
  EXPECT_NO_THROW(log_diff_exp(AVAR(3), AVAR(4)));
  EXPECT_NO_THROW(log_diff_exp(AVAR(3), 4));
  EXPECT_NO_THROW(log_diff_exp(3, AVAR(4)));
}

struct log_diff_exp_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return log_diff_exp(arg1,arg2);
  }
};

TEST(AgradRev, log_diff_exp_nan) {
  log_diff_exp_fun log_diff_exp_;
  test_nan(log_diff_exp_,3.0,5.0,false,true);
}
