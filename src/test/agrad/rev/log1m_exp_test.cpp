#include <stan/agrad/rev/log1m_exp.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev/exp.hpp>
#include <stan/agrad/agrad.hpp>

void test_log1m_exp(double val) {
  using stan::math::log1m_exp;
  using stan::agrad::log1m_exp;
  using stan::agrad::exp;
  using std::exp;

  AVAR a(val);   
  AVEC x = createAVEC(a);
  AVAR f = log1m_exp(a);
  EXPECT_FLOAT_EQ(log1m_exp(val), f.val());
  VEC g;
  f.grad(x,g);
  double f_val = f.val();
  
  AVAR a2(val);
  AVEC x2 = createAVEC(a2);
  AVAR f2 = log(1.0 - exp(a2));
  VEC g2;
  f2.grad(x2,g2);

  EXPECT_EQ(1U,g.size());
  EXPECT_EQ(1U,g2.size());
  EXPECT_FLOAT_EQ(g2[0],g[0]);
  EXPECT_FLOAT_EQ(g2[0],-1/boost::math::expm1(-val)); // analytic deriv
  EXPECT_FLOAT_EQ(f2.val(),f_val);
}

TEST(AgradRev, log1m_exp) {
  test_log1m_exp(-0.01);
  test_log1m_exp(-0.1);
  test_log1m_exp(-2.0);
  test_log1m_exp(-9.0);
  test_log1m_exp(-15.0);
}

TEST(AgradRev, log1m_exp_exception) {
  using stan::math::log1m_exp;
  using stan::agrad::log1m_exp;
  EXPECT_NO_THROW(log1m_exp(AVAR(-3)));
  EXPECT_NO_THROW(log1m_exp(AVAR(3)));
}
