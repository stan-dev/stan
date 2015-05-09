#include <stan/math/rev/scal/fun/log1p_exp.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/fun/log1m_inv_logit.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/prim/scal/fun/inv_logit.hpp>
#include <stan/math/rev/scal/fun/inv_logit.hpp>

void test_log1p_exp(double val) {
  using stan::math::log1p_exp;
  using stan::math::log1p_exp;
  using stan::math::exp;
  using std::exp;

  AVAR a(val);   
  AVEC x = createAVEC(a);
  AVAR f = log1p_exp(a);
  EXPECT_FLOAT_EQ(log1p_exp(val), f.val());
  VEC g;
  f.grad(x,g);
  double f_val = f.val();
  
  AVAR a2(val);
  AVEC x2 = createAVEC(a2);
  AVAR f2 = log(1.0 + exp(a2));
  VEC g2;
  f2.grad(x2,g2);

  EXPECT_EQ(1U,g.size());
  EXPECT_EQ(1U,g2.size());
  EXPECT_FLOAT_EQ(g2[0],g[0]);
  EXPECT_FLOAT_EQ(g2[0],1.0/(1.0 + exp(-val))); // analytic deriv
  EXPECT_FLOAT_EQ(f2.val(),f_val);
}

TEST(AgradRev, log1p_exp) {
  test_log1p_exp(-15.0);
  test_log1p_exp(-5.0);
  test_log1p_exp(-1.0);
  test_log1p_exp(0.0);
  test_log1p_exp(2.0);
  test_log1p_exp(32.0);
  test_log1p_exp(64.0);
}

struct log1p_exp_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log1p_exp(arg1);
  }
};

TEST(AgradRev,log1p_exp_NaN) {
  log1p_exp_fun log1p_exp_;
  test_nan(log1p_exp_,false,true);
}
