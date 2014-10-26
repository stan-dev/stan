#include <stan/agrad/rev/functions/inv_cloglog.hpp>
#include <stan/agrad/rev/functions/exp.hpp>
#include <stan/agrad/rev/operators/operator_unary_negative.hpp>
#include <stan/agrad/rev/operators/operator_subtraction.hpp>
#include <stan/math/functions/inv_cloglog.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/rev/nan_util.hpp>

TEST(AgradRev,inv_cloglog) {
  using std::exp;
  using stan::agrad::exp;
  AVAR a = 2.7;
  AVAR f = inv_cloglog(a);
  EXPECT_FLOAT_EQ(1 - std::exp(-std::exp(2.7)),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);

  AVAR a2 = 2.7;
  AVEC x2 = createAVEC(a2);
  AVAR f2 = 1 - exp(-exp(a2));
  VEC grad_f2;
  f2.grad(x2,grad_f2);

  EXPECT_EQ(1U,grad_f.size());
  EXPECT_FLOAT_EQ(grad_f2[0],grad_f[0]);
}

struct inv_cloglog_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv_cloglog(arg1);
  }
};

TEST(AgradRev,inv_cloglog_NaN) {
  inv_cloglog_fun inv_cloglog_;
  test_nan(inv_cloglog_,false,true);
}
