#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/fun/log_inv_logit.hpp>
#include <stan/math/prim/scal/fun/inv_logit.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/log1p.hpp>

void test_log_inv_logit(const double x) {
  using stan::math::var;
  using stan::math::log_inv_logit;
  using std::log;
  using stan::math::inv_logit;

  
  // test gradient
  AVEC x1 = createAVEC(x);
  AVAR f1 = log_inv_logit(x1[0]);
  std::vector<double> grad_f1;
  f1.grad(x1,grad_f1);

  AVEC x2 = createAVEC(x);
  AVAR f2 = log(inv_logit(x2[0]));
  std::vector<double> grad_f2;
  f2.grad(x2,grad_f2);

  EXPECT_EQ(1U, grad_f1.size());
  EXPECT_EQ(1U, grad_f2.size());
  EXPECT_FLOAT_EQ(grad_f2[0], grad_f1[0]);

  // test value
  EXPECT_FLOAT_EQ(log(inv_logit(x)),
                  log_inv_logit(var(x)).val());

}
TEST(AgradRev, log_inv_logit) {
  test_log_inv_logit(-7.2);
  test_log_inv_logit(0.0);
  test_log_inv_logit(1.9);
}

struct log_inv_logit_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return stan::math::log_inv_logit(arg1);
  }
};

TEST(AgradRev,log_inv_logit_NaN) {
  log_inv_logit_fun log_inv_logit_;
  test_nan(log_inv_logit_,false,true);
}
