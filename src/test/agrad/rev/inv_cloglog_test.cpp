#include <stan/agrad/rev/inv_cloglog.hpp>
<<<<<<< HEAD
#include <stan/agrad/rev/exp.hpp>
#include <stan/agrad/rev/operator_unary_negative.hpp>
#include <stan/agrad/rev/operator_subtraction.hpp>

#include <stan/math/functions/inv_cloglog.hpp>

=======
>>>>>>> 364e38d0e7724275230da060d6afc7f93e8c3d53
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,inv_cloglog) {
  using std::exp;
  using stan::agrad::exp;
  AVAR a = 2.7;
  AVAR f = inv_cloglog(a);
  EXPECT_FLOAT_EQ(std::exp(-std::exp(2.7)),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
<<<<<<< HEAD

  AVAR a2 = 2.7;
  AVEC x2 = createAVEC(a2);
  AVAR f2 = 1 - exp(-exp(a2));
  VEC grad_f2;
  f2.grad(x2,grad_f2);

  EXPECT_EQ(1,grad_f.size());
  EXPECT_FLOAT_EQ(grad_f2[0],grad_f[0]);
=======
  EXPECT_FLOAT_EQ(-std::exp(2.7 - std::exp(2.7)),grad_f[0]);
>>>>>>> 364e38d0e7724275230da060d6afc7f93e8c3d53
}
