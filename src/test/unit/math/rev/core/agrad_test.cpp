#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>

TEST(AgradRev, multiple_grads) {
  for (int i = 0; i < 100; ++i) {
    AVAR a = 2.0;
    AVAR b = 3.0 * a;
    AVAR c = sin(a) * b;
    c = c; // fixes warning regarding unused variable
    
    AVAR nothing;
  }
  
  AVAR d = 2.0;
  AVAR e = 3.0;
  AVAR f = d * e;
  
  AVEC x = createAVEC(d,e);
  VEC grad_f;
  f.grad(x,grad_f);

  EXPECT_FLOAT_EQ(3.0, d.adj());
  EXPECT_FLOAT_EQ(2.0, e.adj());

  EXPECT_FLOAT_EQ(3.0, grad_f[0]);
  EXPECT_FLOAT_EQ(2.0, grad_f[1]);
}

