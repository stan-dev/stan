#include <gtest/gtest.h>
#include <stan/agrad/agrad.hpp>
#include <test/agrad/util.hpp>



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

TEST(AgradRev, stackAllocation) {
  using stan::agrad::vari;
  using stan::agrad::var;

  vari ai(1.0);
  vari bi(2.0);

  var a(&ai);
  var b(&bi);

  AVEC x = createAVEC(a,b);
  var f = a * b;

  VEC g;
  f.grad(x,g);
  
  EXPECT_EQ(2,g.size());
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
}
