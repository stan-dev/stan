#include <gtest/gtest.h>
#include <stan/agrad/agrad.hpp>
#include <test/agrad/util.hpp>




TEST(AgradRev,floor_var) {
  AVAR a = 1.2;
  AVAR f = floor(a);
  EXPECT_FLOAT_EQ(1.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradRev,ceil_var) {
  AVAR a = 1.9;
  AVAR f = ceil(a);
  EXPECT_FLOAT_EQ(2.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradRev,fmod_var_var) {
  AVAR a = 2.7;
  AVAR b = 1.3;
  AVAR f = fmod(a,b);
  EXPECT_FLOAT_EQ(std::fmod(2.7,1.3),f.val());
  
  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
  EXPECT_FLOAT_EQ(-2.0,g[1]); // (int)(2.7/1.3) = 2
}

TEST(AgradRev,fmod_var_double) {
  AVAR a = 2.7;
  double b = 1.3;
  AVAR f = fmod(a,b);
  EXPECT_FLOAT_EQ(fmod(2.7,1.3),f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,fmod_double_var) {
  double a = 2.7;
  AVAR b = 1.3;
  AVAR f = fmod(a,b);
  EXPECT_FLOAT_EQ(fmod(2.7,1.3),f.val());
  
  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-2.0,g[0]); // (int)(2.7/1.3) = 2
}

TEST(AgradRev,jacobian) {
  AVAR x1 = 2.0;
  AVAR x2 = 3.0;
  
  AVAR y1 = x1 * x2;
  AVAR y2 = x1 + x2;
  AVAR y3 = 17.0 * x1;

  AVEC x = createAVEC(x1,x2);
  AVEC y = createAVEC(y1,y2,y3);

  std::vector<std::vector<double> > J;
  jacobian(y,x,J);

  EXPECT_FLOAT_EQ(3.0,J[0][0]); // dy1/dx1
  EXPECT_FLOAT_EQ(2.0,J[0][1]); // dy1/dx2

  EXPECT_FLOAT_EQ(1.0,J[1][0]); // dy2/dx1
  EXPECT_FLOAT_EQ(1.0,J[1][1]); // dy2/dx2

  EXPECT_FLOAT_EQ(17.0,J[2][0]); // dy2/dx1
  EXPECT_FLOAT_EQ(0.0,J[2][1]); // dy2/dx2
}

TEST(AgradRev,free_memory) {
  AVAR a = 2.0;
  AVAR b = -3.0;
  AVAR f = a * b;
  EXPECT_FLOAT_EQ(-6.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-3.0,grad_f[0]);
  EXPECT_FLOAT_EQ(2.0,grad_f[1]);
  stan::agrad::free_memory();

  AVAR aa = 2.0;
  AVAR bb = -3.0;
  AVAR ff = aa * bb;
  EXPECT_FLOAT_EQ(-6.0,ff.val());

  AVEC xx = createAVEC(aa,bb);
  VEC grad_ff;
  ff.grad(xx,grad_ff);
  EXPECT_FLOAT_EQ(-3.0,grad_ff[0]);
  EXPECT_FLOAT_EQ(2.0,grad_ff[1]);
}

TEST(AgradRev, smart_ptrs) {
  AVAR a = 2.0;
  EXPECT_FLOAT_EQ(2.0, (*a).val_);
  EXPECT_FLOAT_EQ(2.0, a->val_);

  EXPECT_FLOAT_EQ(2.0,(*a.vi_).val_);
  EXPECT_FLOAT_EQ(2.0,a.vi_->val_);
}

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
