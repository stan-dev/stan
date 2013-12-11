#include <stan/agrad/rev/functions/fmod.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

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
