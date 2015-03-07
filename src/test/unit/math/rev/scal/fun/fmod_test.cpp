#include <stan/math/rev/scal/fun/fmod.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

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

struct fmod_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return fmod(arg1,arg2);
  }
};

TEST(AgradRev, fmod_nan) {
  fmod_fun fmod_;
  test_nan(fmod_,3.0,5.0,false, true);
}
