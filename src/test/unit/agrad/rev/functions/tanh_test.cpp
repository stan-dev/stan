#include <stan/agrad/rev/functions/tanh.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/rev/nan_util.hpp>

TEST(AgradRev,tanh_var) {
  AVAR a = 0.68;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(0.59151939543, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(0.68) * cosh(0.68)), g[0]);
}

TEST(AgradRev,tanh_neg_var) {
  AVAR a = -.68;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(-0.59151939543,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(-0.68) * cosh(-0.68)), g[0]);
}

TEST(AgradRev,tanh_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(1.0,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(inf) * cosh(inf)), g[0]);
}

TEST(AgradRev,tanh_neg_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = -inf;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(-1,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(-inf) * cosh(-inf)), g[0]);
}

struct tanh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return tanh(arg1);
  }
};

TEST(AgradRev,tanh_NaN) {
  tanh_fun tanh_;
  test_nan(tanh_,false,true);
}
