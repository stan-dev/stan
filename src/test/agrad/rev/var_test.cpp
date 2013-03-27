#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/operator_multiplication.hpp>
#include <gtest/gtest.h>
#include <test/agrad/util.hpp>

TEST(AgradRev,a_eq_x) {
  AVAR a = 5.0;
  EXPECT_FLOAT_EQ(5.0,a.val());
}

TEST(AgradRev,a_of_x) {
  AVAR a(6.0);
  EXPECT_FLOAT_EQ(6.0,a.val());
}

TEST(AgradRev,a__a_eq_x) {
  AVAR a;
  a = 7.0;
  EXPECT_FLOAT_EQ(7.0,a.val());
}

TEST(AgradRev,eq_a) {
  AVAR a = 5.0;
  AVAR f = a;
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}

TEST(AgradRev,a_ostream) {
  AVAR a = 6.0;
  std::ostringstream os;
  
  os << a;
  EXPECT_EQ ("6:0", os.str());

  os.str("");
  a = 10.5;
  os << a;
  EXPECT_EQ ("10.5:0", os.str());
}

TEST(AgradRev, smart_ptrs) {
  AVAR a = 2.0;
  EXPECT_FLOAT_EQ(2.0, (*a).val_);
  EXPECT_FLOAT_EQ(2.0, a->val_);

  EXPECT_FLOAT_EQ(2.0,(*a.vi_).val_);
  EXPECT_FLOAT_EQ(2.0,a.vi_->val_);
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
