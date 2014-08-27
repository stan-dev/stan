#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/operators/operator_multiplication.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>

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
  
  EXPECT_EQ(2U,g.size());
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
}

TEST(AgradRev, print) {
  using stan::agrad::var;

  std::ostringstream output;
  std::string str;

  var initialized_var(0);
  output << initialized_var;
  str = output.str();
  EXPECT_STREQ("0:0", output.str().c_str());


  output.clear();
  output.str("");
  var uninitialized_var;
  output << uninitialized_var;
  str = output.str();
  EXPECT_STREQ("uninitialized", output.str().c_str());
}


TEST(AgradAutoDiff, nestedGradient) {
  // this isn't a unit test of any individual function.
  //    chainable.hpp : grad (function)
  //    var_stack.hpp : recover_memory_nested, start_nested (functions)
  //    var.hpp:      : var (class)

  using stan::agrad::var;
  using stan::agrad::start_nested;
  using stan::agrad::recover_memory_nested;
  using stan::agrad::grad;
  start_nested();
  var a = 4;
  var b = 7;
  var c = a * b;
  start_nested();
  var d = 3;
  var e = 15;
  var f = e * d;
  grad(f.vi_);
  EXPECT_FLOAT_EQ(15.0, d.vi_->adj_);
  EXPECT_FLOAT_EQ(3.0, e.vi_->adj_);
  recover_memory_nested();

  grad(c.vi_);
  EXPECT_FLOAT_EQ(7.0, a.vi_->adj_);
  EXPECT_FLOAT_EQ(4.0, b.vi_->adj_);
  recover_memory_nested();
}

