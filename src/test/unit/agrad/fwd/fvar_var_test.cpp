#include <gtest/gtest.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/var.hpp>
#include <sstream>

TEST(AgradFwdFvar, insertion_operator) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(5.0);
  std::stringstream ss;
  ss << a;
  EXPECT_EQ("5", ss.str());
}
