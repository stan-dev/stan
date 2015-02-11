#include <gtest/gtest.h>
#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/rev/arr/meta/var.hpp>
#include <sstream>

TEST(AgradFwdFvar, insertion_operator) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(5.0);
  std::stringstream ss;
  ss << a;
  EXPECT_EQ("5", ss.str());
}
