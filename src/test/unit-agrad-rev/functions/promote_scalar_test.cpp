#include <stan/math/matrix/promote_scalar.hpp>
#include <test/unit/math/functions/promote_type_test_util.hpp>
#include <stan/agrad/rev/var.hpp>
#include <gtest/gtest.h>

// there is no agrad-defined version of promote_scalar, so this is
// just testing that it works with non-inter-convertible types (double
// can be assigned to var, but not vice-versa)

TEST(AgradRevFunctionsPromoteScalar, Mismatch) {
  using stan::agrad::var;
  using stan::math::promote_scalar;
  EXPECT_FLOAT_EQ(2.3, promote_scalar<var>(2.3).val());
}
