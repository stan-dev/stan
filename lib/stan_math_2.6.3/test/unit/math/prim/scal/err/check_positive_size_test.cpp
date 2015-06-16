#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_positive_size.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(ErrorHandlingScalar, CheckPositiveSize) {
  using stan::math::check_positive_size;
  const char* function = "function";
  const char* name = "name";
  const char* expr = "expr";
  std::string expected_msg;

  
  EXPECT_TRUE(check_positive_size(function, name, expr, 10));

  
  expected_msg = "name must have a positive size, but is 0; "
    "dimension size expression = expr";
  EXPECT_THROW_MSG(check_positive_size(function, name, expr, 0),
                   std::invalid_argument,
                   expected_msg);


  expected_msg = "name must have a positive size, but is -1; "
    "dimension size expression = expr";
  EXPECT_THROW_MSG(check_positive_size(function, name, expr, -1),
                   std::invalid_argument,
                   expected_msg);
  
}
