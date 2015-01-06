#include <stan/error_handling/scalar/check_positive_index.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(ErrorHandlingScalar, CheckPositiveIndex) {
  using stan::error_handling::check_positive_index;
  std::string function = "function";
  std::string name = "name";
  std::string expr = "expr";
  std::string expected_msg;

  
  EXPECT_TRUE(check_positive_index(function, name, expr, 10));

  
  expected_msg = "name must have a positive size, but is 0; "
    "dimension size expression = expr";
  EXPECT_THROW_MSG(check_positive_index(function, name, expr, 0),
                   std::invalid_argument,
                   expected_msg);


  expected_msg = "name must have a positive size, but is -1; "
    "dimension size expression = expr";
  EXPECT_THROW_MSG(check_positive_index(function, name, expr, -1),
                   std::invalid_argument,
                   expected_msg);
  
}
