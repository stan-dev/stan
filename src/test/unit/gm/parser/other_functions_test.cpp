#include <gtest/gtest.h>
#include <test/unit/gm/utility.hpp>

TEST(gm_parser, operator_and_function_signatures) {
  test_parsable("function-signatures/math/operators/and");
}

TEST(gm_parser, operator_and_or_interaction_function_signatures) {
  test_parsable("function-signatures/math/operators/and_or_interaction");
}

TEST(gm_parser, operator_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/equal");
}

TEST(gm_parser, operator_greater_than_or_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/greater_than_or_equal");
}

TEST(gm_parser, operator_greater_than_function_signatures) {
  test_parsable("function-signatures/math/operators/greater_than");
}

TEST(gm_parser, operator_less_than_or_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/less_than_or_equal");
}

TEST(gm_parser, operator_less_than_function_signatures) {
  test_parsable("function-signatures/math/operators/less_than");
}

TEST(gm_parser, operator_not_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/not_equal");
}

TEST(gm_parser, operator_not_function_signatures) {
  test_parsable("function-signatures/math/operators/not");
}

TEST(gm_parser, operator_or_function_signatures) {
  test_parsable("function-signatures/math/operators/or");
}

TEST(gm_parser, if_else_function_signatures) {
  test_parsable("function-signatures/math/if_else");
}

TEST(gm_parser, while_function_signatures) {
  test_parsable("function-signatures/math/while");
}
