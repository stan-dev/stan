#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, operator_and_function_signatures) {
  test_parsable("function-signatures/math/operators/and");
}

TEST(lang_parser, operator_and_or_interaction_function_signatures) {
  test_parsable("function-signatures/math/operators/and_or_interaction");
}

TEST(lang_parser, operator_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/equal");
}

TEST(lang_parser, operator_greater_than_or_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/greater_than_or_equal");
}

TEST(lang_parser, operator_greater_than_function_signatures) {
  test_parsable("function-signatures/math/operators/greater_than");
}

TEST(lang_parser, operator_less_than_or_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/less_than_or_equal");
}

TEST(lang_parser, operator_less_than_function_signatures) {
  test_parsable("function-signatures/math/operators/less_than");
}

TEST(lang_parser, operator_not_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/not_equal");
}

TEST(lang_parser, operator_not_function_signatures) {
  test_parsable("function-signatures/math/operators/not");
}

TEST(lang_parser, operator_or_function_signatures) {
  test_parsable("function-signatures/math/operators/or");
}

TEST(lang_parser, if_else_function_signatures) {
  test_parsable("function-signatures/math/if_else");
}

TEST(lang_parser, while_function_signatures) {
  test_parsable("function-signatures/math/while");
}
