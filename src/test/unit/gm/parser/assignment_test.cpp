#include <gtest/gtest.h>
#include <test/unit/gm/utility.hpp>

TEST(gm_parser, assignment_double_index_lhs_function_signatures) {
  test_parsable("assignment_double_index_lhs");
}

TEST(gm_parser, assignments_double_var_function_signatures) {
  test_parsable("assignments_double_var");
}

TEST(gm_parser, assignments_var_function_signatures) {
  test_parsable("assignments_var");
}

TEST(gm_parser, assignments_function_signatures) {
  test_parsable("assignments");
}

TEST(gm_parser, mat_assign_function_signatures) {
  test_parsable("mat_assign");
}
