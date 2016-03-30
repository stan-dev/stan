#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, assignment_double_index_lhs_funciton_signatures) {
  test_parsable("assignment_double_index_lhs");
}

TEST(lang_parser, assignments_double_var_funciton_signatures) {
  test_parsable("assignments_double_var");
}

TEST(lang_parser, assignments_var_funciton_signatures) {
  test_parsable("assignments_var");
}

TEST(lang_parser, assignments_funciton_signatures) {
  test_parsable("assignments");
}

TEST(lang_parser, mat_assign_funciton_signatures) {
  test_parsable("mat_assign");
}

// NEW ASSIGNMENT AFTER HERE

TEST(lang_parser, new_assign) {
  test_parsable("assignment-new");
  test_warning("assignment-old",
               "Warning (non-fatal): assignment operator <- deprecated"
               " in the Stan language;"
               " use = instead.");

}
