#include <gtest/gtest.h>
#include <exception>
#include <test/unit/gm/utility.hpp>

TEST(gmParserStatement2Grammar, addConditionalCondition) {
  test_parsable("conditional_condition_good");
  test_throws("conditional_condition_bad_1",
              "conditions in if-else");
  test_throws("conditional_condition_bad_2",
              "conditions in if-else");
}

TEST(gmParserStatementGrammar, validateIntExpr2) {
  test_parsable("validate_int_expr2_good");
  test_throws("validate_int_expr2_bad1",
              "expression denoting integer required");
  test_throws("validate_int_expr2_bad2",
              "expression denoting integer required");
  test_throws("validate_int_expr2_bad3",
              "expression denoting integer required");
  test_throws("validate_int_expr2_bad4",
              "expression denoting integer required");
}

TEST(gmParserStatementGrammar, validateAllowSample) {
  test_throws("validate_allow_sample_bad1",
              "sampling only allowed in model");
  test_throws("validate_allow_sample_bad2",
              "sampling only allowed in model");
  test_throws("validate_allow_sample_bad3",
              "sampling only allowed in model");
}

TEST(gmParserStatementGrammarDef, jacobianAdjustmentWarning) {
  test_parsable("validate_jacobian_warning_good");
  test_warning("validate_jacobian_warning1",
               "you must call increment_log_prob() with the log absolute determinant");
  test_warning("validate_jacobian_warning2",
               "you must call increment_log_prob() with the log absolute determinant");
  test_warning("validate_jacobian_warning3",
               "you must call increment_log_prob() with the log absolute determinant");
  test_warning("validate_jacobian_warning4",
               "you must call increment_log_prob() with the log absolute determinant");
  test_warning("validate_jacobian_warning5",
               "you must call increment_log_prob() with the log absolute determinant");
  test_warning("validate_jacobian_warning6",
               "you must call increment_log_prob() with the log absolute determinant");
}

TEST(gmParserStatementGrammarDef, comparisonsInBoundsTest) {
  test_parsable("validate_bounds_comparison");
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad_bounds1.stan"),
               std::invalid_argument);
}

TEST(gmParserStatementGrammar, validateAssignmentTypes) {
  test_throws("bad_var_assignment_type1",
              "mismatch in assignment");
  test_throws("bad_var_assignment_type2",
              "mismatch in assignment");
  test_throws("bad_var_assignment_vec_arr",
              "mismatch in assignment");
}
