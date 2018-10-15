#include <gtest/gtest.h>
#include <exception>
#include <test/unit/lang/utility.hpp>

TEST(langParserStatement2Grammar, addConditionalCondition) {
  test_parsable("conditional_condition_good");
  test_throws("conditional_condition_bad_1",
              "Conditions in if-else");
  test_throws("conditional_condition_bad_2",
              "Conditions in if-else");
}

TEST(langParserStatementGrammar, validateIntExpr2) {
  test_parsable("validate_int_expr2_good");
  test_throws("validate_int_expr2_bad1",
              "Loop must be over container or range");
  test_throws("validate_int_expr2_bad2",
              "Loop must be over container or range");
  test_throws("validate_int_expr2_bad3",
              "Dimension declaration requires expression denoting integer");
  test_throws("validate_int_expr2_bad4",
              "Loop must be over container or range");
}

TEST(langParserStatementGrammar, validateAllowSample) {
  test_throws("validate_allow_sample_bad1",
              "Sampling statements (~) and increment_log_prob() are");
  test_throws("validate_allow_sample_bad2",
              "Sampling statements (~) and increment_log_prob() are");
  test_throws("validate_allow_sample_bad3",
              "Sampling statements (~) and increment_log_prob() are");
}

TEST(langParserStatementGrammar, targetIncrement) {
  test_parsable("increment-target");
}

TEST(langParserStatementGrammar, targetReserved) {
  test_throws("target-reserved",
              "Variable identifier (name) may not be reserved word");
  test_throws("target-reserved",
              "found identifier=target");
}

TEST(langParserStatementGrammar, deprecateIncrementLogProb) {
  test_warning("deprecate-increment-log-prob",
               "Info: increment_log_prob(...);"
               " is deprecated and will be removed in the future.");
  test_warning("deprecate-increment-log-prob",
               "  Use target += ...; instead.");
}

TEST(langParserStatementGrammarDef, jacobianAdjustmentWarning) {
  test_parsable("validate_jacobian_warning_good");
  test_warning("validate_jacobian_warning1",
               "you need to include a target += statement with"
               " the log absolute determinant of the Jacobian of the transform.");
  test_warning("validate_jacobian_warning2",
               "you need to include a target += statement with"
               " the log absolute determinant of the Jacobian of the transform.");
  test_warning("validate_jacobian_warning3",
               "you need to include a target += statement with"
               " the log absolute determinant of the Jacobian of the transform.");
  test_warning("validate_jacobian_warning4",
               "you need to include a target += statement with"
               " the log absolute determinant of the Jacobian of the transform.");
  test_warning("validate_jacobian_warning5",
               "you need to include a target += statement with"
               " the log absolute determinant of the Jacobian of the transform.");
  test_warning("validate_jacobian_warning6",
               "you need to include a target += statement with"
               " the log absolute determinant of the Jacobian of the transform.");
}

TEST(langParserStatementGrammarDef, jacobianUserFacing) {
  test_warning("validate_jacobian_warning_user",
               "exp(y[1]) ~ normal(...)");
}

TEST(langParserStatementGrammarDef, comparisonsInBoundsTest) {
  test_parsable("validate_bounds_comparison");
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_bounds1.stan"),
               std::invalid_argument);
}

TEST(langParserStatementGrammar, validateAssignmentTypes) {
  test_throws("bad_var_assignment_type1",
              "mismatch in assignment");
  test_throws("bad_var_assignment_type2",
              "mismatch in assignment");
  test_throws("bad_var_assignment_vec_arr",
              "mismatch in assignment");
}

TEST(langParserStatementGrammar, assignRealToIntMessage) {
  test_throws("assign_real_to_int",
              "Base type mismatch in assignment");
}

TEST(langParserStatementGrammar, useCdfWithSamplingNotation) {
  test_throws("cdf-sample",
              "CDF and CCDF functions may not be used with sampling notation.");
  test_throws("ccdf-sample",
              "CDF and CCDF functions may not be used with sampling notation.");
  test_throws("multiply_sample",
              "Only distribution names can be used with sampling (~) notation");
  test_throws("binomial_coefficient_sample",
              "Only distribution names can be used with sampling (~) notation");
}

TEST(langParserStatementGrammar, targetFunGetLpDeprecated) {
  test_warning("get-lp-deprecate", 
               "Info: get_lp() function deprecated.");
  test_warning("get-lp-deprecate", 
               "  It will be removed in a future release.");
  test_warning("get-lp-deprecate", 
               "  Use target() instead.");
  test_throws("get-lp-target-data",
              "Function target() or functions suffixed with _lp only"
              " allowed in transformed parameter block");
  test_throws("get-lp-target-data",
              "Found function = target or get_lp"
              " in block = transformed data");
  test_parsable("get-lp-target");
}

TEST(langParserStatementGrammar, removeLpDoubleUnderscore) {
  test_throws("lp-error",
              "Error (fatal):  Use of lp__ is no longer supported.");
  test_throws("lp-error",
              "  Use target += ... statement to increment log density.");
}

TEST(langParserStatementGrammar, plusEqualsGood) {
  test_parsable("compound-assign/plus_equals_prim");
  test_parsable("compound-assign/plus_equals_container");
  test_parsable("compound-assign/plus_equals_manual");
}

TEST(langParserStatementGrammar, minusEqualsGood) {
  test_parsable("compound-assign/minus_equals_prim");
  test_parsable("compound-assign/minus_equals_container");
  test_parsable("compound-assign/minus_equals_manual");
}

TEST(langParserStatementGrammar, multiplyEqualsGood) {
  test_parsable("compound-assign/multiply_equals_prim");
  test_parsable("compound-assign/multiply_equals_container");
  test_parsable("compound-assign/multiply_equals_manual");
}

TEST(langParserStatementGrammar, divideEqualsGood) {
  test_parsable("compound-assign/divide_equals_prim");
  test_parsable("compound-assign/divide_equals_container");
  test_parsable("compound-assign/divide_equals_manual");
}

TEST(langParserStatementGrammar, eltOpEqualsGood) {
  test_parsable("compound-assign/elt_multiply_equals");
  test_parsable("compound-assign/elt_divide_equals");
}

TEST(langParserStatementGrammar, plusEqualsBad) {
  test_throws("compound-assign/plus_equals_bad_var_lhs","does not exist");
  test_throws("compound-assign/plus_equals_bad_var_lhs2",
              "Cannot assign to variable outside of declaration block");
  test_throws("compound-assign/plus_equals_bad_lhs_idxs",
              "Left-hand side indexing incompatible with variable");
  test_throws("compound-assign/plus_equals_bad_var_rhs",
              "does not exist");
  test_throws("compound-assign/plus_equals_type_mismatch",
              "Cannot apply operator '+='");
  test_throws("compound-assign/plus_equals_type_mismatch2",
              "Cannot apply operator '+='");
  test_throws("compound-assign/plus_equals_matrix_array",
              "Cannot apply operator '+='");
  test_throws("compound-assign/plus_equals_matrix_array2",
              "Cannot apply operator '+='");
  test_throws("compound-assign/plus_equals_prim_array",
              "Cannot apply operator '+='");
  test_throws("compound-assign/plus_equals_row_vec_array",
              "Cannot apply operator '+='");
  test_throws("compound-assign/plus_equals_row_vec_array",
              "Cannot apply operator '+='");
  test_throws("compound-assign/plus_equals_bad_init",
              "PARSER EXPECTED: \";\"");
}

TEST(langParserStatementGrammar, timesEqualsBad) {
  test_throws("compound-assign/times_equals_matrix_array",
              "Cannot apply operator '*='");
}


TEST(langParserStatementGrammar, eltOpEqualsBad) {
  test_throws("compound-assign/elt_times_equals_prim",
              "Cannot apply element-wise operation to scalar");
  test_throws("compound-assign/elt_divide_equals_prim",
              "Cannot apply element-wise operation to scalar");
}

TEST(langParserStatementGrammar, noCloseBrace) {
  test_throws("expect_statement_seq_close_brace",
              "PARSER EXPECTED: \"}\"");
}              

TEST(langParserStatementGrammar, noCloseBrace_2) {
  test_throws("expect_statement_seq_close_brace_2",
              "PARSER EXPECTED: \"}\"");
}              

TEST(langParserStatementGrammar, noCloseBrace_3) {
  test_throws("expect_statement_seq_close_brace_3",
              "Unexpected open block, missing close block \"}\""
              " before keyword");
}

TEST(langParserStatementGrammar, noCloseBrace_4) {
  test_throws("expect_statement_seq_close_brace_4",
              "\'}\' to close variable declarations");
}

