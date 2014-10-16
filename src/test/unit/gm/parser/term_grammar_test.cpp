#include <gtest/gtest.h>
#include <test/unit/gm/utility.hpp>

TEST(gmParserTermGrammar, infixExponentiation) {
  test_parsable("validate_exponentiation_good");
  test_parsable("validate_exponentiation_precedence");
  test_throws("validate_exponentiation_bad", 
              "base type mismatch in assignment; variable name = z");
}

TEST(gmParserTermGrammar, modulusOp) {
  test_parsable("validate_modulus_good");
  test_throws("validate_modulus_bad", 
              "both operands of % must be int; cannot modulo real by real");
}

TEST(gmParserTermGrammar, multiplicationFun) {
  test_parsable("validate_multiplication");
}

TEST(gmParserTermGrammar, divisionFun) {
  test_warning("validate_division_int_warning", 
               "integer division implicitly rounds");
  test_parsable("validate_division_good");
}


TEST(gmParserTermGrammar, leftDivisionFun) {
  test_parsable("validate_left_division_good");
}

TEST(gmParserTermGrammar, eltMultiplicationFun) {
  test_parsable("validate_elt_multiplication_good");
}

TEST(gmParserTermGrammar, eltDivisionFun) {
  test_parsable("validate_elt_division_good");
}

TEST(gmParserTermGrammar, negateExprFun) {
  test_parsable("validate_negate_expr_good");
}

TEST(gmParserTermGrammar, logicalNegateExprFun) {
  test_throws("validate_logical_negate_expr_bad",
              "logical negation operator ! only applies to int or real");
  test_parsable("validate_logical_negate_expr_good");
}

TEST(gmParserTermGrammar, addExpressionDimssFun) {
  test_throws("validate_add_expression_dimss_bad",
              "indexes inappropriate");
  test_parsable("validate_add_expression_dimss_good");
}

TEST(gmParserTermGrammar, setFunTypeNamed) {
  test_throws("validate_set_fun_type_named_bad1",
              "random number generators only allowed in generated quantities");
  test_parsable("validate_set_fun_type_named_good");
}
