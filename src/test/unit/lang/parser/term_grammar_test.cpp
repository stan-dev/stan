#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParserTermGrammar, vector_expr) {
  test_parsable("vec-expr/row_vector_expr_terms");
  test_throws("row_vector_expr_bad1",
              "variable definition base type mismatch");
  test_throws("row_vector_expr_bad2",
              "Row vector expression elements must be int or real");
  test_throws("row_vector_expr_bad3",
              "PARSER EXPECTED: <expression>");
}

TEST(langParserTermGrammar, matrix_expr) {
  test_parsable("vec-expr/matrix_expr_terms");
  test_throws("matrix_expr_bad1",
              "variable definition base type mismatch");
  test_throws("matrix_expr_bad2",
              "Matrix expression elements must be type row_vector");
  test_throws("matrix_expr_bad3",
              "Matrix expression elements must be type row_vector");
  test_throws("matrix_expr_bad4",
              "found element of type matrix");
  test_throws("matrix_expr_bad5",
              "PARSER EXPECTED: <expression>");
}

TEST(langParserTermGrammar, array_expr) {
  test_parsable("array-expr/validate_array_expr_big");
  test_parsable("array-expr/validate_array_expr_primitives");
  test_parsable("array-expr/validate_array_expr_containers");
  test_throws("validate_array_expr_bad1", "base type mismatch");
  test_throws("validate_array_expr_bad2", "dimension mismatch");
  test_throws("validate_array_expr_bad3", "ERROR");
  test_throws("validate_array_expr_bad3a", "ERROR");
}

TEST(langParserTermGrammar, infixExponentiation) {
  test_parsable("validate_exponentiation_good");
  test_parsable("validate_exponentiation_precedence");
  test_throws("validate_exponentiation_bad", 
              "base type mismatch in assignment; variable name = z");
}

TEST(langParserTermGrammar, modulusOp) {
  test_parsable("validate_modulus_good");
  test_throws("validate_modulus_bad", 
              "both operands of % must be int; cannot modulo real by real");
}

TEST(langParserTermGrammar, multiplicationFun) {
  test_parsable("validate_multiplication");
}

TEST(langParserTermGrammar, divisionFun) {
  test_warning("validate_division_int_warning", 
               "integer division implicitly rounds");
  test_parsable("validate_division_good");
}


TEST(langParserTermGrammar, leftDivisionFun) {
  test_parsable("validate_left_division_good");
}

TEST(langParserTermGrammar, eltMultiplicationFun) {
  test_parsable("validate_elt_multiplication_good");
}

TEST(langParserTermGrammar, eltDivisionFun) {
  test_parsable("validate_elt_division_good");
}

TEST(langParserTermGrammar, negateExprFun) {
  test_parsable("validate_negate_expr_good");
}

TEST(langParserTermGrammar, logicalNegateExprFun) {
  test_throws("validate_logical_negate_expr_bad",
              "logical negation operator ! only applies to int or real");
  test_parsable("validate_logical_negate_expr_good");
}

TEST(langParserTermGrammar, addExpressionDimssFun) {
  test_throws("validate_add_expression_dimss_bad",
              "Indexed expression must have at least as many dimensions");
  test_parsable("validate_add_expression_dimss_good");
}

TEST(langParserTermGrammar, setFunTypeNamed) {
  test_parsable("validate_set_fun_type_named_good");
}

TEST(langGrammarsTermGrammar, operatorErrorMsg) {
  test_throws("op_addition_bad",
              "matrix + vector",
              "Available argument signatures for operator+");
  test_throws("op_subtraction_bad",
              "vector - matrix",
              "Available argument signatures for operator-");
  test_throws("op_multiplication_bad",
              "int[] * matrix",
              "Available argument signatures for operator*");
  test_throws("op_divide_bad",
              "int[] / matrix",
              "Available argument signatures for operator/");
  test_throws("op_modulus_bad",
              "both operands of % must be int; cannot modulo int[] by matrix");
  test_throws("op_mdivide_left_bad",
              "int[] \\ matrix",
              "Available argument signatures for operator\\");
  test_throws("op_divide_right_bad",
              "int[] / matrix",
              "Available argument signatures for operator/");
  test_throws("op_elt_multiply_bad",
              "int[] .* matrix",
              "Available argument signatures for operator.*");
  test_throws("op_elt_divide_bad",
              "int[] ./ matrix",
              "Available argument signatures for operator./");
  test_throws("op_minus_bad",
              "-int[]",
              "Available argument signatures for operator-");
  test_throws("op_logical_negation_bad",
              "!int[]",
              "Available argument signatures for operator!");
  test_throws("op_transpose_bad",
              "int[]'",
              "Available argument signatures for operator'");
}
