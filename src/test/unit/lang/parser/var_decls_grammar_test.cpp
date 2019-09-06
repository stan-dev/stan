#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParserVarDeclsGrammarDef, addVar) {
  test_throws("validate_add_var_bad1", "Duplicate declaration of variable");
  test_throws("validate_add_var_bad2", "Parameters or transformed parameters "
                                       "cannot be integer or integer array");
  test_parsable("validate_add_var_good");
}

TEST(langParserVarDeclsGrammarDef, validateIntExpr) {
  test_parsable("validate_validate_int_expr_good");
  for (int i = 1; i <= 13; ++i) {
    std::string model_name("validate_validate_int_expr_bad");
    model_name += boost::lexical_cast<std::string>(i);
    test_throws(model_name, "Dimension declaration requires");
  }
  for (int i = 1; i <= 14; ++i) {
    std::string model_name("data_index/non_data_index");
    model_name += boost::lexical_cast<std::string>(i);
    test_throws(model_name,
                "Non-data variables are not allowed in dimension declarations");
  }
}

TEST(langParserVarDeclsGrammarDef, setIntRangeLower) {
  test_parsable("validate_set_int_range_lower_good");
  test_throws("validate_set_int_range_lower_bad1",
              "Expression denoting integer required");
  test_throws("validate_set_int_range_lower_bad2",
              "Expression denoting integer required");
  test_throws("validate_set_int_range_lower_bad3",
              "Expression denoting integer required");
}

TEST(langParserVarDeclsGrammarDef, setIntRangeUpper) {
  test_parsable("validate_set_int_range_upper_good");
  test_throws("validate_set_int_range_upper_bad1",
              "Expression denoting integer required");
  test_throws("validate_set_int_range_upper_bad2",
              "Expression denoting integer required");
}

TEST(langParserVarDeclsGrammarDef, setDoubleRangeLower) {
  test_parsable("validate_set_double_range_lower_good");
  test_throws("validate_set_double_range_lower_bad1",
              "Expression denoting real required");
  test_throws("validate_set_double_range_lower_bad2",
              "Expression denoting real required");
}

TEST(langParserVarDeclsGrammarDef, setDoubleRangeUpper) {
  test_parsable("validate_set_double_range_upper_good");
  test_throws("validate_set_double_range_upper_bad1",
              "Expression denoting real required");
  test_throws("validate_set_double_range_upper_bad2",
              "Expression denoting real required");
}

TEST(langParserVarDeclsGrammarDef, setDoubleOffsetMultiplier) {
  test_parsable("validate_set_double_offset_multiplier_good");
  test_throws("validate_set_double_offset_multiplier_bad1",
              "Expression denoting real required; found type=vector.");
  test_throws("validate_set_double_offset_multiplier_bad2",
              "Expression denoting real required; found type=vector.");
  test_throws("validate_set_double_offset_multiplier_bad3",
              "PARSER EXPECTED: \"upper\"");
}

TEST(langParserVarDeclsGrammarDef, parametersInLocals) {
  // test_parsable("var-decls-in-functions");
  test_throws("var-decl-bad-1",
              "Non-data variables are not allowed in dimension declarations");
}

TEST(langParserVarDeclsGrammarDef, constraintsInLocals) {
  test_throws("local_var_constraint",
              "PARSER EXPECTED: <vector length declaration");
  test_throws("local_var_constraint2",
              "PARSER EXPECTED: <vector length declaration");
  test_throws("local_var_constraint3", "PARSER EXPECTED: \"[\"");
  test_throws("local_var_constraint4", "PARSER EXPECTED: <identifier>");
}

TEST(langParserVarDeclsGrammarDef, zeroVecs) { test_parsable("vector-zero"); }

TEST(langParserVarDeclsGrammarDef, defDeclIntVar) {
  test_parsable("declare-define-var-int");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclIntVar1) {
  test_throws("declare-define-var-int-1",
              "Variable definition base type mismatch");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclIntVar2) {
  test_throws("declare-define-var-int-2",
              "Variable definition base type mismatch");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclIntVar3) {
  test_throws("declare-define-var-int-3",
              "Variable definition dimensions mismatch");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclIntVar4) {
  test_throws("declare-define-var-int-4",
              "Variable definition dimensions mismatch");
}

TEST(langParserVarDeclsGrammarDef, defDeclDoubleVar) {
  test_parsable("declare-define-var-double");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclDoubleVar1) {
  test_throws("declare-define-var-double-1",
              "Variable definition dimensions mismatch");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclDoubleVar2) {
  test_throws("declare-define-var-double-2",
              "Variable definition base type mismatch");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclDoubleVar3) {
  test_throws("declare-define-var-double-3",
              "Variable definition not possible in this block");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclDoubleVar4) {
  test_throws("declare-define-var-double-4",
              "Variable definition not possible in this block");
}

TEST(langParserVarDeclsGrammarDef, defDeclVecTypesVar) {
  test_parsable("declare-define-var-vec-types");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclVec1) {
  test_throws("declare-define-var-vec-1",
              "Variable definition base type mismatch");
}

TEST(langParserVarDeclsGrammarDef, defDeclMatrixVar) {
  test_parsable("declare-define-var-matrix");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclMatrix1) {
  test_throws("declare-define-var-matrix-1",
              "Variable definition base type mismatch");
}

TEST(langParserVarDeclsGrammarDef, defDeclConstrainedVectorVar) {
  test_parsable("declare-define-var-constrained-vector");
}

TEST(langParserVarDeclsGrammarDef, defDeclConstrainedMatrixVar) {
  test_parsable("declare-define-var-constrained-matrix");
}

TEST(langParserVarDeclsGrammarDef, badDefParamBlock) {
  test_throws("declare-define-param-block",
              "Variable definition not possible in this block");
}

TEST(langParserVarDeclsGrammarDef, gqLocalRngFunCall) {
  test_parsable("declare-define-gq-local-rng");
}
