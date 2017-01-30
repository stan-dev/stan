#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParserVarDeclsGrammarDef, addVar) {
  test_throws("validate_add_var_bad1",
              "duplicate declaration of variable");
  test_throws("validate_add_var_bad2",
              "integer parameters or transformed parameters are not allowed");
  test_parsable("validate_add_var_good");
}

TEST(langParserVarDeclsGrammarDef, validateIntExpr) {
  test_parsable("validate_validate_int_expr_good");
  for (int i = 1; i <= 13; ++i) {
    std::string model_name("validate_validate_int_expr_bad");
    model_name += boost::lexical_cast<std::string>(i);
    test_throws(model_name,
                "dimension declaration requires expression"
                " denoting integer; found type=real");
  }
  for (int i = 1; i <= 14; ++i) {
    std::string model_name("data_index/non_data_index");
    model_name += boost::lexical_cast<std::string>(i);
    test_throws(model_name,
                "non-data variables not allowed in dimension declarations");
  }
}

TEST(langParserVarDeclsGrammarDef, setIntRangeLower) {
  test_parsable("validate_set_int_range_lower_good");
  test_throws("validate_set_int_range_lower_bad1",
              "expression denoting integer required");
  test_throws("validate_set_int_range_lower_bad2",
              "expression denoting integer required");
  test_throws("validate_set_int_range_lower_bad3",
              "expression denoting integer required");
}

TEST(langParserVarDeclsGrammarDef, setIntRangeUpper) {
  test_parsable("validate_set_int_range_upper_good");
  test_throws("validate_set_int_range_upper_bad1",
              "expression denoting integer required");
  test_throws("validate_set_int_range_upper_bad2",
              "expression denoting integer required");
}

TEST(langParserVarDeclsGrammarDef, setDoubleRangeLower) {
  test_parsable("validate_set_double_range_lower_good");
  test_throws("validate_set_double_range_lower_bad1",
              "expression denoting real required");
  test_throws("validate_set_double_range_lower_bad2",
              "expression denoting real required");
}

TEST(langParserVarDeclsGrammarDef, setDoubleRangeUpper) {
  test_parsable("validate_set_double_range_upper_good");
  test_throws("validate_set_double_range_upper_bad1",
              "expression denoting real required");
  test_throws("validate_set_double_range_upper_bad2",
              "expression denoting real required");
}

TEST(langParserVarDeclsGrammarDef, parametersInLocals) {
  // test_parsable("var-decls-in-functions");
  test_throws("var-decl-bad-1",
              "non-data variables not allowed in dimension declarations");
}

TEST(langParserVarDeclsGrammarDef, constraintsInLocals) {
  test_throws("local_var_constraint",
              "require unconstrained. found range constraint.");
  test_throws("local_var_constraint2",
              "require unconstrained. found range constraint.");
  test_throws("local_var_constraint3",
              "require unconstrained. found range constraint.");
}

TEST (langParserVarDeclsGrammarDef, zeroVecs) {
  test_parsable("vector-zero");
}

TEST(langParserVarDeclsGrammarDef, defDeclIntVar) {
  test_parsable("declare-define-var-int");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclIntVar1) {
  test_throws("declare-define-var-int-1",
              "variable definition base type mismatch");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclIntVar2) {
  test_throws("declare-define-var-int-2",
              "variable definition base type mismatch");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclIntVar3) {
  test_throws("declare-define-var-int-3",
              "variable definition dimensions mismatch");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclIntVar4) {
  test_throws("declare-define-var-int-4",
              "variable definition dimensions mismatch");
}

TEST(langParserVarDeclsGrammarDef, defDeclDoubleVar) {
  test_parsable("declare-define-var-double");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclDoubleVar1) {
  test_throws("declare-define-var-double-1",
              "variable definition dimensions mismatch");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclDoubleVar2) {
  test_throws("declare-define-var-double-2",
              "variable definition base type mismatch");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclDoubleVar3) {
  test_throws("declare-define-var-double-3",
              "variable definition not possible in this block");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclDoubleVar4) {
  test_throws("declare-define-var-double-4",
              "variable definition not possible in this block");
}

TEST(langParserVarDeclsGrammarDef, defDeclVecTypesVar) {
  test_parsable("declare-define-var-vec-types");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclVec1) {
  test_throws("declare-define-var-vec-1",
              "variable definition base type mismatch");
}

TEST(langParserVarDeclsGrammarDef, defDeclMatrixVar) {
  test_parsable("declare-define-var-matrix");
}

TEST(langParserVarDeclsGrammarDef, badDefDeclMatrix1) {
  test_throws("declare-define-var-matrix-1",
              "variable definition base type mismatch");
}

TEST(langParserVarDeclsGrammarDef, defDeclConstrainedVectorVar) {
  test_parsable("declare-define-var-constrained-vector");
}

TEST(langParserVarDeclsGrammarDef, defDeclConstrainedMatrixVar) {
   test_parsable("declare-define-var-constrained-matrix");
}


