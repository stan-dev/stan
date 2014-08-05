#include <gtest/gtest.h>
#include <test/unit/gm/utility.hpp>

TEST(gmParserVarDeclsGrammarDef, addVar) {
  test_throws("validate_add_var_bad1",
              "duplicate declaration of variable");
  test_throws("validate_add_var_bad2",
              "integer parameters or transformed parameters are not allowed");
  test_parsable("validate_add_var_good");
}

TEST(gmParserVarDeclsGrammarDef, validateIntExpr) {
  test_parsable("validate_validate_int_expr_good");
  for (int i = 1; i <= 13; ++i) {
    std::string model_name("validate_validate_int_expr_bad");
    model_name += boost::lexical_cast<std::string>(i);
    test_throws(model_name,
                "expression denoting integer required");
  }
}

TEST(gmParserVarDeclsGrammarDef, setIntRangeLower) {
  test_parsable("validate_set_int_range_lower_good");
  test_throws("validate_set_int_range_lower_bad1",
              "expression denoting integer required");
  test_throws("validate_set_int_range_lower_bad2",
              "expression denoting integer required");
  test_throws("validate_set_int_range_lower_bad3",
              "expression denoting integer required");
}

TEST(gmParserVarDeclsGrammarDef, setIntRangeUpper) {
  test_parsable("validate_set_int_range_upper_good");
  test_throws("validate_set_int_range_upper_bad1",
              "expression denoting integer required");
  test_throws("validate_set_int_range_upper_bad2",
              "expression denoting integer required");
}

TEST(gmParserVarDeclsGrammarDef, setDoubleRangeLower) {
  test_parsable("validate_set_double_range_lower_good");
  test_throws("validate_set_double_range_lower_bad1",
              "expression denoting real required");
  test_throws("validate_set_double_range_lower_bad2",
              "expression denoting real required");
}

TEST(gmParserVarDeclsGrammarDef, setDoubleRangeUpper) {
  test_parsable("validate_set_double_range_upper_good");
  test_throws("validate_set_double_range_upper_bad1",
              "expression denoting real required");
  test_throws("validate_set_double_range_upper_bad2",
              "expression denoting real required");
}

TEST(gmParserVarDeclsGrammarDef, parametersInLocals) {
  // test_parsable("var-decls-in-functions");
   test_throws("var-decl-bad-1",
               "non-data variables not allowed in dimension declarations");
}
