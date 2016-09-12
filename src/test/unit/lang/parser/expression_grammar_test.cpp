#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParserExpressionGrammarDef, intDivUserFacing) {
  test_warning("int_div_user",
               "a[1] / b[2]");
}

TEST(langParserExpressionGrammarDef, absDeprecate) {
  test_warning("abs-deprecate",
               "Warning: Function abs(real) is deprecated in the Stan language.");
}

TEST(langParserExpressionGrammarDef, conditionalOp) {
  test_parsable("validate_conditional_op_good");
}

TEST(langParserExpressionGrammarDef, conditionalOpBad1) {
  test_throws("validate_conditional_op_bad-1","condition in ternary expression");
}

TEST(langParserExpressionGrammarDef, conditionalOpBad2) {
  test_throws("validate_conditional_op_bad-2","base type mismatch");
}
