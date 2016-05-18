#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParserStatementGrammarDef, intDivUserFacing) {
  test_warning("int_div_user",
               "a[1] / b[2]");
}

TEST(langParserStatementGrammarDef, absDeprecate) {
  test_warning("abs-deprecate",
               "Warning: Function abs(real) is deprecated in the Stan language.");
}
