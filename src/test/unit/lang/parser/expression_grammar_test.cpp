#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParserStatementGrammarDef, intDivUserFacing) {
  test_warning("int_div_user",
               "a[1] / b[2]");
}

