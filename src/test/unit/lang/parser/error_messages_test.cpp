#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(LangGrammars,test1) {
  test_throws("err-open-block",
              "PARSER EXPECTED: \"{");
  test_throws("err-close-block",
              "PARSER EXPECTED: \"}");
  test_throws("err-transformed-params",
              "PARSER EXPECTED: \"parameters");
  test_throws("err-expected-model",
              "PARSER FAILED TO PARSE INPUT COMPLETELY");
  test_throws("err-expected-generated",
              "PARSER EXPECTED: \"quantities");
  test_throws("err-expected-bracket",
              "PARSER EXPECTED: \"{");
  test_throws("err-expected-end-of-model",
              "PARSER FAILED TO PARSE INPUT COMPLETELY");
  test_throws("err-second-operand-plus",
              "PARSER EXPECTED: <expression>");
  test_throws("err-nested-parens",
              "PARSER EXPECTED: <expression>");
  test_throws("err-nested-parens-close",
              "PARSER EXPECTED: \")");
  test_throws("err-integrate-ode-comma",
              "PARSER EXPECTED: \",");
  test_throws("err-non-int-dims",
              "index must be integer; found type=real");
  test_throws("err-no-cond-else-if",
              "PARSER EXPECTED: \"(");
  test_throws("err-no-cond",
              "PARSER EXPECTED: <expression>");
  test_throws("err-no-statement",
              "PARSER EXPECTED: <statement>");
  test_throws("err-incr-log-prob-scope",
              "Sampling statements (~) and increment_log_prob() are");
  test_throws("err-decl-vector",
              "PARSER EXPECTED: <vector length declaration");
  test_throws("err-decl-vector-2",
              "PARSER EXPECTED: \"]");
  test_throws("err-decl-matrix",
              "PARSER EXPECTED: \",");
  test_throws("err-decl-matrix-2",
              "Too many indexes, expression dimensions=0, indexes found=1");
  test_throws("err-decl-no-expression",
              "PARSER EXPECTED: <expression>");
  test_throws("err-decl-double",
              "a variable declaration, beginning with type");
  test_throws("err-decl-double-params",
              "Variable \"lijaflj\" does not exist");
  test_throws("err-fun-bare-types-int",
              "comma to indicate more dimensions or ]");
  test_throws("err-bare-type-close-square",
              "comma to indicate more dimensions or ] to end type declaration");
  test_throws("err-close-function-args",
              "PARSER EXPECTED: <argument declaration or close paren");
  test_throws("err-if-else",
              "PARSER EXPECTED: \")");
  test_throws("err-if-else-no-cond",
              "PARSER EXPECTED: <expression>");
  test_throws("err-if-else-double-else",
              "Variable \"else\" does not exist.");
  test_throws("err-var-decl-after-statement",
              "Variable \"real\" does not exist");
  test_throws("err-double-dims",
              "Dimension declaration requires expression denoting integer; found type=real");
  test_throws("oneline-error",
              "1: parameters { vector y[10]; } model { }");
}

