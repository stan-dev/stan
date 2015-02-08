#include <gtest/gtest.h>
#include <test/unit/gm/utility.hpp>

TEST(GmGrammars,test1) {
  test_throws("err-open-block", 
              "PARSER EXPECTED: \"{");
  test_throws("err-close-block", 
              "PARSER EXPECTED: \"}");
  test_throws("err-transformed-params", 
              "PARSER EXPECTED: \"parameters");
  test_throws("err-expected-model", 
              "PARSER EXPECTED: <model");
  test_throws("err-expected-generated", 
              "PARSER EXPECTED: \"quantities");
  test_throws("err-expected-bracket", 
              "PARSER EXPECTED: \"{");
  test_throws("err-expected-end-of-model", 
              "PARSER EXPECTED: whitespace to end of file");
  test_throws("err-bare-type-close-square", 
              "PARSER EXPECTED: \"]");
  test_throws("err-second-operand-plus", 
              "PARSER EXPECTED: <expression>");
  test_throws("err-nested-parens", 
              "PARSER EXPECTED: <expression>");
  test_throws("err-nested-parens-close", 
              "PARSER EXPECTED: \")");
  test_throws("err-integrate-ode-comma",
              "PARSER EXPECTED: \",");
  test_throws("err-non-int-dims",
              "array indices must be integer expressions; found type=real");
  test_throws("err-no-cond-else-if",
              "PARSER EXPECTED: \"(");
  test_throws("err-no-cond",
              "PARSER EXPECTED: <expression>");
  test_throws("err-no-statement",
              "PARSER EXPECTED: <statement>");
  test_throws("err-incr-log-prob-scope",
              "Sampling statements (~) and increment_log_prob() are");
}
 
