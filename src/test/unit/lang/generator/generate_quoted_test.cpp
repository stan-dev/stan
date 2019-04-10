#include <stan/lang/generator.hpp>
#include <test/unit/lang/utility.hpp>
#include <gtest/gtest.h>
#include <sstream>

void test_generate_quoted_string(const std::string& s,
                                 const std::string& quoted_s) {
  std::stringstream ss;
  stan::lang::generate_quoted_string(s, ss);
  EXPECT_EQ(quoted_s, ss.str());
}

void test_generate_quoted_string_quote(const std::string& s,
                           const std::string& expected_output_content) {
  std::stringstream ss;
  stan::lang::generate_quoted_string(s,ss);
  std::string s_rendered = ss.str();
  EXPECT_EQ("\"" + expected_output_content + "\"", ss.str());
}

void test_generate_quoted_expression(const stan::lang::expression& e,
                                     const std::string& e_exp) {
  std::stringstream ss;
  stan::lang::generate_quoted_expression(e, ss);
  EXPECT_EQ(e_exp, ss.str());
}

TEST(langGenerator, quotedString) {
  test_generate_quoted_string_quote("","");
  test_generate_quoted_string_quote("abc", "abc");
  test_generate_quoted_string_quote("abc'def", "abc\\'def");
  test_generate_quoted_string_quote("\"abc", "\\\"abc");
  test_generate_quoted_string_quote("abc\"", "abc\\\"");
  test_generate_quoted_string_quote("abc\"def", "abc\\\"def");
  test_generate_quoted_string_quote("abc\"def\"ghi", "abc\\\"def\\\"ghi");
}

TEST(lang,printStringLiteral) {
  test_generate_quoted_string("", "\"\"");
  test_generate_quoted_string("\\d\\", "\"\\\\d\\\\\"");
  test_generate_quoted_string("ab\"c", "\"ab\\\"c\"");
  test_generate_quoted_string("'hey,' he said.","\"\\'hey,\\' he said.\"");
}

TEST(lang,printQuotedExpression) {
  using stan::lang::expression;
  using stan::lang::index_op;
  using stan::lang::int_literal;
  using stan::lang::variable;
  using std::vector;
  test_generate_quoted_expression(int_literal(1), "\"1\"");
  vector<expression> args;

  expression expr(variable("foo"));
  vector<vector<expression> > dimss;
  vector<expression> dim;
  dim.push_back(int_literal(1));
  dimss.push_back(dim);
  std::string s_exp = "\"get_base1(foo, 1, \\\"foo\\\", 1)\"";
  test_generate_quoted_expression(index_op(expr,dimss), s_exp);
}
