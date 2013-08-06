#include <gtest/gtest.h>
#include <stan/gm/ast.hpp>
#include <stan/gm/generator.hpp>



void test_print_string_literal(const std::string& s,
                               const std::string& s_exp) {
  std::stringstream ss;
  stan::gm::print_string_literal(ss,s);
  EXPECT_EQ(s_exp, ss.str());
}

void test_print_quoted_expression(const stan::gm::expression& e,
                                  const std::string& e_exp) {
  std::stringstream ss;
  stan::gm::print_quoted_expression(ss,e);
  EXPECT_EQ(e_exp, ss.str());
}

TEST(gm,printStringLiteral) {
  test_print_string_literal("","\"\"");
  test_print_string_literal("\\d\\","\"\\\\d\\\\\"");
  test_print_string_literal("ab\"c", "\"ab\\\"c\"");
  test_print_string_literal("'hey,' he said.","\"\\'hey,\\' he said.\"");
}

TEST(gm,printQuotedExpression) {
  using stan::gm::expression;
  using stan::gm::index_op;
  using stan::gm::int_literal;
  using stan::gm::variable;
  using std::vector;
  test_print_quoted_expression(int_literal(1),"\"1\"");
  vector<expression> args;

  expression expr(variable("foo"));
  vector<vector<expression> > dimss;
  vector<expression> dim;
  dim.push_back(int_literal(1));
  dimss.push_back(dim);
  std::string s_exp = "\"get_base1(foo,1,\\\"foo\\\",1)\"";
  test_print_quoted_expression(index_op(expr,dimss),
                               s_exp);
}
