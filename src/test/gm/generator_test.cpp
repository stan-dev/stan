#include <gtest/gtest.h>
#include <stan/agrad/var.hpp>
#include <stan/gm/ast.hpp>
#include <stan/gm/generator.hpp>
#include <test/gm/model_specs/no_main/test_lp.cpp>
#include <stan/io/dump.hpp>


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


TEST(gm, logProbPolymorphismDouble) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  std::string txt = "foo <- 3\nbar <- 4";
  std::stringstream in(txt);
  stan::io::dump dump(in);

  test_lp_namespace::test_lp model(dump);

  std::vector<double> params_r(2);
  params_r[0] = 1.0;
  params_r[1] = -3.2;

  std::vector<int> params_i;
  
  Matrix<double, Dynamic, 1> params_r_vec(2);
  for (int i = 0; i < 2; ++i)
    params_r_vec(i) = params_r[i];

  double lp1 = model.log_prob<true,true>(params_r, params_i, 0);
  double lp2 = model.log_prob<true,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);

  lp1 = model.log_prob<true,false>(params_r, params_i, 0);
  lp2 = model.log_prob<true,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);

  lp1 = model.log_prob<false,true>(params_r, params_i, 0);
  lp2 = model.log_prob<false,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);

  lp1 = model.log_prob<false,false>(params_r, params_i, 0);
  lp2 = model.log_prob<false,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);
}
TEST(gm, logProbPolymorphismVar) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::var;

  std::string txt = "foo <- 3\nbar <- 4";
  std::stringstream in(txt);
  stan::io::dump dump(in);

  test_lp_namespace::test_lp model(dump);

  std::vector<var> params_r(2);
  params_r[0] = 1.0;
  params_r[1] = -3.2;

  std::vector<int> params_i;
  
  Matrix<var, Dynamic, 1> params_r_vec(2);
  for (int i = 0; i < 2; ++i)
    params_r_vec(i) = params_r[i];

  var lp1 = model.log_prob<true,true>(params_r, params_i, 0);
  var lp2 = model.log_prob<true,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());

  lp1 = model.log_prob<true,false>(params_r, params_i, 0);
  lp2 = model.log_prob<true,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());

  lp1 = model.log_prob<false,true>(params_r, params_i, 0);
  lp2 = model.log_prob<false,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());

  lp1 = model.log_prob<false,false>(params_r, params_i, 0);
  lp2 = model.log_prob<false,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());
}
