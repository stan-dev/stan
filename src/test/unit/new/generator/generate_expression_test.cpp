#include <stan/lang/ast_def.cpp>

// all of this needed for good error messages from parser
#include <stan/lang/generator/expression_visgen.hpp>
#include <stan/lang/generator/generate_array_builder_adds.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_idxs.hpp>
#include <stan/lang/generator/generate_idxs_user.hpp>
#include <stan/lang/generator/generate_idx.hpp>
#include <stan/lang/generator/generate_idx_user.hpp>
#include <stan/lang/generator/idx_visgen.hpp>
#include <stan/lang/generator/idx_user_visgen.hpp>

#include <gtest/gtest.h>
#include <sstream>
#include <string>

TEST(generateExpression, gen_int_lit_expr) {
  static const bool user_facing = true;
  std::stringstream msgs;
  stan::lang::int_literal intLit(5);
  stan::lang::expression e1 = intLit;
  generate_expression(e1, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, gen_double_lit_expr) {
  static const bool user_facing = true;
  std::stringstream msgs;
  stan::lang::double_literal dblLit(5.10);
  stan::lang::expression e1 = dblLit;
  generate_expression(e1, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, row_vector_expr) {
  static const bool user_facing = true;
  std::stringstream msgs;

  stan::lang::double_literal dblLit(5.1);
  stan::lang::expression e1 = dblLit;
  std::vector<stan::lang::expression> elements;
  elements.push_back(e1);
  elements.push_back(e1);
  stan::lang::row_vector_expr rv1(elements);
  stan::lang::expression e2 = rv1;

  generate_expression(e2, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, matrix_expr) {
  static const bool user_facing = true;
  std::stringstream msgs;

  stan::lang::double_literal dblLit(5.1);
  stan::lang::expression e1 = dblLit;
  std::vector<stan::lang::expression> elements;
  elements.push_back(e1);
  elements.push_back(e1);
  elements.push_back(e1);
  elements.push_back(e1);
  stan::lang::matrix_expr m1(elements);
  stan::lang::expression e2 = m1;

  generate_expression(e2, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, array_expr) {
  static const bool user_facing = true;
  std::stringstream msgs;

  stan::lang::double_literal dblLit(5.1);
  stan::lang::expression e1 = dblLit;
  std::vector<stan::lang::expression> elements;
  elements.push_back(e1);
  elements.push_back(e1);
  elements.push_back(e1);
  stan::lang::array_expr ar1(elements);
  stan::lang::expression e2 = ar1;

  generate_expression(e2, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, variable_expr) {
  static const bool user_facing = true;
  std::stringstream msgs;

  stan::lang::matrix_type tMat;
  stan::lang::bare_array_type d2_ar(tMat,2);
  stan::lang::bare_expr_type bet(d2_ar);
  stan::lang::variable v("foo");
  v.set_type(bet);
  stan::lang::expression e1 = v;

  generate_expression(e1, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, fun_expr) {
  static const bool user_facing = true;
  std::stringstream msgs;

  stan::lang::double_literal dblLit(5.1);
  stan::lang::expression e1 = dblLit;
  std::vector<stan::lang::expression> args;
  args.push_back(e1);
  stan::lang::fun f("x", args);
  f.type_ = stan::lang::void_type();
  stan::lang::expression e2 = f;

  generate_expression(e2, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, algebra_solver) {
  static const bool user_facing = true;
  std::stringstream msgs;

  stan::lang::algebra_solver so;  // null ctor should work and not raise error
  std::string system_function_name = "bronzino";
  stan::lang::variable y("y_var_name");
  y.set_type(stan::lang::vector_type());
  stan::lang::variable theta("theta_var_name");
  theta.set_type(stan::lang::vector_type());
  stan::lang::variable x_r("x_r_r_var_name");
  x_r.set_type(stan::lang::bare_array_type(stan::lang::double_type()));
  stan::lang::variable x_i("x_i_var_name");
  x_i.set_type(stan::lang::bare_array_type(stan::lang::int_type()));
  stan::lang::algebra_solver so2(system_function_name, y, theta, x_r, x_i);
  stan::lang::expression e1 = so2;

  generate_expression(e1, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}


TEST(generateExpression, integrate_ode) {
  static const bool user_facing = true;
  std::stringstream msgs;

  stan::lang::integrate_ode so; // null ctor should work and not raise error

  std::string integration_function_name = "bar";
  std::string system_function_name = "foo";
  stan::lang::variable y0("y0_var_name");
  y0.set_type(stan::lang::bare_array_type(stan::lang::double_type()));
  stan::lang::variable t0("t0_var_name");
  t0.set_type(stan::lang::double_type());
  stan::lang::variable ts("ts_var_name");
  ts.set_type(stan::lang::bare_array_type(stan::lang::double_type()));
  stan::lang::variable theta("theta_var_name");
  theta.set_type(stan::lang::bare_array_type(stan::lang::double_type()));
  stan::lang::variable x("x_var_name");
  x.set_type(stan::lang::bare_array_type(stan::lang::double_type()));
  stan::lang::variable x_int("x_int_var_name");
  x.set_type(stan::lang::bare_array_type(stan::lang::int_type()));
  stan::lang::integrate_ode so2(integration_function_name, system_function_name,
                    y0, t0, ts, theta, x, x_int);
  stan::lang::expression e1(so2);

  generate_expression(e1, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, conditional_op) {
  static const bool user_facing = true;
  std::stringstream msgs;

  stan::lang::expression e1 = stan::lang::int_literal(5);
  stan::lang::expression e2 = stan::lang::double_literal(2.0);
  stan::lang::expression e3 = stan::lang::double_literal(-2.0);
  stan::lang::expression e4(stan::lang::conditional_op(e1,e2,e3));

  generate_expression(e4, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}


TEST(generateExpression, binary_op) {
  static const bool user_facing = true;
  std::stringstream msgs;

  stan::lang::expression e1 = stan::lang::int_literal(5);
  stan::lang::expression e2 = stan::lang::double_literal(-2.0);
  stan::lang::expression e3(stan::lang::binary_op(e1,"+",e2));

  generate_expression(e3, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, unary_op) {
  static const bool user_facing = true;
  std::stringstream msgs;

  stan::lang::expression e1 = stan::lang::int_literal(5);
  stan::lang::expression e2(stan::lang::unary_op('-',e1));

  generate_expression(e2, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, index_op) {
  static const bool user_facing = true;
  std::stringstream msgs;

  // expr:  row_vector
  stan::lang::double_literal dblLit(5.1);
  stan::lang::expression e1 = dblLit;
  std::vector<stan::lang::expression> elements;
  elements.push_back(e1);
  elements.push_back(e1);
  stan::lang::row_vector_expr rv1(elements);
  stan::lang::expression e2 = rv1;

  // dimensions:  vector of vector of dimensions - just 1
  std::vector<std::vector<stan::lang::expression> > dimss;
  std::vector<stan::lang::expression> dim;
  dim.push_back(stan::lang::int_literal(1));
  dimss.push_back(dim);
  stan::lang::index_op i_op(e2, dimss);
  stan::lang::expression e3 = i_op;

  generate_expression(e3, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}

TEST(generateExpression, index_op_sliced) {
  static const bool user_facing = true;
  std::stringstream msgs;

  // foo is array of int
  stan::lang::variable v1("foo");
  v1.set_type(stan::lang::bare_array_type(stan::lang::int_type()));
  stan::lang::expression e1(v1);
  // use foo as multi-idx
  stan::lang::multi_idx i1(e1);
  stan::lang::idx idx(i1);
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);

  // bar is 2-d array of int
  stan::lang::bare_expr_type d2_ar_int = stan::lang::bare_array_type(stan::lang::int_type(),2);
  stan::lang::variable v2("bar");
  v2.set_type(d2_ar_int);
  stan::lang::expression e2(v2);
  // apply multi-idx to bar
  stan::lang::index_op_sliced i_op_slice(e2, idxs);
  stan::lang::expression e3 = i_op_slice;

  generate_expression(e3, user_facing, msgs);
  std::cout << msgs.str() << std::endl;
  EXPECT_TRUE(true);
}
