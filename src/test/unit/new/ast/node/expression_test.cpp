#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

using std::vector;

TEST(astExpression, nil) {
  stan::lang::expression e1 = stan::lang::nil();
  stan::lang::expression e2 = stan::lang::double_literal(-2.0);
  EXPECT_TRUE(stan::lang::is_nil(e1));
  EXPECT_FALSE(stan::lang::is_nil(e2));
}

TEST(astExpression, int_literal) {
  stan::lang::int_literal intLit(5);
  EXPECT_TRUE(intLit.type_.is_int_type());
  EXPECT_EQ(intLit.val_, 5);

  stan::lang::expression e1 = intLit;
  EXPECT_TRUE(e1.bare_type().is_int_type());
}

TEST(astExpression, double_literal) {
  stan::lang::double_literal dblLit(5.1);
  EXPECT_TRUE(dblLit.type_.is_double_type());
  EXPECT_EQ(dblLit.val_, 5.1);

  stan::lang::expression e1 = dblLit;
  EXPECT_TRUE(e1.bare_type().is_double_type());
}

TEST(astExpression, row_vector_expr) {
  stan::lang::double_literal dblLit(5.1);
  stan::lang::expression e1 = dblLit;
  std::vector<stan::lang::expression> elements;
  elements.push_back(e1);
  elements.push_back(e1);
  stan::lang::row_vector_expr rv1(elements);
  stan::lang::expression e2 = rv1;
  EXPECT_TRUE(e2.bare_type().is_row_vector_type());
}

TEST(astExpression, matrix_expr) {
  stan::lang::double_literal dblLit(5.1);
  stan::lang::expression e1 = dblLit;
  std::vector<stan::lang::expression> elements;
  elements.push_back(e1);
  elements.push_back(e1);
  elements.push_back(e1);
  elements.push_back(e1);
  stan::lang::matrix_expr m1(elements);
  stan::lang::expression e2 = m1;
  EXPECT_TRUE(e2.bare_type().is_matrix_type());
}

TEST(astExpression, array_expr) {
  stan::lang::double_literal dblLit(5.1);
  stan::lang::expression e1 = dblLit;
  std::vector<stan::lang::expression> elements;
  elements.push_back(e1);
  elements.push_back(e1);
  elements.push_back(e1);
  stan::lang::array_expr ar1(elements);
  stan::lang::expression e2 = ar1;
  EXPECT_TRUE(e2.bare_type().is_array_type());
}

TEST(astExpression, variable_expr) {
  stan::lang::matrix_type tMat;
  stan::lang::bare_array_type d1(tMat);
  stan::lang::bare_expr_type x(d1);
  stan::lang::bare_array_type d2(x);
  stan::lang::bare_expr_type y(d2);
  stan::lang::variable v("foo");
  v.set_type(y);
  stan::lang::expression e1 = v;
  EXPECT_TRUE(e1.bare_type().is_array_type());
  EXPECT_EQ(e1.bare_type().order_id(), "array_array_06_matrix_type");
}

TEST(astExpression, fun_expr) {
  stan::lang::double_literal dblLit(5.1);
  stan::lang::expression e1 = dblLit;
  std::vector<stan::lang::expression> args;
  args.push_back(e1);
  stan::lang::fun f("x", args);
  f.type_ = stan::lang::void_type();
  stan::lang::expression e2 = f;
  EXPECT_TRUE(e2.bare_type().is_void_type());
}

// TODO:mitzi need similar test for algebra_solver_control
TEST(astExpression, algebra_solver) {
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

  // example of instantiation
  stan::lang::algebra_solver so2(system_function_name, y, theta, x_r, x_i);
  // check algebra_solver
  EXPECT_EQ(system_function_name, so2.system_function_name_);
  EXPECT_EQ(y.type_, so2.y_.bare_type());
  EXPECT_EQ(theta.type_, so2.theta_.bare_type());
  EXPECT_EQ(x_r.type_, so2.x_r_.bare_type());
  EXPECT_EQ(x_i.type_, so2.x_i_.bare_type());

  stan::lang::expression e1 = so2;
  // check expression
  EXPECT_TRUE(e1.bare_type().is_vector_type());
}

// TODO:mitzi need similar test for integrate_ode_control
TEST(astExpression, integrate_ode) {
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

  // example of instantiation
  stan::lang::integrate_ode so2(integration_function_name, system_function_name,
                    y0, t0, ts, theta, x, x_int);
  // check integrate_ode
  EXPECT_EQ(integration_function_name, so2.integration_function_name_);
  EXPECT_EQ(system_function_name, so2.system_function_name_);
  EXPECT_EQ(y0.type_, so2.y0_.bare_type());
  EXPECT_EQ(t0.type_, so2.t0_.bare_type());
  EXPECT_EQ(ts.type_, so2.ts_.bare_type());
  EXPECT_EQ(theta.type_, so2.theta_.bare_type());
  EXPECT_EQ(x.type_, so2.x_.bare_type());
  EXPECT_EQ(x_int.type_, so2.x_int_.bare_type());

  stan::lang::expression e1(so2);
  // check expression
  EXPECT_EQ(e1.bare_type().order_id(), "array_array_03_double_type");
}

TEST(astExpression, binary_op) {
  stan::lang::expression e1 = stan::lang::int_literal(5);
  stan::lang::expression e2 = stan::lang::double_literal(-2.0);

  stan::lang::expression e3(stan::lang::binary_op(e1,"+",e2));
  EXPECT_TRUE(e3.bare_type().is_double_type());

  stan::lang::expression e4(stan::lang::binary_op(e1,"+",e1));
  EXPECT_TRUE(e4.bare_type().is_int_type());

  stan::lang::expression e5(stan::lang::binary_op(e2,"+",e2));
  EXPECT_TRUE(e5.bare_type().is_double_type());
}

TEST(astExpression, unary_op) {
  stan::lang::expression e1 = stan::lang::int_literal(5);
  stan::lang::expression neg_e1(stan::lang::unary_op('-',e1));
  EXPECT_TRUE(neg_e1.bare_type().is_int_type());
}

TEST(astExpression, index_op) {
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
  EXPECT_EQ(dim.size(), 1);
  dimss.push_back(dim);
  stan::lang::index_op i_op(e2, dimss);
  stan::lang::expression e3 = i_op;
  EXPECT_TRUE(e3.bare_type().is_double_type());
}

TEST(astExpression, index_op_sliced) {
  stan::lang::variable v1("foo");
  // multi-idx index expression
  v1.set_type(stan::lang::bare_array_type(stan::lang::int_type()));
  stan::lang::expression e1(v1);
  stan::lang::multi_idx i1(e1);
  stan::lang::idx idx(i1);
  EXPECT_TRUE(stan::lang::is_multi_index(i1));
  std::vector<stan::lang::idx> idxs;
  idxs.push_back(idx);

  stan::lang::bare_expr_type d1 = stan::lang::bare_array_type(stan::lang::int_type());
  stan::lang::bare_expr_type d2 = stan::lang::bare_array_type(d1);
  stan::lang::variable v2("bar");
  v2.set_type(d2);
  stan::lang::expression e2(v2);
  stan::lang::index_op_sliced i_op_slice(e2, idxs);
  stan::lang::expression e3 = i_op_slice;
  EXPECT_EQ(e3.bare_type(), d2);
}
