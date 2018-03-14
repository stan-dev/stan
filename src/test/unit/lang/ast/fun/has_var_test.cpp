#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>


TEST(langAst, hasVar) {
  using stan::lang::var_decl;
  using stan::lang::int_type;
  using stan::lang::double_type;
  using stan::lang::binary_op;
  using stan::lang::expression;
  using stan::lang::model_name_origin;
  using stan::lang::parameter_origin;
  using stan::lang::unary_op;
  using stan::lang::scope;
  using stan::lang::variable;
  using stan::lang::variable_map;

  variable_map vm;

  var_decl alpha_decl = var_decl("alpha", double_type());
  scope alpha_origin = parameter_origin;
  vm.add("alpha", alpha_decl, alpha_origin);

  variable v("alpha");
  v.set_type(double_type());
  expression e(v);
  EXPECT_TRUE(has_var(e, vm));

  var_decl beta_decl = var_decl("beta", int_type());
  vm.add("beta", beta_decl, model_name_origin);

  variable v_beta("beta");
  v_beta.set_type(int_type());
  expression e_beta(v_beta);
  EXPECT_FALSE(has_var(e_beta, vm));

  expression e2(binary_op(e,"+",e));
  EXPECT_TRUE(has_var(e2,vm));

  expression e_beta2(unary_op('!',unary_op('-',e_beta)));
  EXPECT_FALSE(has_var(e_beta2,vm));
}

// was part of test/unit/lang/ast_test.hpp

template <typename T>
void expect_has_var_bool(const T& x) {
  EXPECT_TRUE(x.has_var_ == 0 || x.has_var_ == 1);
}

TEST(StanLangAst, ConditionalOp) {
  expect_has_var_bool(stan::lang::conditional_op());

  stan::lang::expression e = stan::lang::int_literal(3);
  expect_has_var_bool(stan::lang::conditional_op(e, e, e));
}

TEST(StanLangAst, RowVectorExpr) {
  expect_has_var_bool(stan::lang::row_vector_expr());
}

TEST(StanLangAst, MatrixExpr) {
  expect_has_var_bool(stan::lang::matrix_expr());
}
