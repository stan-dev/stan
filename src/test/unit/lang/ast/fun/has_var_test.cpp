#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>


TEST(langAst, hasVar) {
  using stan::lang::var_decl;
  using stan::lang::int_type;
  using stan::lang::double_type;
  using stan::lang::bare_array_type;
  using stan::lang::binary_op;
  using stan::lang::expression;
  using stan::lang::model_name_origin;
  using stan::lang::parameter_origin;
  using stan::lang::unary_op;
  using stan::lang::scope;
  using stan::lang::variable;
  using stan::lang::variable_map;

  variable_map vm;

  var_decl scalar_real_var_decl = var_decl("scalar_real_var", double_type());
  scope scalar_real_var_origin = parameter_origin;
  vm.add("scalar_real_var", scalar_real_var_decl, scalar_real_var_origin);

  variable v("scalar_real_var");
  v.set_type(double_type());
  expression e(v);
  EXPECT_TRUE(has_var(e, vm));
  expression e2(binary_op(e,"+",e));
  EXPECT_TRUE(has_var(e2,vm));

  var_decl scalar_int_var_decl = var_decl("scalar_int_var", int_type());
  vm.add("scalar_int_var", scalar_int_var_decl, model_name_origin);
  variable v_scalar_int_var("scalar_int_var");
  v_scalar_int_var.set_type(int_type());

  expression e_scalar_int_var(v_scalar_int_var);
  EXPECT_FALSE(has_var(e_scalar_int_var, vm));

  expression e3(binary_op(e,"+",e_scalar_int_var));
  EXPECT_TRUE(has_var(e3,vm));

  expression not_e_scalar_int_var(unary_op('!',unary_op('-',e_scalar_int_var)));
  EXPECT_FALSE(has_var(not_e_scalar_int_var,vm));

  var_decl array_int_var_decl = var_decl("array_int_var",
                                         bare_array_type(int_type(), 1));
  vm.add("array_int_var", array_int_var_decl, model_name_origin);
  variable v_array_int_var("array_int_var");
  v_array_int_var.set_type(bare_array_type(int_type(), 1));

  expression e_array_int_var(v_array_int_var);
  EXPECT_FALSE(has_var(e_array_int_var, vm));
}

// test initialization of composite expressions
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
