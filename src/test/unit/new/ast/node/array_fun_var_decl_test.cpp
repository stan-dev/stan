#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <iostream>

TEST(arrayFunVarDecl, createVar1) {
  // 1-d array of real
  stan::lang::double_type dbl;
  stan::lang::bare_array_type batDbl(dbl);
  stan::lang::array_fun_var_decl x("x", batDbl);

  // check array_fun_var_decl
  EXPECT_TRUE(x.bare_type_.is_array_type());
  EXPECT_TRUE(x.bare_type_.array_element_type().is_double_type());
  EXPECT_EQ(x.bare_type_.array_dims(), 1);
  
  // check fun_var_decl wrapper
  stan::lang::fun_var_decl fvar(x);
  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::bare_array_type(stan::lang::double_type()));

  EXPECT_TRUE(fvar.bare_type().is_array_type());
  EXPECT_TRUE(fvar.bare_type().array_contains().is_double_type());
  EXPECT_EQ(fvar.bare_type().array_dims(), 1);
  EXPECT_EQ(fvar.bare_type().num_dims(), 1);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, fvar.bare_type());
  EXPECT_EQ("real[ ]", ss.str());
}

TEST(arrayFunVarDecl, createVar2) {
  stan::lang::matrix_type tMat;
  stan::lang::bare_array_type batMat(tMat);
  stan::lang::array_fun_var_decl x("x", batMat);

  //  check fun_array_type
  EXPECT_TRUE(x.bare_type_.is_array_type());
  EXPECT_TRUE(x.bare_type_.array_element_type().is_matrix_type());
  EXPECT_EQ(x.bare_type_.array_dims(), 1);
  EXPECT_EQ(x.bare_type_.num_dims(), 3);
  
  //  check fun_var_decl wrapper
  stan::lang::fun_var_decl fvar(x);
  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::bare_array_type(stan::lang::matrix_type()));

  EXPECT_TRUE(fvar.bare_type().is_array_type());
  EXPECT_TRUE(fvar.bare_type().array_contains().is_matrix_type());
  EXPECT_EQ(fvar.bare_type().array_dims(), 1);
  EXPECT_EQ(fvar.bare_type().num_dims(), 3);

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, fvar.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

