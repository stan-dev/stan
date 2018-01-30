#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

TEST(matrixFunVarDecl, createVar1) {
  stan::lang::matrix_fun_var_decl x("x");

  // check matrix_bloc_var_decl
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_matrix_type());
  EXPECT_TRUE(is_nil(x.def_));

  // check fun_var_decl wrapper
  stan::lang::fun_var_decl fvar(x);
  EXPECT_EQ(fvar.name(), "x");
  EXPECT_EQ(fvar.bare_type(), stan::lang::matrix_type());
}
