#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

TEST(matrixLocalVarDecl, createVar1) {
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::matrix_local_type tMatrix(M, N);
  stan::lang::matrix_local_var_decl x("x", tMatrix);

  // check matrix_local_var_decl
  EXPECT_EQ(x.name_, "x");
  EXPECT_TRUE(x.bare_type_.is_matrix_type());
  EXPECT_TRUE(is_nil(x.def_));

  // check matrix_local_var_type
  stan::lang::expression m_size = x.type_.M_;
  EXPECT_TRUE(m_size.bare_type().is_int_type());
  stan::lang::expression n_size = x.type_.N_;
  EXPECT_TRUE(m_size.bare_type().is_int_type());

  // check local_var_decl wrapper
  stan::lang::local_var_decl lvar(x);
  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::matrix_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 2);
  EXPECT_TRUE(lvar_sizes.at(0).bare_type().is_int_type());
  EXPECT_TRUE(lvar_sizes.at(1).bare_type().is_int_type());
}
