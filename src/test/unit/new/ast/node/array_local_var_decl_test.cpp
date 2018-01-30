#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

TEST(arrayLocalVarDecl, createVar1) {
  // 1-d array of real
  stan::lang::double_type dbt;
  stan::lang::local_var_type lvtDouble(dbt);
  stan::lang::expression array_len(stan::lang::int_literal(7));
  stan::lang::array_local_var_decl x("x", lvtDouble, array_len);

  // check local_array_type
  EXPECT_EQ(x.type_.dims(), 1);
  EXPECT_TRUE(x.type_.array_len_.bare_type().is_int_type());
  EXPECT_TRUE(x.type_.element_type_.bare_type().is_double_type());
  
  // // check local_var_decl wrapper
  stan::lang::local_var_decl lvar(x);
  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::bare_array_type(stan::lang::double_type()));
  EXPECT_FALSE(lvar.has_def());

  EXPECT_TRUE(lvar.type().is_array_type());
  EXPECT_TRUE(lvar.type().array_contains().bare_type().is_double_type());

  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 1);
  EXPECT_TRUE(lvar_sizes.at(0).bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, lvar.bare_type());
  EXPECT_EQ("real[ ]", ss.str());
}

TEST(arrayLocalVarDecl, createVar2) {
  // 1-d array of matrix
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));

  stan::lang::matrix_local_type lvtMatrix(M, N);
  stan::lang::expression array_len(stan::lang::int_literal(7));
  stan::lang::array_local_var_decl x("x", lvtMatrix, array_len);

  // check local_array_type
  EXPECT_TRUE(x.type_.element_type_.bare_type().is_matrix_type());
  EXPECT_EQ(x.type_.dims(), 1);
  EXPECT_TRUE(x.type_.array_len_.bare_type().is_int_type());
  
  // // check local_var_decl wrapper
  stan::lang::local_var_decl lvar(x);
  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::bare_array_type(stan::lang::matrix_type()));
  EXPECT_FALSE(lvar.has_def());

  EXPECT_TRUE(lvar.type().is_array_type());
  EXPECT_TRUE(lvar.type().array_contains().bare_type().is_matrix_type());

  EXPECT_EQ(lvar.type().num_dims(), 3);
  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 3);
  EXPECT_TRUE(lvar_sizes.at(0).bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, lvar.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

