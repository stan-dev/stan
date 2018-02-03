#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

TEST(arrayLocalVarDecl, createVar1) {
  // 1-d array of real
  stan::lang::double_type dbt;
  stan::lang::local_var_type lvtDouble(dbt);
  stan::lang::expression array_len(stan::lang::int_literal(7));
  stan::lang::local_array_type lat(lvtDouble, array_len);

  stan::lang::local_var_decl lvar("x",lat);

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
  stan::lang::local_array_type lat(lvtMatrix, array_len);

  stan::lang::local_var_decl lvar("x",lat);

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

TEST(doubleLocalVarDecl, createVar1) {
  stan::lang::double_type dbt;
  stan::lang::local_var_type lvtDouble(dbt);
  stan::lang::local_var_decl lvar("x",lvtDouble);

  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::double_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 0);
}

TEST(intLocalVarDecl, createVar1) {
  stan::lang::int_type dbt;
  stan::lang::local_var_type lvtInt(dbt);
  stan::lang::local_var_decl lvar("x",lvtInt);

  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::int_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 0);
}

TEST(matrixLocalVarDecl, createVar1) {
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::matrix_local_type tMatrix(M, N);
  stan::lang::local_var_decl lvar("x", tMatrix);

  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::matrix_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 2);
  EXPECT_TRUE(lvar_sizes.at(0).bare_type().is_int_type());
  EXPECT_TRUE(lvar_sizes.at(1).bare_type().is_int_type());
}

TEST(rowVectorLocalVarDecl, createVar1) {
  stan::lang::expression N(stan::lang::int_literal(1));
  stan::lang::row_vector_local_type tRowVector(N);
  stan::lang::local_var_decl lvar("x", tRowVector);

  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::row_vector_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 1);
  EXPECT_TRUE(lvar_sizes.at(0).bare_type().is_int_type());
}

TEST(vectorLocalVarDecl, createVar1) {
  stan::lang::expression N(stan::lang::int_literal(1));
  stan::lang::vector_local_type tVector(N);
  stan::lang::local_var_decl lvar("x", tVector);

  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::vector_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_sizes = lvar.type().size();
  EXPECT_EQ(lvar_sizes.size(), 1);
  EXPECT_TRUE(lvar_sizes.at(0).bare_type().is_int_type());
}

TEST(illFormedLocalVarDecl, createVar1) {
  stan::lang::local_var_decl lvar;

  EXPECT_EQ(lvar.name(), "");
  EXPECT_TRUE(lvar.type().bare_type().is_ill_formed_type());
  EXPECT_EQ(lvar.bare_type(), stan::lang::ill_formed_type());
  EXPECT_FALSE(lvar.has_def());
}
