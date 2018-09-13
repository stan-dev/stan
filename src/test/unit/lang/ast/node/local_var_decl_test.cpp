#include <stan/lang/ast_def.cpp>

// use to check expressions
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

  std::vector<stan::lang::expression> lvar_array_lens = lvar.type().array_lens();
  EXPECT_EQ(lvar_array_lens.size(), 1);
  EXPECT_TRUE(lvar_array_lens.at(0).bare_type().is_int_type());

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
  EXPECT_TRUE(lvar.type().array_contains().arg1().bare_type().is_int_type());
  EXPECT_TRUE(lvar.type().array_contains().arg2().bare_type().is_int_type());

  std::vector<stan::lang::expression> lvar_array_lens = lvar.type().array_lens();
  EXPECT_EQ(lvar_array_lens.size(), 1);
  EXPECT_TRUE(lvar_array_lens.at(0).bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, lvar.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

TEST(arrayLocalVarDecl, createVar3) {
  // 3-d array of matrix
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::matrix_local_type lvtMatrix(M, N);
  stan::lang::expression d1_array_len(stan::lang::int_literal(7));
  stan::lang::local_array_type d1(lvtMatrix, d1_array_len);
  stan::lang::expression d2_array_len(stan::lang::int_literal(8));
  stan::lang::local_array_type d2(d1, d2_array_len);
  stan::lang::expression d3_array_len(stan::lang::int_literal(9));
  stan::lang::local_array_type d3(d2, d3_array_len);
  stan::lang::local_var_decl lvar("x", d3);
  
  EXPECT_EQ(lvar.name(), "x");

  EXPECT_TRUE(lvar.type().is_array_type());
  EXPECT_TRUE(lvar.type().array_contains().bare_type().is_matrix_type());
  EXPECT_EQ(lvar.type().array_dims(), 3);
  EXPECT_EQ(lvar.type().num_dims(), 5);


  // get var_decl component
  stan::lang::var_decl vdecl = static_cast<stan::lang::var_decl>(lvar);
  EXPECT_EQ(vdecl.name_, "x");
  EXPECT_TRUE(vdecl.bare_type_.is_array_type());
  EXPECT_TRUE(is_nil(vdecl.def_));

  std::vector<stan::lang::expression> lvar_array_lens = lvar.type().array_lens();
  EXPECT_EQ(lvar_array_lens.size(), lvar.type().array_dims());
  EXPECT_TRUE(lvar_array_lens.at(0).bare_type().is_int_type());

  std::stringstream ss;
  generate_expression(lvar_array_lens.at(0), false, ss);
  EXPECT_EQ("9", ss.str());

  ss.str(std::string());
  ss.clear();
  generate_expression(lvar_array_lens.at(1), false, ss);
  EXPECT_EQ("8", ss.str());

  ss.str(std::string());
  ss.clear();
  generate_expression(lvar_array_lens.at(2), false, ss);
  EXPECT_EQ("7", ss.str());

  ss.str(std::string());
  ss.clear();
  generate_expression(lvar.type().array_contains().arg1(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  generate_expression(lvar.type().array_contains().arg2(), false, ss);
  EXPECT_EQ("4", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, lvar.bare_type());
  EXPECT_EQ("matrix[ , , ]", ss.str());
}

TEST(doubleLocalVarDecl, createVar1) {
  stan::lang::double_type dbt;
  stan::lang::local_var_type lvtDouble(dbt);
  stan::lang::local_var_decl lvar("x",lvtDouble);

  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::double_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_array_lens = lvar.type().array_lens();
  EXPECT_EQ(lvar_array_lens.size(), lvar.type().array_dims());
}

TEST(intLocalVarDecl, createVar1) {
  stan::lang::int_type dbt;
  stan::lang::local_var_type lvtInt(dbt);
  stan::lang::local_var_decl lvar("x",lvtInt);

  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::int_type());
  EXPECT_FALSE(lvar.has_def());

  std::vector<stan::lang::expression> lvar_array_lens = lvar.type().array_lens();
  EXPECT_EQ(lvar_array_lens.size(), lvar.type().array_dims());
}

TEST(matrixLocalVarDecl, createVar1) {
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::matrix_local_type tMatrix(M, N);
  stan::lang::local_var_decl lvar("x", tMatrix);

  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::matrix_type());
  EXPECT_FALSE(lvar.has_def());

  EXPECT_TRUE(lvar.type().arg1().bare_type().is_int_type());

  std::stringstream ss;
  generate_expression(lvar.type().arg1(), false, ss);
  EXPECT_EQ("3", ss.str());

  EXPECT_TRUE(lvar.type().arg2().bare_type().is_int_type());

  ss.str(std::string());
  ss.clear();
  generate_expression(lvar.type().arg2(), false, ss);
  EXPECT_EQ("4", ss.str());

  std::vector<stan::lang::expression> lvar_array_lens = lvar.type().array_lens();
  EXPECT_EQ(lvar_array_lens.size(), lvar.type().array_dims());
}

TEST(rowVectorLocalVarDecl, createVar1) {
  stan::lang::expression N(stan::lang::int_literal(11));
  stan::lang::row_vector_local_type tRowVector(N);
  stan::lang::local_var_decl lvar("x", tRowVector);

  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::row_vector_type());
  EXPECT_FALSE(lvar.has_def());

  EXPECT_TRUE(lvar.type().arg1().bare_type().is_int_type());
  std::stringstream ss;
  generate_expression(lvar.type().arg1(), false, ss);
  EXPECT_EQ("11", ss.str());

  EXPECT_TRUE(lvar.type().arg2().bare_type().is_ill_formed_type());

  std::vector<stan::lang::expression> lvar_array_lens = lvar.type().array_lens();
  EXPECT_EQ(lvar_array_lens.size(), lvar.type().array_dims());
}

TEST(vectorLocalVarDecl, createVar1) {
  stan::lang::expression N(stan::lang::int_literal(11));
  stan::lang::vector_local_type tVector(N);
  stan::lang::local_var_decl lvar("x", tVector);

  EXPECT_EQ(lvar.name(), "x");
  EXPECT_EQ(lvar.bare_type(), stan::lang::vector_type());
  EXPECT_FALSE(lvar.has_def());

  EXPECT_TRUE(lvar.type().arg1().bare_type().is_int_type());
  std::stringstream ss;
  generate_expression(lvar.type().arg1(), false, ss);
  EXPECT_EQ("11", ss.str());

  EXPECT_TRUE(lvar.type().arg2().bare_type().is_ill_formed_type());

  std::vector<stan::lang::expression> lvar_array_lens = lvar.type().array_lens();
  EXPECT_EQ(lvar_array_lens.size(), lvar.type().array_dims());
}

TEST(illFormedLocalVarDecl, createVar1) {
  stan::lang::local_var_decl lvar;

  EXPECT_EQ(lvar.name(), "");
  EXPECT_TRUE(lvar.type().bare_type().is_ill_formed_type());
  EXPECT_EQ(lvar.bare_type(), stan::lang::ill_formed_type());
  EXPECT_FALSE(lvar.has_def());
  EXPECT_TRUE(lvar.type().arg1().bare_type().is_ill_formed_type());
  EXPECT_TRUE(lvar.type().arg2().bare_type().is_ill_formed_type());
  std::vector<stan::lang::expression> lvar_array_lens = lvar.type().array_lens();
  EXPECT_EQ(lvar_array_lens.size(), 0);
  EXPECT_EQ(lvar.type().num_dims(), 0);
  EXPECT_EQ(lvar.type().array_dims(), 0);
}
