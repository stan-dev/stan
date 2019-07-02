#include <stan/lang/ast_def.cpp>

// use to check expressions
#include <stan/lang/generator/expression_visgen.hpp>
#include <stan/lang/generator/generate_array_builder_adds.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_idx.hpp>
#include <stan/lang/generator/generate_idx_user.hpp>
#include <stan/lang/generator/generate_idxs.hpp>
#include <stan/lang/generator/generate_idxs_user.hpp>
#include <stan/lang/generator/idx_user_visgen.hpp>
#include <stan/lang/generator/idx_visgen.hpp>

#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>

TEST(arrayBlockVarDecl, createVar1) {
  // 1-d array of real
  stan::lang::double_block_type dbt;
  stan::lang::block_var_type bvtDouble(dbt);
  stan::lang::expression array_len(stan::lang::int_literal(7));
  stan::lang::block_array_type bat(bvtDouble, array_len);

  stan::lang::block_var_decl bvar("x", bat);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(),
            stan::lang::bare_array_type(stan::lang::double_type()));

  EXPECT_TRUE(bvar.type().is_array_type());
  EXPECT_TRUE(bvar.type().array_element_type().bare_type().is_double_type());
  EXPECT_TRUE(bvar.type().array_contains().bare_type().is_double_type());
  EXPECT_EQ(bvar.type().array_dims(), 1);
  EXPECT_EQ(bvar.type().num_dims(), 1);

  // get var_decl component
  stan::lang::var_decl vdecl = static_cast<stan::lang::var_decl>(bvar);
  EXPECT_EQ(vdecl.name_, "x");
  EXPECT_TRUE(vdecl.bare_type().is_array_type());
  EXPECT_TRUE(is_nil(vdecl.def_));

  EXPECT_FALSE(bvar.has_def());

  EXPECT_FALSE(bvar.type().has_def_bounds());
  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar_array_lens.at(0).bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("1-dim array of real", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, bvar.bare_type());
  EXPECT_EQ("real[ ]", ss.str());

  stan::lang::block_var_decl bvd3(bvar);

  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvd3.type());
  EXPECT_EQ("1-dim array of real", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, bvd3.bare_type());
  EXPECT_EQ("real[ ]", ss.str());
}

TEST(arrayBlockVarDecl, createVar2) {
  // 1-d array of matrix
  stan::lang::double_literal real_lb(-2.0);
  stan::lang::double_literal real_ub(2.0);
  stan::lang::expression lb(real_lb);
  stan::lang::expression ub(real_ub);
  stan::lang::range m_bounds(lb, ub);
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::matrix_block_type bvtMatrix(m_bounds, M, N);
  stan::lang::expression array_len(stan::lang::int_literal(7));
  stan::lang::block_array_type bat(bvtMatrix, array_len);

  stan::lang::block_var_decl bvar("x", bat);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(),
            stan::lang::bare_array_type(stan::lang::matrix_type()));
  EXPECT_TRUE(bvar.type().is_array_type());
  EXPECT_TRUE(bvar.type().array_element_type().bare_type().is_matrix_type());
  EXPECT_TRUE(bvar.type().array_contains().bare_type().is_matrix_type());
  EXPECT_EQ(bvar.type().array_dims(), 1);
  EXPECT_EQ(bvar.type().num_dims(), 3);

  // get var_decl component
  stan::lang::var_decl vdecl = static_cast<stan::lang::var_decl>(bvar);
  EXPECT_EQ(vdecl.name_, "x");
  EXPECT_TRUE(vdecl.bare_type().is_array_type());
  EXPECT_TRUE(is_nil(vdecl.def_));

  EXPECT_TRUE(bvar.type().array_contains().has_def_bounds());
  EXPECT_TRUE(bvar.type().array_contains().bounds().has_low());
  EXPECT_TRUE(bvar.type().array_contains().bounds().has_high());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar_array_lens.at(0).bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bvar.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("1-dim array of matrix< lower, upper>", ss.str());
}
TEST(arrayBlockVarDecl, createVar3) {
  // 3-d array of matrix
  stan::lang::double_literal real_lb(-2.0);
  stan::lang::double_literal real_ub(2.0);
  stan::lang::expression lb(real_lb);
  stan::lang::expression ub(real_ub);
  stan::lang::range m_bounds(lb, ub);
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::matrix_block_type bvtMatrix(m_bounds, M, N);
  stan::lang::expression d1_array_len(stan::lang::int_literal(7));
  stan::lang::block_array_type d1(bvtMatrix, d1_array_len);
  stan::lang::expression d2_array_len(stan::lang::int_literal(8));
  stan::lang::block_array_type d2(d1, d2_array_len);
  stan::lang::expression d3_array_len(stan::lang::int_literal(9));
  stan::lang::block_array_type d3(d2, d3_array_len);
  stan::lang::block_var_decl bvar("x", d3);

  EXPECT_EQ(bvar.name(), "x");

  EXPECT_TRUE(bvar.type().is_array_type());
  EXPECT_TRUE(bvar.type().array_contains().bare_type().is_matrix_type());
  EXPECT_EQ(bvar.type().array_dims(), 3);
  EXPECT_EQ(bvar.type().num_dims(), 5);

  // get var_decl component
  stan::lang::var_decl vdecl = static_cast<stan::lang::var_decl>(bvar);
  EXPECT_EQ(vdecl.name_, "x");
  EXPECT_TRUE(vdecl.bare_type().is_array_type());
  EXPECT_TRUE(is_nil(vdecl.def_));

  EXPECT_TRUE(bvar.type().array_contains().has_def_bounds());
  EXPECT_TRUE(bvar.type().array_contains().bounds().has_low());
  EXPECT_TRUE(bvar.type().array_contains().bounds().has_high());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar_array_lens.at(0).bare_type().is_int_type());

  std::stringstream ss;
  generate_expression(bvar_array_lens.at(0), false, ss);
  EXPECT_EQ("9", ss.str());

  ss.str(std::string());
  ss.clear();
  generate_expression(bvar_array_lens.at(1), false, ss);
  EXPECT_EQ("8", ss.str());

  ss.str(std::string());
  ss.clear();
  generate_expression(bvar_array_lens.at(2), false, ss);
  EXPECT_EQ("7", ss.str());

  ss.str(std::string());
  ss.clear();
  generate_expression(bvar.type().array_contains().arg1(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  generate_expression(bvar.type().array_contains().arg2(), false, ss);
  EXPECT_EQ("4", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, bvar.bare_type());
  EXPECT_EQ("matrix[ , , ]", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("3-dim array of matrix< lower, upper>", ss.str());
}

TEST(arrayBlockVarDecl, createVar4) {
  // 1-d array of matrix
  stan::lang::double_literal real_loc(-2.0);
  stan::lang::double_literal real_multiplier(2.0);
  stan::lang::expression offset(real_loc);
  stan::lang::expression multiplier(real_multiplier);
  stan::lang::offset_multiplier m_ls(offset, multiplier);
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::matrix_block_type bvtMatrix(m_ls, M, N);
  stan::lang::expression array_len(stan::lang::int_literal(7));
  stan::lang::block_array_type bat(bvtMatrix, array_len);

  stan::lang::block_var_decl bvar("x", bat);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(),
            stan::lang::bare_array_type(stan::lang::matrix_type()));
  EXPECT_TRUE(bvar.type().is_array_type());
  EXPECT_TRUE(bvar.type().array_element_type().bare_type().is_matrix_type());
  EXPECT_TRUE(bvar.type().array_contains().bare_type().is_matrix_type());
  EXPECT_EQ(bvar.type().array_dims(), 1);
  EXPECT_EQ(bvar.type().num_dims(), 3);

  // get var_decl component
  stan::lang::var_decl vdecl = static_cast<stan::lang::var_decl>(bvar);
  EXPECT_EQ(vdecl.name_, "x");
  EXPECT_TRUE(vdecl.bare_type().is_array_type());
  EXPECT_TRUE(is_nil(vdecl.def_));

  EXPECT_TRUE(bvar.type().array_contains().has_def_offset_multiplier());
  EXPECT_TRUE(bvar.type().array_contains().ls().has_offset());
  EXPECT_TRUE(bvar.type().array_contains().ls().has_multiplier());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar_array_lens.at(0).bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_bare_expr_type(ss, bvar.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("1-dim array of matrix< offset, multiplier>", ss.str());
}

TEST(choleskyCorrBlockVarDecl, createVar1) {
  stan::lang::int_literal int_len(5);
  stan::lang::expression K(int_len);
  stan::lang::cholesky_factor_corr_block_type ccbt(K);

  stan::lang::block_var_decl bvar("x", ccbt);

  EXPECT_EQ(bvar.name_, "x");
  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::matrix_type());
  EXPECT_TRUE(is_nil(bvar.def_));
  EXPECT_FALSE(bvar.has_def());
  EXPECT_FALSE(bvar.type().has_def_bounds());
  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());
  EXPECT_TRUE(bvar.type().arg2().bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("cholesky_factor_corr", ss.str());
}

TEST(choleskyFactorBlockVarDecl, createVar1) {
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::cholesky_factor_cov_block_type cfbt(M, N);

  stan::lang::block_var_decl bvar("x", cfbt);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_TRUE(bvar.bare_type().is_matrix_type());
  EXPECT_TRUE(is_nil(bvar.def()));
  EXPECT_FALSE(bvar.has_def());
  EXPECT_FALSE(bvar.type().has_def_bounds());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());
  EXPECT_TRUE(bvar.type().arg2().bare_type().is_int_type());

  // stan::lang::cholesky_factor_cov_block_type cfbt2
  //   =
  //   boost::get<stan::lang::cholesky_factor_cov_block_type>(bvar.type().var_type_);
  // stan::lang::expression m_size = cfbt2.M();
  // EXPECT_TRUE(m_size.bare_type().is_int_type());
  // stan::lang::expression n_size = cfbt2.N();
  // EXPECT_TRUE(m_size.bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("cholesky_factor_cov", ss.str());
}

TEST(corrMatrixBlockVarDecl, createVar1) {
  stan::lang::int_literal int_len(5);
  stan::lang::expression K(int_len);
  stan::lang::corr_matrix_block_type cmbt(K);

  stan::lang::block_var_decl bvar("x", cmbt);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_TRUE(bvar.bare_type().is_matrix_type());
  EXPECT_FALSE(bvar.type().has_def_bounds());
  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());
  EXPECT_TRUE(is_nil(bvar.def()));
  EXPECT_FALSE(bvar.has_def());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());

  //  // check corr_matrix_block_var_type
  // stan::lang::corr_matrix_block_type cmbt2
  //   = boost::get<stan::lang::corr_matrix_block_type>(bvar.type().var_type_);
  // stan::lang::expression k_size = cmbt2.K();
  // EXPECT_TRUE(k_size.bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("corr_matrix", ss.str());
}

TEST(covMatrixBlockVarDecl, createVar1) {
  stan::lang::int_literal int_len(5);
  stan::lang::expression K(int_len);
  stan::lang::cov_matrix_block_type cmbt(K);

  stan::lang::block_var_decl bvar("x", cmbt);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_TRUE(bvar.bare_type().is_matrix_type());
  EXPECT_FALSE(bvar.type().has_def_bounds());
  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());
  EXPECT_TRUE(is_nil(bvar.def()));
  EXPECT_FALSE(bvar.has_def());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());

  //  // check cov_matrix_block_var_type
  // stan::lang::cov_matrix_block_type cmbt2
  //   = boost::get<stan::lang::cov_matrix_block_type>(bvar.type().var_type_);
  // stan::lang::expression k_size = cmbt2.K();
  // EXPECT_TRUE(k_size.bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("cov_matrix", ss.str());
}

TEST(doubleBlockVarDecl, createVar1) {
  stan::lang::double_literal real_lb(-2.0);
  stan::lang::double_literal real_ub(2.0);
  stan::lang::expression lb(real_lb);
  stan::lang::expression ub(real_ub);
  stan::lang::range v_bounds(lb, ub);
  stan::lang::double_block_type dbt(v_bounds);

  stan::lang::block_var_decl bvar("x", dbt);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::double_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_TRUE(bvar.type().bounds().has_low());
  EXPECT_TRUE(bvar.type().bounds().has_high());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("real< lower, upper>", ss.str());
}

TEST(intBlockVarDecl, createVar1) {
  stan::lang::range v_bounds(stan::lang::int_literal(-1),
                             stan::lang::int_literal(1));
  stan::lang::int_block_type ibt(v_bounds);

  stan::lang::block_var_decl bvar("x", ibt);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::int_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_TRUE(bvar.type().bounds().has_low());
  EXPECT_TRUE(bvar.type().bounds().has_high());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("int< lower, upper>", ss.str());
}

TEST(matrixBlockVarDecl, createVar1) {
  stan::lang::double_literal real_lb(-2.0);
  stan::lang::double_literal real_ub(2.0);
  stan::lang::expression lb(real_lb);
  stan::lang::expression ub(real_ub);
  stan::lang::range m_bounds(lb, ub);

  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::matrix_block_type tMatrix(m_bounds, M, N);

  stan::lang::block_var_decl bvar("x", tMatrix);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::matrix_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_TRUE(bvar.type().bounds().has_low());
  EXPECT_TRUE(bvar.type().bounds().has_high());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());
  EXPECT_TRUE(bvar.type().arg2().bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("matrix< lower, upper>", ss.str());
}

TEST(matrixBlockVarDecl, unbounded) {
  stan::lang::range m_bounds;
  stan::lang::expression M(stan::lang::int_literal(3));
  stan::lang::expression N(stan::lang::int_literal(4));
  stan::lang::matrix_block_type tMatrix(m_bounds, M, N);

  stan::lang::block_var_decl bvar("x", tMatrix);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::matrix_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());
  EXPECT_TRUE(bvar.type().arg2().bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(orderedBlockVarDecl, createVar1) {
  stan::lang::int_literal int_len(5);
  stan::lang::expression K(int_len);
  stan::lang::ordered_block_type tOrd(K);

  stan::lang::block_var_decl bvar("x", tOrd);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::vector_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());
  EXPECT_FALSE(bvar.type().has_def_bounds());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("ordered", ss.str());
}

TEST(positiveOrderedBlockVarDecl, createVar1) {
  stan::lang::int_literal int_len(5);
  stan::lang::expression K(int_len);
  stan::lang::positive_ordered_block_type tPosOrd(K);

  stan::lang::block_var_decl bvar("x", tPosOrd);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::vector_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());
  EXPECT_FALSE(bvar.type().has_def_bounds());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("positive_ordered", ss.str());
}

TEST(rowVectorBlockVarDecl, createVar1) {
  stan::lang::double_literal real_lb(-2.0);
  stan::lang::double_literal real_ub(2.0);
  stan::lang::expression lb(real_lb);
  stan::lang::expression ub(real_ub);
  stan::lang::range vec_bounds(lb, ub);
  stan::lang::int_literal int_len(5);
  stan::lang::expression vec_len(int_len);
  stan::lang::row_vector_block_type tRowVec(vec_bounds, vec_len);

  stan::lang::block_var_decl bvar("x", tRowVec);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::row_vector_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_TRUE(bvar.type().bounds().has_low());
  EXPECT_TRUE(bvar.type().bounds().has_high());
  EXPECT_TRUE(bvar.type().has_def_bounds());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("row_vector< lower, upper>", ss.str());
}

TEST(simplexBlockVarDecl, createVar1) {
  stan::lang::int_literal int_len(5);
  stan::lang::expression K(int_len);
  stan::lang::simplex_block_type sbt(K);

  stan::lang::block_var_decl bvar("x", sbt);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::vector_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());
  EXPECT_FALSE(bvar.type().has_def_bounds());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("simplex", ss.str());
}

TEST(unitVectorBlockVarDecl, createVar1) {
  stan::lang::int_literal int_len(5);
  stan::lang::expression K(int_len);
  stan::lang::unit_vector_block_type uvbt(K);

  stan::lang::block_var_decl bvar("x", uvbt);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::vector_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_FALSE(bvar.type().bounds().has_low());
  EXPECT_FALSE(bvar.type().bounds().has_high());
  EXPECT_FALSE(bvar.type().has_def_bounds());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("unit_vector", ss.str());
}

TEST(vectorBlockVarDecl, createVar1) {
  stan::lang::double_literal real_lb(-2.0);
  stan::lang::double_literal real_ub(2.0);
  stan::lang::expression lb(real_lb);
  stan::lang::expression ub(real_ub);
  stan::lang::range vec_bounds(lb, ub);
  stan::lang::int_literal int_len(5);
  stan::lang::expression vec_len(int_len);
  stan::lang::vector_block_type tVec(vec_bounds, vec_len);

  stan::lang::block_var_decl bvar("x", tVec);

  EXPECT_EQ(bvar.name(), "x");
  EXPECT_EQ(bvar.bare_type(), stan::lang::vector_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_TRUE(bvar.type().bounds().has_low());
  EXPECT_TRUE(bvar.type().bounds().has_high());
  EXPECT_TRUE(bvar.type().has_def_bounds());

  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), bvar.type().array_dims());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_int_type());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvar.type());
  EXPECT_EQ("vector< lower, upper>", ss.str());
}

TEST(illFormedBlockVarDecl, createVar1) {
  stan::lang::block_var_decl bvar;

  EXPECT_EQ(bvar.name(), "");
  EXPECT_TRUE(bvar.type().bare_type().is_ill_formed_type());
  EXPECT_EQ(bvar.bare_type(), stan::lang::ill_formed_type());
  EXPECT_FALSE(bvar.has_def());
  EXPECT_TRUE(bvar.type().arg1().bare_type().is_ill_formed_type());
  EXPECT_TRUE(bvar.type().arg2().bare_type().is_ill_formed_type());
  std::vector<stan::lang::expression> bvar_array_lens =
      bvar.type().array_lens();
  EXPECT_EQ(bvar_array_lens.size(), 0);
  EXPECT_EQ(bvar.type().num_dims(), 0);
  EXPECT_EQ(bvar.type().array_dims(), 0);
}

// from old src/test/unit/lang/ast_test.cpp
TEST(langAst, baseVarDecl) {
  std::vector<stan::lang::expression> dims;
  dims.push_back(stan::lang::expression(stan::lang::int_literal(0)));
  // 1d array of int
  stan::lang::int_block_type btInt;
  stan::lang::block_array_type bat_int_1d(btInt, dims);
  stan::lang::block_var_decl bvd("foo", bat_int_1d);
  EXPECT_EQ("foo", bvd.name());
  EXPECT_EQ(1, bvd.bare_type().num_dims());
  EXPECT_EQ(stan::lang::expression(stan::lang::int_literal(0)).bare_type(),
            bvd.type().array_lens().at(0).bare_type());
  EXPECT_EQ(stan::lang::bare_expr_type(stan::lang::int_type()),
            bvd.bare_type().innermost_type());
}
