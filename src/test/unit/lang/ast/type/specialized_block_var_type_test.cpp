#include <stan/lang/ast_def.cpp>

// using these to check row/col args for specialized types
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
#include <iostream>

using stan::lang::block_var_type;
using stan::lang::block_array_type;
using stan::lang::cholesky_factor_corr_block_type;
using stan::lang::cholesky_factor_cov_block_type;
using stan::lang::corr_matrix_block_type;
using stan::lang::cov_matrix_block_type;
using stan::lang::double_block_type;
using stan::lang::ill_formed_type;
using stan::lang::int_block_type;
using stan::lang::matrix_block_type;
using stan::lang::ordered_block_type;
using stan::lang::positive_ordered_block_type;
using stan::lang::row_vector_block_type;
using stan::lang::simplex_block_type;
using stan::lang::unit_vector_block_type;
using stan::lang::vector_block_type;

using stan::lang::expression;
using stan::lang::int_literal;
using stan::lang::double_literal;
using stan::lang::range;
using stan::lang::write_bare_expr_type;
using stan::lang::write_block_var_type;

TEST(blockVarType, createCholeskyFactorCovDefault) {
  cholesky_factor_cov_block_type tCFCov;
  block_var_type x(tCFCov);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.num_dims(), 2);
  EXPECT_TRUE(x.arg1().bare_type().is_ill_formed_type());
  EXPECT_TRUE(x.arg2().bare_type().is_ill_formed_type());
  
  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  std::stringstream ss;
  write_block_var_type(ss, x);
  EXPECT_EQ("cholesky_factor_cov", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createCholeskyFactorCovSquare) {
  expression N(int_literal(4));
  cholesky_factor_cov_block_type tCFCov(N, N);
  block_var_type x(tCFCov);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.num_dims(), 2);
  EXPECT_TRUE(x.arg1().bare_type().is_int_type());
  EXPECT_TRUE(x.arg2().bare_type().is_int_type());
  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  std::stringstream ss;

  generate_expression(x.arg1(), false, ss);
  EXPECT_EQ("4", ss.str());
  ss.str(std::string());

  ss.clear();
  generate_expression(x.arg2(), false, ss);
  EXPECT_EQ("4", ss.str());
  ss.str(std::string());
  ss.clear();

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("cholesky_factor_cov", ss.str());

  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createCholeskyFactorCovRect) {
  expression M(int_literal(3));
  expression N(int_literal(4));
  cholesky_factor_cov_block_type tCFCov(M, N);
  block_var_type x(tCFCov);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.num_dims(), 2);
  EXPECT_TRUE(x.arg1().bare_type().is_int_type());
  EXPECT_TRUE(x.arg2().bare_type().is_int_type());

  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());
  std::stringstream ss;

  generate_expression(x.arg1(), false, ss);
  EXPECT_EQ("3", ss.str());
  ss.str(std::string());

  ss.clear();
  generate_expression(x.arg2(), false, ss);
  EXPECT_EQ("4", ss.str());
  ss.str(std::string());
  ss.clear();

  write_block_var_type(ss, x);
  EXPECT_EQ("cholesky_factor_cov", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createCholeskyFactorCorr) {
  expression K(int_literal(3));
  cholesky_factor_corr_block_type tCFCorr(K);
  block_var_type x(tCFCorr);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.num_dims(), 2);
  EXPECT_TRUE(x.arg1().bare_type().is_int_type());
  EXPECT_TRUE(x.arg2().bare_type().is_int_type());

  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  std::stringstream ss;

  generate_expression(x.arg1(), false, ss);
  EXPECT_EQ("3", ss.str());
  ss.str(std::string());

  ss.clear();
  generate_expression(x.arg2(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("cholesky_factor_corr", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createArrayCFCorr) {
  expression K(int_literal(3));
  cholesky_factor_corr_block_type tCFCorr(K);
  block_array_type d1(tCFCorr,K);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 3);
  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  EXPECT_TRUE(x.arg1().bare_type().is_ill_formed_type());
  EXPECT_TRUE(x.arg2().bare_type().is_ill_formed_type());

  block_var_type y = x.array_contains();
  EXPECT_TRUE(y.arg1().bare_type().is_int_type());
  EXPECT_TRUE(y.arg2().bare_type().is_int_type());

  std::stringstream ss;

  generate_expression(y.arg1(), false, ss);
  EXPECT_EQ("3", ss.str());
  ss.str(std::string());

  ss.clear();
  generate_expression(y.arg2(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of cholesky_factor_corr", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

TEST(blockVarType, createCorrMatrix) {
  expression K(int_literal(3));
  corr_matrix_block_type tCorrMat(K);
  block_var_type x(tCorrMat);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.num_dims(), 2);

  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  EXPECT_TRUE(x.arg1().bare_type().is_int_type());
  EXPECT_TRUE(x.arg2().bare_type().is_int_type());

  std::stringstream ss;

  generate_expression(x.arg1(), false, ss);
  EXPECT_EQ("3", ss.str());
  ss.str(std::string());

  ss.clear();
  generate_expression(x.arg2(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("corr_matrix", ss.str());

  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createArrayCorrMat) {
  expression K(int_literal(3));
  corr_matrix_block_type tCorrMat(K);
  block_array_type d1(tCorrMat,K);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 3);
  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  std::stringstream ss;

  generate_expression(x.array_contains().arg1(), false, ss);
  EXPECT_EQ("3", ss.str());
  ss.str(std::string());

  ss.clear();
  generate_expression(x.array_contains().arg2(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of corr_matrix", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

TEST(blockVarType, createCovMatrix) {
  expression K(int_literal(3));
  cov_matrix_block_type tCovMat(K);
  block_var_type x(tCovMat);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.num_dims(), 2);

  EXPECT_TRUE(x.arg1().bare_type().is_int_type());
  EXPECT_TRUE(x.arg2().bare_type().is_int_type());

  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  std::stringstream ss;

  generate_expression(x.arg1(), false, ss);
  EXPECT_EQ("3", ss.str());
  ss.str(std::string());

  ss.clear();
  generate_expression(x.arg2(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("cov_matrix", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createArrayCovMat) {
  expression K(int_literal(3));
  cov_matrix_block_type tCovMat(K);
  block_array_type d1(tCovMat,K);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 3);
  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());
  
  EXPECT_TRUE(x.arg1().bare_type().is_ill_formed_type());
  EXPECT_TRUE(x.arg2().bare_type().is_ill_formed_type());

  block_var_type y = x.array_contains();
  EXPECT_TRUE(y.arg1().bare_type().is_int_type());
  EXPECT_TRUE(y.arg2().bare_type().is_int_type());

  std::stringstream ss;

  generate_expression(y.arg1(), false, ss);
  EXPECT_EQ("3", ss.str());
  ss.str(std::string());

  ss.clear();
  generate_expression(y.arg2(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of cov_matrix", ss.str());

  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

TEST(blockVarType, createOrdered) {
  expression K(int_literal(3));
  ordered_block_type tOrd(K);
  block_var_type x(tOrd);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  EXPECT_TRUE(x.arg1().bare_type().is_int_type());
  EXPECT_TRUE(x.arg2().bare_type().is_ill_formed_type());

  std::stringstream ss;

  generate_expression(x.arg1(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  generate_expression(x.arg2(), false, ss);
  EXPECT_EQ("nil", ss.str());

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("ordered", ss.str());

  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(blockVarType, createArrayOrd) {
  expression K(int_literal(3));
  ordered_block_type tOrd(K);
  block_array_type d1(tOrd,K);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 2);
  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  std::stringstream ss;
  write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of ordered", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector[ ]", ss.str());
}

TEST(blockVarType, createPosOrdered) {
  expression K(int_literal(3));
  positive_ordered_block_type tPosOrd(K);
  block_var_type x(tPosOrd);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  EXPECT_TRUE(x.arg1().bare_type().is_int_type());
  EXPECT_TRUE(x.arg2().bare_type().is_ill_formed_type());

  std::stringstream ss;
  generate_expression(x.arg1(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("positive_ordered", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(blockVarType, createArrayPosOrd) {
  expression K(int_literal(3));
  positive_ordered_block_type tPosOrd(K);
  block_array_type d1(tPosOrd,K);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 2);
  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  std::stringstream ss;
  write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of positive_ordered", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector[ ]", ss.str());
}

TEST(blockVarType, createSimplex) {
  expression K(int_literal(3));
  simplex_block_type tSimplex(K);
  block_var_type x(tSimplex);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  EXPECT_TRUE(x.arg1().bare_type().is_int_type());
  EXPECT_TRUE(x.arg2().bare_type().is_ill_formed_type());

  std::stringstream ss;
  generate_expression(x.arg1(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("simplex", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(blockVarType, createArraySimplex) {
  expression K(int_literal(3));
  simplex_block_type tSimplex(K);
  block_array_type d1(tSimplex,K);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 2);
  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  std::stringstream ss;
  write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of simplex", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector[ ]", ss.str());
}


TEST(blockVarType, createUnitVec) {
  expression K(int_literal(3));
  unit_vector_block_type tUnitVec(K);
  block_var_type x(tUnitVec);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  EXPECT_TRUE(x.arg1().bare_type().is_int_type());
  EXPECT_TRUE(x.arg2().bare_type().is_ill_formed_type());

  std::stringstream ss;
  generate_expression(x.arg1(), false, ss);
  EXPECT_EQ("3", ss.str());

  ss.str(std::string());
  ss.clear();
  write_block_var_type(ss, x);
  EXPECT_EQ("unit_vector", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(blockVarType, createArrayUnitVec) {
  expression K(int_literal(3));
  unit_vector_block_type tUnitVec(K);
  block_array_type d1(tUnitVec,K);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_TRUE(x.is_specialized());
  EXPECT_TRUE(x.is_constrained());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 2);
  std::vector<expression> array_lens = x.array_lens();
  EXPECT_EQ(array_lens.size(), x.array_dims());

  std::stringstream ss;
  write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of unit_vector", ss.str());
  ss.str(std::string());
  ss.clear();
  write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector[ ]", ss.str());
}
