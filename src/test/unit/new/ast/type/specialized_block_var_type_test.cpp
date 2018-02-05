#include <stan/lang/ast_def.cpp>
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
using stan::lang::range;

TEST(blockVarType, createCholeskyFactorCovSquare) {
  expression e1;
  cholesky_factor_cov_block_type tCFCov(e1, e1);
  block_var_type x(tCFCov);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("cholesky_factor_cov", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createCholeskyFactorCovRect) {
  expression e1;
  expression e2;
  cholesky_factor_cov_block_type tCFCov(e1, e2);
  block_var_type x(tCFCov);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("cholesky_factor_cov", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createArrayCFCov) {
  expression e1;
  cholesky_factor_cov_block_type tCFCov(e1, e1);
  block_array_type d1(tCFCov,e1);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 3);
  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 3);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of cholesky_factor_cov", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

TEST(blockVarType, createCholeskyFactorCorr) {
  expression e1;
  cholesky_factor_corr_block_type tCFCorr(e1);
  block_var_type x(tCFCorr);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("cholesky_factor_corr", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createArrayCFCorr) {
  expression e1;
  cholesky_factor_corr_block_type tCFCorr(e1);
  block_array_type d1(tCFCorr,e1);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 3);
  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 3);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of cholesky_factor_corr", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

TEST(blockVarType, createCorrMatrix) {
  expression e1;
  corr_matrix_block_type tCorrMat(e1);
  block_var_type x(tCorrMat);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("corr_matrix", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createArrayCorrMat) {
  expression e1;
  corr_matrix_block_type tCorrMat(e1);
  block_array_type d1(tCorrMat,e1);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 3);
  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 3);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of corr_matrix", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

TEST(blockVarType, createCovMatrix) {
  expression e1;
  cov_matrix_block_type tCovMat(e1);
  block_var_type x(tCovMat);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 2);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("cov_matrix", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix", ss.str());
}

TEST(blockVarType, createArrayCovMat) {
  expression e1;
  cov_matrix_block_type tCovMat(e1);
  block_array_type d1(tCovMat,e1);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 3);
  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 3);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of cov_matrix", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("matrix[ ]", ss.str());
}

TEST(blockVarType, createOrdered) {
  expression e1;
  ordered_block_type tOrd(e1);
  block_var_type x(tOrd);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("ordered", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(blockVarType, createArrayOrd) {
  expression e1;
  ordered_block_type tOrd(e1);
  block_array_type d1(tOrd,e1);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 2);
  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of ordered", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector[ ]", ss.str());
}

TEST(blockVarType, createPosOrdered) {
  expression e1;
  positive_ordered_block_type tPosOrd(e1);
  block_var_type x(tPosOrd);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("positive_ordered", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(blockVarType, createArrayPosOrd) {
  expression e1;
  positive_ordered_block_type tPosOrd(e1);
  block_array_type d1(tPosOrd,e1);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 2);
  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of positive_ordered", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector[ ]", ss.str());
}

TEST(blockVarType, createSimplex) {
  expression e1;
  simplex_block_type tSimplex(e1);
  block_var_type x(tSimplex);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("simplex", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(blockVarType, createArraySimplex) {
  expression e1;
  simplex_block_type tSimplex(e1);
  block_array_type d1(tSimplex,e1);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 2);
  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of simplex", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector[ ]", ss.str());
}


TEST(blockVarType, createUnitVec) {
  unit_vector_block_type tUnitVec;
  block_var_type x(tUnitVec);
  EXPECT_FALSE(x.is_array_type());
  EXPECT_EQ(x.num_dims(), 1);

  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 1);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("unit_vector", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector", ss.str());
}

TEST(blockVarType, createArrayUnitVec) {
  expression e1;
  unit_vector_block_type tUnitVec(e1);
  block_array_type d1(tUnitVec,e1);
  block_var_type x(d1);
  EXPECT_TRUE(x.is_array_type());
  EXPECT_EQ(x.array_dims(), 1);
  EXPECT_EQ(x.num_dims(), 2);
  std::vector<expression> sizes = x.size();
  EXPECT_EQ(sizes.size(), 2);

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, x);
  EXPECT_EQ("1-dim array of unit_vector", ss.str());
  ss.str(std::string());
  ss.clear();
  stan::lang::write_bare_expr_type(ss, x.bare_type());
  EXPECT_EQ("vector[ ]", ss.str());
}
