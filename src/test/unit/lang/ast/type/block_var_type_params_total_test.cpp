#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <iostream>

// needed to test params_total method
#include <stan/lang/generator/expression_visgen.hpp>
#include <stan/lang/generator/generate_array_builder_adds.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_idxs.hpp>
#include <stan/lang/generator/generate_idxs_user.hpp>
#include <stan/lang/generator/generate_idx.hpp>
#include <stan/lang/generator/generate_idx_user.hpp>
#include <stan/lang/generator/idx_visgen.hpp>
#include <stan/lang/generator/idx_user_visgen.hpp>

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

TEST(blockVarType, createIllFormed) {
  ill_formed_type tIll;
  block_var_type x(tIll);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
  EXPECT_EQ("0", msgs.str());
}

TEST(blockVarType, createDouble) {
  double_block_type tDouble;
  block_var_type x(tDouble);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
  EXPECT_EQ("1", msgs.str());
}

TEST(blockVarType, createVector) {
  range r1(double_literal(-0.1), double_literal(0.1));
  expression d1 = int_literal(2);
  vector_block_type tVector(r1, d1);
  block_var_type x(tVector);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
  EXPECT_EQ("2", msgs.str());
}

TEST(blockVarType, createRowVector) {
  range r1(double_literal(-0.1), double_literal(0.1));
  expression d1 = int_literal(3);
  row_vector_block_type tRowVector(r1, d1);
  block_var_type x(tRowVector);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
  EXPECT_EQ("3", msgs.str());
}

TEST(blockVarType, createMatrix) {
  range r1(double_literal(-0.1), double_literal(0.1));
  expression d1 = int_literal(4);
  expression d2 = int_literal(5);
  matrix_block_type tMatrix(r1, d1, d2);
  block_var_type x(tMatrix);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
  EXPECT_EQ("(4 * 5)", msgs.str());
}


TEST(blockVarType, createCholeskyFactor) {
  expression M(int_literal(2));
  expression N(int_literal(3));
  cholesky_factor_cov_block_type tCFCov(M, N);
  block_var_type x(tCFCov);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}


TEST(blockVarType, createCholeskyFactorCorr) {
  expression K(int_literal(3));
  cholesky_factor_corr_block_type tCFCorr(K);
  block_var_type x(tCFCorr);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}

TEST(blockVarType, createArrayCFCorr) {
  expression d1(int_literal(7));
  expression d2(int_literal(8));
  std::vector<expression> dims;
  dims.push_back(d1);
  dims.push_back(d2);
  expression K(int_literal(3));
  cholesky_factor_corr_block_type tCFCorr(K);
  block_array_type bat(tCFCorr,dims);
  block_var_type x(bat);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}

TEST(blockVarType, createCorrMatrix) {
  expression K(int_literal(3));
  corr_matrix_block_type tCorrMat(K);
  block_var_type x(tCorrMat);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}

TEST(blockVarType, createCovMatrix) {
  expression K(int_literal(4));
  cov_matrix_block_type tCovMat(K);
  block_var_type x(tCovMat);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}

TEST(blockVarType, createOrdered) {
  expression K(int_literal(5));
  ordered_block_type tOrd(K);
  block_var_type x(tOrd);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}

TEST(blockVarType, createArrayOrd) {
  expression d1(int_literal(7));
  expression K(int_literal(3));
  ordered_block_type tOrd(K);
  block_array_type bat(tOrd,d1);
  block_var_type x(bat);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}

TEST(blockVarType, createPosOrdered) {
  expression K(int_literal(3));
  positive_ordered_block_type tPosOrd(K);
  block_var_type x(tPosOrd);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}

TEST(blockVarType, createSimplex) {
  expression K(int_literal(8));
  simplex_block_type tSimplex(K);
  block_var_type x(tSimplex);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}

TEST(blockVarType, createArraySimplex) {
  expression d1(int_literal(7));
  expression K(int_literal(3));
  simplex_block_type tSimplex(K);
  block_array_type bat(tSimplex,K);
  block_var_type x(bat);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}

TEST(blockVarType, createUnitVec) {
  expression K(int_literal(3));
  unit_vector_block_type tUnitVec(K);
  block_var_type x(tUnitVec);

  std::stringstream msgs;
  generate_expression(x.params_total(), false, msgs);
}
