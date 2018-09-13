#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <set>
#include <vector>

using stan::lang::idx;
using stan::lang::uni_idx;
using stan::lang::omni_idx;
using stan::lang::expression;
using stan::lang::int_literal;
using stan::lang::function_signatures;
using stan::lang::bare_array_type;
using stan::lang::bare_expr_type;
using stan::lang::ill_formed_type;
using stan::lang::void_type;
using stan::lang::double_type;
using stan::lang::int_type;
using stan::lang::vector_type;
using stan::lang::row_vector_type;
using stan::lang::matrix_type;
using std::vector;

// Type Inference Tests for Generalized Indexing

// tests recovery of base expression type and number of dims
// given expression and indexing
void test_recover(bare_expr_type base_et_expected,
                  size_t num_dims_expected,
                  bare_expr_type base_et, size_t num_dims,
                  const std::vector<stan::lang::idx>& idxs) {
  stan::lang::variable v("foo");
  if (num_dims == 0)
    v.set_type(base_et);
  else
    v.set_type(bare_array_type(base_et, num_dims));
  stan::lang::expression e(v);
  bare_expr_type et = indexed_type(e, idxs);
  EXPECT_EQ(base_et_expected, et.innermost_type());
  EXPECT_EQ(num_dims_expected, et.array_dims());
}

void test_err(bare_expr_type base_et, size_t num_dims,
              const std::vector<stan::lang::idx>& idxs) {
  stan::lang::variable v("foo");
  if (num_dims == 0)
    v.set_type(base_et);
  else
    v.set_type(bare_array_type(base_et, num_dims));
  stan::lang::expression e(v);
  bare_expr_type et = indexed_type(e, idxs);
  EXPECT_EQ(bare_expr_type(ill_formed_type()), et);
}

TEST(langAst, idxs) {
  const stan::lang::bare_expr_type bet[]
    = { bare_expr_type(int_type()), bare_expr_type(double_type()),
        bare_expr_type(vector_type()), bare_expr_type(row_vector_type()),
        bare_expr_type(matrix_type()) };
  vector<idx> idxs;
  for (size_t n = 0; n < 4; ++n)
    for (int i = 0; i < 5; ++i)
      test_recover(bet[i], n, bet[i], n, idxs);
}

void one_index_recover(const std::vector<stan::lang::idx>& idxs, size_t redux) {
  const stan::lang::bare_expr_type bet[]
    = { bare_expr_type(int_type()), bare_expr_type(double_type()),
        bare_expr_type(vector_type()), bare_expr_type(row_vector_type()),
        bare_expr_type(matrix_type()) };
  for (size_t n = 1; n < 4; ++n)
    for (int i = 0; i < 5; ++i)
      test_recover(bet[i], n - redux, bet[i], n, idxs);
}

void one_index_errs(const std::vector<stan::lang::idx>& idxs) {
  test_err(bare_expr_type(double_type()), 0U, idxs);
  test_err(bare_expr_type(int_type()), 0U, idxs);
}

TEST(langAst, idxs0) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));

  one_index_errs(idxs);
  one_index_recover(idxs, 1U);
  test_recover(bare_expr_type(double_type()), 0U,
               bare_expr_type(vector_type()), 0U, idxs);
  test_recover(bare_expr_type(double_type()), 0U,
               bare_expr_type(row_vector_type()), 0U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 0U,
               bare_expr_type(matrix_type()), 0U, idxs);
}

TEST(langAst, idxs1) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());

  one_index_errs(idxs);
  one_index_recover(idxs, 0U);
  test_recover(bare_expr_type(vector_type()), 0U, bare_expr_type(vector_type()), 0U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 0U, bare_expr_type(row_vector_type()), 0U, idxs);
  test_recover(bare_expr_type(matrix_type()), 0U, bare_expr_type(matrix_type()), 0U, idxs);
}

void two_index_recover(const std::vector<stan::lang::idx>& idxs, size_t redux) {
  const stan::lang::bare_expr_type bet[]
    = { bare_expr_type(int_type()), bare_expr_type(double_type()),
        bare_expr_type(vector_type()), bare_expr_type(row_vector_type()),
        bare_expr_type(bare_expr_type(matrix_type())) };
  for (size_t n = 2; n < 4; ++n)
    for (int i = 0; i < 5; ++i)
      test_recover(bet[i], n - redux, bet[i], n, idxs);
}

void two_index_errs(const std::vector<stan::lang::idx>& idxs) {
  test_err(bare_expr_type(double_type()), 0U, idxs);
  test_err(bare_expr_type(double_type()), 1U, idxs);
  test_err(bare_expr_type(int_type()), 0U, idxs);
  test_err(bare_expr_type(int_type()), 1U, idxs);
  test_err(bare_expr_type(vector_type()), 0U, idxs);
  test_err(bare_expr_type(row_vector_type()), 0U, idxs);
}

TEST(langAst, idxs00) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));

  two_index_errs(idxs);
  two_index_recover(idxs, 2U);
  test_recover(bare_expr_type(double_type()), 0U, bare_expr_type(vector_type()), 1U, idxs);
  test_recover(bare_expr_type(double_type()), 0U, bare_expr_type(row_vector_type()), 1U, idxs);
  test_recover(bare_expr_type(double_type()), 0U, bare_expr_type(matrix_type()), 0U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 0U, bare_expr_type(matrix_type()), 1U, idxs);
}

TEST(langAst, idxs01) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(5))));
  idxs.push_back(omni_idx());

  two_index_errs(idxs);
  two_index_recover(idxs, 1U);
  test_recover(bare_expr_type(vector_type()), 0U, bare_expr_type(vector_type()), 1U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 0U, bare_expr_type(row_vector_type()), 1U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 0U, bare_expr_type(matrix_type()), 0U, idxs);
  test_recover(bare_expr_type(matrix_type()), 0U, bare_expr_type(matrix_type()), 1U, idxs);
}

TEST(langAst, idxs10) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(5))));

  two_index_errs(idxs);
  two_index_recover(idxs, 1U);
  test_recover(bare_expr_type(double_type()), 1U, bare_expr_type(vector_type()), 1U, idxs);
  test_recover(bare_expr_type(double_type()), 1U, bare_expr_type(row_vector_type()), 1U, idxs);
  test_recover(bare_expr_type(vector_type()), 0U, bare_expr_type(matrix_type()), 0U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 1U, bare_expr_type(matrix_type()), 1U, idxs);
}

TEST(langAst, idxs11) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());

  two_index_errs(idxs);
  two_index_recover(idxs, 0U);
  test_recover(bare_expr_type(vector_type()), 1U, bare_expr_type(vector_type()), 1U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 1U, bare_expr_type(row_vector_type()), 1U, idxs);
  test_recover(bare_expr_type(matrix_type()), 0U, bare_expr_type(matrix_type()), 0U, idxs);
  test_recover(bare_expr_type(matrix_type()), 1U, bare_expr_type(matrix_type()), 1U, idxs);
}

void three_index_recover(const std::vector<stan::lang::idx>& idxs, size_t redux) {
  const stan::lang::bare_expr_type bet[]
    = { bare_expr_type(int_type()), bare_expr_type(double_type()),
        bare_expr_type(vector_type()), bare_expr_type(row_vector_type()),
        bare_expr_type(matrix_type()) };
  for (int i = 0; i < 5; ++i)
    for (size_t n = 3; n < 5; ++n)
      test_recover(bet[i], n - redux, bet[i], n, idxs);
}

void three_index_errs(const std::vector<stan::lang::idx>& idxs) {
  test_err(bare_expr_type(double_type()), 0U, idxs);
  test_err(bare_expr_type(double_type()), 1U, idxs);
  test_err(bare_expr_type(double_type()), 2U, idxs);
  test_err(bare_expr_type(int_type()), 0U, idxs);
  test_err(bare_expr_type(int_type()), 1U, idxs);
  test_err(bare_expr_type(int_type()), 2U, idxs);
  test_err(bare_expr_type(vector_type()), 0U, idxs);
  test_err(bare_expr_type(vector_type()), 1U, idxs);
  test_err(bare_expr_type(row_vector_type()), 0U, idxs);
  test_err(bare_expr_type(row_vector_type()), 1U, idxs);
  test_err(bare_expr_type(matrix_type()), 0U, idxs);
}

TEST(langAst, idxs000) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));
  idxs.push_back(uni_idx(expression(int_literal(7))));

  three_index_errs(idxs);
  three_index_recover(idxs, 3U);
  test_recover(bare_expr_type(double_type()), 0U, bare_expr_type(vector_type()), 2U, idxs);
  test_recover(bare_expr_type(double_type()), 0U, bare_expr_type(row_vector_type()), 2U, idxs);
  test_recover(bare_expr_type(double_type()), 0U, bare_expr_type(matrix_type()), 1U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 0U, bare_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs001) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 2U);
  test_recover(bare_expr_type(vector_type()), 0U, bare_expr_type(vector_type()), 2U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 0U, bare_expr_type(row_vector_type()), 2U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 0U, bare_expr_type(matrix_type()), 1U, idxs);
  test_recover(bare_expr_type(matrix_type()), 0U, bare_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs011) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 1U);
  test_recover(bare_expr_type(vector_type()), 1U, bare_expr_type(vector_type()), 2U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 1U, bare_expr_type(row_vector_type()), 2U, idxs);
  test_recover(bare_expr_type(matrix_type()), 0U, bare_expr_type(matrix_type()), 1U, idxs);
  test_recover(bare_expr_type(matrix_type()), 1U, bare_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs100) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));

  three_index_errs(idxs);
  three_index_recover(idxs, 2U);
  test_recover(bare_expr_type(double_type()), 1U, bare_expr_type(vector_type()), 2U, idxs);
  test_recover(bare_expr_type(double_type()), 1U, bare_expr_type(row_vector_type()), 2U, idxs);
  test_recover(bare_expr_type(double_type()), 1U, bare_expr_type(matrix_type()), 1U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 1U, bare_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs101) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 1U);
  test_recover(bare_expr_type(vector_type()), 1U, bare_expr_type(vector_type()), 2U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 1U, bare_expr_type(row_vector_type()), 2U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 1U, bare_expr_type(matrix_type()), 1U, idxs);
  test_recover(bare_expr_type(matrix_type()), 1U, bare_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs110) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));

  three_index_errs(idxs);
  three_index_recover(idxs, 1U);
  test_recover(bare_expr_type(double_type()), 2U, bare_expr_type(vector_type()), 2U, idxs);
  test_recover(bare_expr_type(double_type()), 2U, bare_expr_type(row_vector_type()), 2U, idxs);
  test_recover(bare_expr_type(vector_type()), 1U, bare_expr_type(matrix_type()), 1U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 2U, bare_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs111) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 0U);
  test_recover(bare_expr_type(vector_type()), 2U, bare_expr_type(vector_type()), 2U, idxs);
  test_recover(bare_expr_type(row_vector_type()), 2U, bare_expr_type(row_vector_type()), 2U, idxs);
  test_recover(bare_expr_type(matrix_type()), 1U, bare_expr_type(matrix_type()), 1U, idxs);
  test_recover(bare_expr_type(matrix_type()), 2U, bare_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, indexOpSliced) {
  using stan::lang::index_op_sliced;

  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));

  // no need to retest all of type inference here --- just that it's plumbed
  index_op_sliced ios;
  stan::lang::variable v("foo");
  v.set_type(bare_array_type(vector_type(), 1U));
  ios.expr_ = v;
  ios.idxs_ = idxs;
  ios.infer_type();
  EXPECT_EQ(bare_expr_type(double_type()), ios.type_.innermost_type());
  EXPECT_EQ(1U, ios.type_.array_dims());
}
