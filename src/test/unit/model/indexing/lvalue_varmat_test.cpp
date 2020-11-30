#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/lvalue_varmat.hpp>
#include <stan/model/indexing/lvalue.hpp>
#include <stan/model/indexing/rvalue.hpp>
#include <stan/math/rev.hpp>
#include <test/unit/util.hpp>
#include <test/unit/model/indexing/util.hpp>
#include <gtest/gtest.h>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
using stan::model::assign;
using stan::model::cons_index_list;
using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_uni;
using stan::model::nil_index_list;
using std::vector;

struct VarAssign : public testing::Test {
  void SetUp() {
    // make sure memory's clean before starting each test
    stan::math::recover_memory();
  }
  void TearDown() {
    // make sure memory's clean before starting each test
    stan::math::recover_memory();
  }
};

template <typename T1, typename I, typename T2>
void test_throw_out_of_range(T1& lhs, const I& idxs, const T2& rhs) {
  EXPECT_THROW(stan::model::assign(lhs, idxs, rhs), std::out_of_range);
}

template <typename T1, typename I, typename T2>
void test_throw_invalid_arg(T1& lhs, const I& idxs, const T2& rhs) {
  EXPECT_THROW(stan::model::assign(lhs, idxs, rhs), std::invalid_argument);
}

TEST_F(VarAssign, nil) {
  using stan::math::var_value;
  auto x = stan::model::test::generate_linear_var_vector(5);
  auto y = stan::model::test::generate_linear_var_vector(5, 1.0);
  assign(x, nil_index_list(), y);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x.val().coeffRef(i), i + 1);
  }
  stan::math::sum(x).grad();
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x.adj().coeffRef(i), 1);
    EXPECT_FLOAT_EQ(y.adj().coeffRef(i), 1);
  }
}

template <typename Vec>
void test_uni_vec() {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<Vec>(5);
  Vec x_val = x.val();
  stan::math::var y(18);
  assign(x, index_list(index_uni(2)), y);
  EXPECT_FLOAT_EQ(y.val(), x.val()[1]);
  stan::math::sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i != 1; };
  check_vector_adjs(check_i, x, "lhs");
  EXPECT_FLOAT_EQ(y.adj(), 1);
  test_throw_out_of_range(x, index_list(index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_uni(6)), y);
}

TEST_F(VarAssign, uni_vec) { test_uni_vec<Eigen::VectorXd>(); }

TEST_F(VarAssign, uni_rowvec) { test_uni_vec<Eigen::RowVectorXd>(); }

template <typename Vec>
void test_multi_vec() {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<Vec>(5);
  Vec x_val = x.val();
  auto y = generate_linear_var_vector<Vec>(3, 10);
  vector<int> ns;
  ns.push_back(2);
  ns.push_back(4);
  ns.push_back(2);
  assign(x, index_list(index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[3]);
  EXPECT_FLOAT_EQ(y.val()[2], x.val()[1]);
  stan::math::sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i_x = [](int i) { return i == 1 || i == 3; };
  check_vector_adjs(check_i_x, x, "lhs", 0);
  auto check_i_y = [](int i) { return i > 0; };
  check_vector_adjs(check_i_y, y, "rhs", 1);
}

TEST_F(VarAssign, multi_vec) { test_multi_vec<Eigen::VectorXd>(); }

TEST_F(VarAssign, multi_rowvec) { test_multi_vec<Eigen::RowVectorXd>(); }

template <typename Vec>
void test_omni_vec() {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<Vec>(5);
  auto y = generate_linear_var_vector<Vec>(5, 10);
  Vec y_val = y.val();

  assign(x, index_list(index_omni()), y);
  EXPECT_FLOAT_EQ(y.val()[0], x.val()[0]);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[1]);
  EXPECT_FLOAT_EQ(y.val()[2], x.val()[2]);
  EXPECT_FLOAT_EQ(y.val()[3], x.val()[3]);
  EXPECT_FLOAT_EQ(y.val()[4], x.val()[4]);
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), y_val);
  auto check_i = [](int i) { return true; };
  check_vector_adjs(check_i, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Vec::Ones(5));
}

TEST_F(VarAssign, omni_vec) { test_omni_vec<Eigen::VectorXd>(); }

TEST_F(VarAssign, omni_rowvec) { test_omni_vec<Eigen::RowVectorXd>(); }

template <typename Vec>
void test_min_vec() {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<Vec>(5);
  auto y = generate_linear_var_vector<Vec>(3, 10);
  Vec x_val(x.val());
  assign(x, index_list(index_min(3)), y);
  EXPECT_FLOAT_EQ(x.val()(2), y.val()(0));
  EXPECT_FLOAT_EQ(x.val()(3), y.val()(1));
  EXPECT_FLOAT_EQ(x.val()(4), y.val()(2));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i < 2; };
  check_vector_adjs(check_i, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Vec::Ones(3));
  test_throw_out_of_range(x, index_list(index_min(0)), y);
}
TEST_F(VarAssign, min_vec) { test_min_vec<Eigen::VectorXd>(); }

TEST_F(VarAssign, min_rowvec) { test_min_vec<Eigen::RowVectorXd>(); }

template <typename Vec>
void test_max_vec() {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<Vec>(5);
  Vec x_val = x.val();
  auto y = generate_linear_var_vector<Vec>(2, 10);

  assign(x, index_list(index_max(2)), y);
  EXPECT_FLOAT_EQ(y.val()[0], x.val()[0]);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[1]);
  stan::math::sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i > 1; };
  check_vector_adjs(check_i, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Vec::Ones(2));
}

TEST_F(VarAssign, max_vec) { test_max_vec<Eigen::VectorXd>(); }

TEST_F(VarAssign, max_rowvec) { test_max_vec<Eigen::RowVectorXd>(); }

template <typename Vec>
void test_positive_minmax_varvector() {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<Vec>(5);
  Vec x_val = x.val();
  auto y = generate_linear_var_vector<Vec>(4, 10);

  assign(x, index_list(index_min_max(1, 4)), y);
  EXPECT_FLOAT_EQ(x.val()(0), 10);
  EXPECT_FLOAT_EQ(x.val()(1), 11);
  EXPECT_FLOAT_EQ(x.val()(2), 12);
  EXPECT_FLOAT_EQ(x.val()(3), 13);
  EXPECT_FLOAT_EQ(x.val()(4), 4);
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return (i > 3); };
  check_vector_adjs(check_i, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Vec::Ones(4));
  test_throw_out_of_range(x, index_list(index_min_max(0, 3)), y);
  test_throw_out_of_range(x, index_list(index_min_max(1, 8)), y);
}

TEST_F(VarAssign, positive_minmax_vec) {
  test_positive_minmax_varvector<Eigen::VectorXd>();
}

TEST_F(VarAssign, positive_minmax_rowvec) {
  test_positive_minmax_varvector<Eigen::RowVectorXd>();
}

template <typename Vec>
void test_negative_minmax_varvector() {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<Vec>(5);
  Vec x_val = x.val();
  auto y = generate_linear_var_vector<Vec>(4, 10);

  assign(x, index_list(index_min_max(4, 1)), y);
  EXPECT_FLOAT_EQ(x.val()(0), 13);
  EXPECT_FLOAT_EQ(x.val()(1), 12);
  EXPECT_FLOAT_EQ(x.val()(2), 11);
  EXPECT_FLOAT_EQ(x.val()(3), 10);
  EXPECT_FLOAT_EQ(x.val()(4), 4);
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return (i > 3); };
  check_vector_adjs(check_i, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Vec::Ones(4));
  test_throw_out_of_range(x, index_list(index_min_max(3, 0)), y);
  test_throw_out_of_range(x, index_list(index_min_max(8, 1)), y);
}

TEST_F(VarAssign, negative_minmax_vec) {
  test_negative_minmax_varvector<Eigen::VectorXd>();
}

TEST_F(VarAssign, negative_minmax_rowvec) {
  test_negative_minmax_varvector<Eigen::RowVectorXd>();
}

template <typename Vec>
void test_uni_uni_vec_eigvec() {
  using stan::math::sum;
  using stan::math::var;
  using stan::math::var_value;

  Vec xs0_val(3);
  xs0_val << 0.0, 0.1, 0.2;

  Vec xs1_val(3);
  xs1_val << 1.0, 1.1, 1.2;

  var_value<Vec> xs0(xs0_val);
  var_value<Vec> xs1(xs1_val);
  vector<var_value<Vec>> xs;
  xs.push_back(xs0);
  xs.push_back(xs1);

  var y = 15;
  assign(xs, index_list(index_uni(2), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y.val(), xs[1].val()(2));

  (sum(xs[0]) + sum(xs[1])).grad();
  EXPECT_FLOAT_EQ(y.adj(), 1);

  for (Eigen::Index i = 0; i < xs[0].size(); ++i) {
    EXPECT_FLOAT_EQ(xs[0].val()(i), xs0_val(i));
    EXPECT_FLOAT_EQ(xs[0].adj()(i), 1);
  }

  for (Eigen::Index i = 0; i < xs[0].size(); ++i) {
    EXPECT_FLOAT_EQ(xs[1].val()(i), xs1_val(i));
    if (i == 2) {
      EXPECT_FLOAT_EQ(xs[1].adj()(i), 0);
    } else {
      EXPECT_FLOAT_EQ(xs[1].adj()(i), 1);
    }
  }

  test_throw_out_of_range(xs, index_list(index_uni(0), index_uni(3)), y);
  test_throw_out_of_range(xs, index_list(index_uni(2), index_uni(0)), y);
  test_throw_out_of_range(xs, index_list(index_uni(10), index_uni(3)), y);
  test_throw_out_of_range(xs, index_list(index_uni(2), index_uni(10)), y);
}

TEST_F(VarAssign, uni_uni_std_vecvec) {
  test_uni_uni_vec_eigvec<Eigen::VectorXd>();
}

TEST_F(VarAssign, uni_uni_std_vecrowvec) {
  test_uni_uni_vec_eigvec<Eigen::RowVectorXd>();
}

/**
  * Tests are not exhaustive. They cover each index individually and each
  * index as the right hand side of a double index.
  * index_uni - A single cell.
  * index_multi - Access multiple cells.
  * index_omni - A no-op for all indices along a dimension.
  * index_min - index from min:N
  * index_max - index from 1:max
  * index_min_max - index from min:max
  * nil_index_list - no-op
  * Tests are sorted in the above order and first call the individual index
  * and then with the index on the right hand side of an index list.
  */

// uni
TEST_F(VarAssign, uni_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_vector<Eigen::RowVectorXd>(5, 10);
  assign(x, index_list(index_uni(1)), y);
  EXPECT_MATRIX_EQ(y.val().row(0), x.val().row(0));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i != 0; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Eigen::RowVectorXd::Ones(5));
}

TEST_F(VarAssign, uni_uni_matrix) {
  using stan::math::sum;
  using stan::math::var;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;
  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  var y = 10.12;
  assign(x, index_list(index_uni(2), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y.val(), x.val()(1, 2));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i == 1; };
  auto check_j = [](int j) { return j == 2; };
  check_matrix_adjs(check_i, check_j, x, "lhs", 0);
  EXPECT_FLOAT_EQ(y.adj(), 1);

  test_throw_out_of_range(x, index_list(index_uni(0), index_uni(3)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_uni(7), index_uni(3)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_uni(7)), y);
}

TEST_F(VarAssign, multi_uni_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(3, 4);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_vector<Eigen::RowVectorXd>(3, 10);

  std::vector<int> ns;
  ns.push_back(3);
  ns.push_back(1);
  ns.push_back(1);
  assign(x, index_list(index_multi(ns), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y.val()(0), x.val()(2, 2));
  EXPECT_FLOAT_EQ(y.val()(2), x.val()(0, 2));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i_x = [](int i) { return (i == 0 || i == 2); };
  auto check_j_x = [](int j) { return j == 2; };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 0);
  auto check_i_y = [](int i) { return i != 1; };
  check_vector_adjs(check_i_y, y, "rhs", 1);

  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_list(index_multi(ns), index_uni(3)), y);
  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, index_list(index_multi(ns), index_uni(3)), y);
  ns.push_back(2);
  test_throw_invalid_arg(x, index_list(index_multi(ns), index_uni(3)), y);
}

TEST_F(VarAssign, omni_uni_matrix) {
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_vector<Eigen::VectorXd>(5, 10);
  assign(x, index_list(index_omni(), index_uni(1)), y);
  EXPECT_MATRIX_EQ(y.val(), x.val().col(0));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return true; };
  auto check_j = [](int j) { return j == 0; };
  check_matrix_adjs(check_i, check_j, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::VectorXd::Ones(5));
}

TEST_F(VarAssign, minmax_uni_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(3, 4);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_vector(2, 10);

  assign(x, index_list(index_min_max(2, 3), index_uni(4)), y);
  EXPECT_MATRIX_EQ(y.val(), x.val().col(3).segment(1, 2));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return (i == 1 || i == 2); };
  auto check_j = [](int j) { return j == 3; };
  check_matrix_adjs(check_i, check_j, x, "lhs", 0);
  EXPECT_FLOAT_EQ(y.adj()(0), 1);
  EXPECT_FLOAT_EQ(y.adj()(1), 1);

  test_throw_out_of_range(x, index_list(index_min_max(2, 3), index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_min_max(2, 3), index_uni(5)), y);
  test_throw_out_of_range(x, index_list(index_min_max(0, 1), index_uni(4)), y);
  test_throw_invalid_arg(x, index_list(index_min_max(1, 3), index_uni(4)), y);
}

// multi
TEST_F(VarAssign, multi_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(7, 5, 10);
  std::vector<int> row_idx;
  row_idx.push_back(3);
  row_idx.push_back(4);
  row_idx.push_back(1);
  row_idx.push_back(4);
  row_idx.push_back(1);
  row_idx.push_back(4);
  row_idx.push_back(5);
  stan::arena_t<std::vector<int>> x_idx;
  stan::arena_t<std::vector<int>> y_idx;
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = row_idx.size() - 1; i >= 0; --i) {
    if (!stan::model::internal::check_duplicate(x_idx, row_idx[i] - 1)) {
      y_idx.push_back(i);
      x_idx.push_back(row_idx[i] - 1);
    }
  }
  assign(x, index_list(index_multi(row_idx)), y);
  // We use these to check the adjoints
  for (int i = 0; i < x_idx.size(); ++i) {
    EXPECT_MATRIX_EQ(x.val().row(x_idx[i]), y.val().row(y_idx[i]))
  }
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  // We don't assign to row 1
  auto check_i_x = [](int i) { return i == 1; };
  auto check_j_x = [](int j) { return true; };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 1);

  auto check_i_y = [](int i) { return (i == 0 || i > 3); };
  auto check_j_y = [](int j) { return true; };
  check_matrix_adjs(check_i_y, check_j_y, y, "rhs", 1);
  row_idx.pop_back();
  test_throw_invalid_arg(x, index_list(index_multi(row_idx)), y);
  row_idx.push_back(22);
  test_throw_out_of_range(x, index_list(index_multi(row_idx)), y);
}

TEST_F(VarAssign, uni_multi_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_vector<Eigen::RowVectorXd>(4, 10);

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  ns.push_back(3);
  assign(x, index_list(index_uni(3), index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y.val()(0), x.val()(2, 3));
  EXPECT_FLOAT_EQ(y.val()(1), x.val()(2, 0));
  EXPECT_FLOAT_EQ(y.val()(3), x.val()(2, 2));

  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i_x = [](int i) { return i == 2; };
  auto check_j_x = [](int j) { return (j == 0 || j == 2 || j == 3); };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 0);
  auto check_i_y = [](int i) { return i != 2; };
  check_vector_adjs(check_i_y, y, "rhs", 1);
  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_list(index_uni(3), index_multi(ns)), y);
  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, index_list(index_uni(3), index_multi(ns)), y);
  ns.push_back(2);
  test_throw_invalid_arg(x, index_list(index_uni(3), index_multi(ns)), y);
}

TEST_F(VarAssign, multi_multi_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(7, 7, 10);
  std::vector<int> row_idx;
  std::vector<int> col_idx;
  row_idx.push_back(3);
  row_idx.push_back(4);
  row_idx.push_back(1);
  row_idx.push_back(4);
  row_idx.push_back(1);
  row_idx.push_back(4);
  row_idx.push_back(5);

  col_idx.push_back(1);
  col_idx.push_back(4);
  col_idx.push_back(4);
  col_idx.push_back(3);
  col_idx.push_back(2);
  col_idx.push_back(1);
  col_idx.push_back(5);
  stan::arena_t<std::vector<std::array<int, 2>>> x_idx;
  stan::arena_t<std::vector<std::array<int, 2>>> y_idx;
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int j = col_idx.size() - 1; j >= 0; --j) {
    for (int i = row_idx.size() - 1; i >= 0; --i) {
      if (!stan::model::internal::check_duplicate(x_idx, row_idx[i] - 1,
                                                  col_idx[j] - 1)) {
        y_idx.push_back(std::array<int, 2>{i, j});
        x_idx.push_back(std::array<int, 2>{row_idx[i] - 1, col_idx[j] - 1});
      }
    }
  }
  assign(x, index_list(index_multi(row_idx), index_multi(col_idx)), y);
  // We use these to check the adjoints
  for (int i = 0; i < x_idx.size(); ++i) {
    EXPECT_FLOAT_EQ(x.val()(x_idx[i][0], x_idx[i][1]),
                    y.val()(y_idx[i][0], y_idx[i][1]))
        << "Failed for \ni: " << i << "\nx_idx[i][0]: " << x_idx[i][0]
        << " x_idx[i][1]: " << x_idx[i][1] << "\ny_idx[i][0]: " << y_idx[i][0]
        << " y_idx[i][1]: " << y_idx[i][1];
  }
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  // We don't assign to row 1
  auto check_i_x = [](int i) { return i == 1; };
  auto check_j_x = [](int j) { return true; };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 1);

  auto check_i_y = [](int i) { return (i == 0 || i > 3); };
  auto check_j_y = [](int j) { return (j > 1); };
  check_matrix_adjs(check_i_y, check_j_y, y, "rhs", 1);
  col_idx.pop_back();
  test_throw_invalid_arg(
      x, index_list(index_multi(row_idx), index_multi(col_idx)), y);
  col_idx.push_back(22);
  test_throw_out_of_range(
      x, index_list(index_multi(row_idx), index_multi(col_idx)), y);
  col_idx.pop_back();
  col_idx.push_back(5);

  row_idx.pop_back();
  test_throw_invalid_arg(
      x, index_list(index_multi(row_idx), index_multi(col_idx)), y);
  row_idx.push_back(22);
  test_throw_out_of_range(
      x, index_list(index_multi(row_idx), index_multi(col_idx)), y);
}

// omni
TEST_F(VarAssign, omni_matrix) {
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(5, 5, 10);
  assign(x, index_list(index_omni()), y);
  EXPECT_MATRIX_EQ(y.val(), x.val());
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), y.val());
  EXPECT_MATRIX_EQ(x.adj(), Eigen::MatrixXd::Ones(5, 5));
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(5, 5));
}

TEST_F(VarAssign, omni_omni_matrix) {
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(5, 5, 10);
  assign(x, index_list(index_omni(), index_omni()), y);
  EXPECT_MATRIX_EQ(y.val(), x.val());
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), y.val());
  EXPECT_MATRIX_EQ(x.adj(), Eigen::MatrixXd::Ones(5, 5));
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(5, 5));
}

TEST_F(VarAssign, uni_omni_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_vector<Eigen::RowVectorXd>(5, 10);
  assign(x, index_list(index_uni(1), index_omni()), y);
  EXPECT_MATRIX_EQ(y.val().row(0), x.val().row(0));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i != 0; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Eigen::RowVectorXd::Ones(5));
}

// min
TEST_F(VarAssign, min_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(3, 4);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(2, 4, 10);

  assign(x, index_list(index_min(2)), y);
  EXPECT_MATRIX_EQ(x.val().bottomRows(2), y.val());
  sum(x).grad();
  // We don't assign to row 1
  auto check_i_x = [](int i) { return i > 0; };
  auto check_j_x = [](int j) { return true; };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(2, 4));
  test_throw_invalid_arg(x, index_list(index_min(1)), y);
  var_value<MatrixXd> z(MatrixXd::Ones(1, 2));
  test_throw_invalid_arg(x, index_list(index_min(1)), z);
  test_throw_invalid_arg(x, index_list(index_min(2)), z);
}

TEST_F(VarAssign, minmax_min_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(3, 4);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(2, 3, 10);
  assign(x, index_list(index_min_max(2, 3), index_min(2)), y);
  EXPECT_MATRIX_EQ(y.val(), x.val().block(1, 1, 2, 3));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return (i == 1 || i == 2); };
  auto check_j = [](int j) { return j > 0; };
  check_matrix_adjs(check_i, check_j, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), MatrixXd::Ones(2, 3));
}

// max
TEST_F(VarAssign, max_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(3, 4);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(2, 4, 10);

  assign(x, index_list(index_max(2)), y);
  EXPECT_MATRIX_EQ(x.val().topRows(2), y.val());
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i_x = [](int i) { return i < 2; };
  auto check_j_x = [](int j) { return true; };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(2, 4));
  test_throw_invalid_arg(x, index_list(index_max(1)), y);
  var_value<MatrixXd> z(MatrixXd::Ones(1, 4));
  test_throw_invalid_arg(x, index_list(index_max(2)), z);
  test_throw_invalid_arg(x, index_list(index_max(3)), z);
}

TEST_F(VarAssign, min_max_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(3, 4);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(2, 2, 10);

  assign(x, index_list(index_min(2), index_max(2)), y);
  EXPECT_MATRIX_EQ(x.val().block(1, 0, 2, 2), y.val());
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  // We don't assign to row 1
  auto check_i_x = [](int i) { return i > 0; };
  auto check_j_x = [](int j) { return j < 2; };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(2, 2));
  test_throw_invalid_arg(x, index_list(index_min(2), index_max(1)), y);
  var_value<MatrixXd> z(MatrixXd::Ones(1, 4));
  test_throw_invalid_arg(x, index_list(index_min(2), index_max(2)), z);
  test_throw_invalid_arg(x, index_list(index_min(2), index_max(3)), z);
}

// minmax
TEST_F(VarAssign, positive_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;
  Eigen::Matrix<double, -1, -1> x_val(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
    x_rev_val(i) = x_val.size() - i - 1;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    var_value<Eigen::MatrixXd> x_rev(x_rev_val);
    const int ii = i + 1;
    assign(x, index_list(index_min_max(1, ii)),
           x_rev.block(0, 0, ii, 5));
    auto x_val_check = x.val().block(0, 0, ii, 5);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, 5);
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return true; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    stan::math::recover_memory();
  }
}

TEST_F(VarAssign, negative_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x_val(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
    x_rev_val(i) = x_val.size() - i - 1;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    var_value<Eigen::MatrixXd> x_rev(x_rev_val);
    const int ii = i + 1;
    assign(x, index_list(index_min_max(ii, 1)), x_rev.block(0, 0, ii, 5));
    auto x_val_check = x.val().block(0, 0, ii, 5);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, 5).colwise().reverse();
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return true; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    stan::math::recover_memory();
  }
}

TEST_F(VarAssign, positive_minmax_positive_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;
  Eigen::Matrix<double, -1, -1> x_val(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
    x_rev_val(i) = x_val.size() - i - 1;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    var_value<Eigen::MatrixXd> x_rev(x_rev_val);
    const int ii = i + 1;
    assign(x, index_list(index_min_max(1, ii), index_min_max(1, ii)),
           x_rev.block(0, 0, ii, ii));
    auto x_val_check = x.val().block(0, 0, ii, ii);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, ii);
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return jj <= i; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    stan::math::recover_memory();
  }
}

TEST_F(VarAssign, positive_minmax_negative_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x_val(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
    x_rev_val(i) = x_val.size() - i - 1;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    var_value<Eigen::MatrixXd> x_rev(x_rev_val);
    const int ii = i + 1;
    assign(x, index_list(index_min_max(1, ii), index_min_max(ii, 1)),
           x_rev.block(0, 0, ii, ii));
    auto x_val_check = x.val().block(0, 0, ii, ii);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, ii).rowwise().reverse();
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return jj <= i; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    stan::math::recover_memory();
  }
}

TEST_F(VarAssign, negative_minmax_positive_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x_val(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
    x_rev_val(i) = x_val.size() - i - 1;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    var_value<Eigen::MatrixXd> x_rev(x_rev_val);
    const int ii = i + 1;
    assign(x, index_list(index_min_max(ii, 1), index_min_max(1, ii)),
           x_rev.block(0, 0, ii, ii));
    auto x_val_check = x.val().block(0, 0, ii, ii);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, ii).colwise().reverse();
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return jj <= i; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    stan::math::recover_memory();
  }
}

TEST_F(VarAssign, negative_minmax_negative_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x_val(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
    x_rev_val(i) = x_val.size() - i - 1;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    var_value<Eigen::MatrixXd> x_rev(x_rev_val);
    const int ii = i + 1;
    assign(x, index_list(index_min_max(ii, 1), index_min_max(ii, 1)),
           x_rev.block(0, 0, ii, ii));
    auto x_val_check = x.val().block(0, 0, ii, ii);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, ii).reverse();
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return jj <= i; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    stan::math::recover_memory();
  }
}

TEST_F(VarAssign, uni_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_vector<Eigen::RowVectorXd>(3, 10);
  assign(x, index_list(index_uni(2), index_min_max(2, 4)), y);
  EXPECT_MATRIX_EQ(y.val().segment(0, 3), x.val().row(1).segment(1, 3));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i == 1; };
  auto check_j = [](int j) { return (j > 0 && j < 4); };
  check_matrix_adjs(check_i, check_j, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::RowVectorXd::Ones(3));
  test_throw_out_of_range(x, index_list(index_uni(0), index_min_max(2, 4)), y);
  test_throw_out_of_range(x, index_list(index_uni(7), index_min_max(2, 4)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_min_max(0, 2)), y);
}

// nil only shows up as a single index
TEST_F(VarAssign, nil_matrix) {
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(5, 5, 10);
  assign(x, nil_index_list(), y);
  EXPECT_MATRIX_EQ(y.val(), x.val());
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), y.val());
  EXPECT_MATRIX_EQ(x.adj(), Eigen::MatrixXd::Ones(5, 5));
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(5, 5));
}
