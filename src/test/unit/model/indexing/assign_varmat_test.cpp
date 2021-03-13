#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/assign_varmat.hpp>
#include <stan/model/indexing/assign.hpp>
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
using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_uni;
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

inline bool check_multi_duplicate(
    const stan::arena_t<std::vector<std::array<int, 2>>>& x_idx, int i, int j) {
  for (size_t k = 0; k < x_idx.size(); ++k) {
    if (x_idx[k][0] == i && x_idx[k][1] == j) {
      return true;
    }
  }
  return false;
}

inline bool check_multi_duplicate(const stan::arena_t<std::vector<int>>& x_idx,
                                  int i) {
  for (size_t k = 0; k < x_idx.size(); ++k) {
    if (x_idx[k] == i) {
      return true;
    }
  }
  return false;
}

template <typename T1, typename T2, typename... I>
void test_throw_out_of_range(T1& lhs, const T2& rhs, const I&... idxs) {
  EXPECT_THROW(stan::model::assign(lhs, rhs, "", idxs...), std::out_of_range);
}

template <typename T1, typename T2, typename... I>
void test_throw_invalid_arg(T1& lhs, const T2& rhs, const I&... idxs) {
  EXPECT_THROW(stan::model::assign(lhs, rhs, "", idxs...),
               std::invalid_argument);
}

TEST_F(VarAssign, nil) {
  using stan::math::var_value;
  auto x = stan::model::test::generate_linear_var_vector(5);
  auto y = stan::model::test::generate_linear_var_vector(5, 1.0);
  assign(x, y, "");
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
  assign(x, y, "", index_uni(2));
  EXPECT_FLOAT_EQ(y.val(), x.val()[1]);
  stan::math::sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i != 1; };
  check_vector_adjs(check_i, x, "lhs");
  EXPECT_FLOAT_EQ(y.adj(), 1);
  test_throw_out_of_range(x, y, index_uni(0));
  test_throw_out_of_range(x, y, index_uni(6));
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
  assign(x, y, "", index_multi(ns));
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[3]);
  EXPECT_FLOAT_EQ(y.val()[2], x.val()[1]);
  stan::math::sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i_x = [](int i) { return i == 1 || i == 3; };
  check_vector_adjs(check_i_x, x, "lhs", 0);
  auto check_i_y = [](int i) { return i > 0; };
  check_vector_adjs(check_i_y, y, "rhs", 1);
  ns[2] = 20;
  test_throw_out_of_range(x, y, index_multi(ns));
  ns[2] = 0;
  test_throw_out_of_range(x, y, index_multi(ns));
  ns.push_back(2);
  test_throw_invalid_arg(x, y, index_multi(ns));
  ns.pop_back();
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(4),
                         index_multi(ns));
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(2),
                         index_multi(ns));
}

TEST_F(VarAssign, multi_vec) { test_multi_vec<Eigen::VectorXd>(); }

TEST_F(VarAssign, multi_rowvec) { test_multi_vec<Eigen::RowVectorXd>(); }

template <typename Vec>
void test_multi_alias_vec() {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<Eigen::VectorXd>(5, 1);
  Eigen::VectorXd x_val = x.val();
  vector<int> ns{1, 1, 2, 3};
  assign(x, x.segment(1, 4), "", index_multi(ns));
  EXPECT_MATRIX_EQ(x.val().segment(0, 3), x_val.segment(2, 3));
  EXPECT_MATRIX_EQ(x.val().segment(3, 2), x_val.segment(3, 2));
  stan::math::sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  Eigen::VectorXd exp_adj(5);
  exp_adj << 0, 0, 1, 2, 2;
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
}

TEST_F(VarAssign, multi_alias_vec) { test_multi_alias_vec<Eigen::VectorXd>(); }

template <typename Vec>
void test_omni_vec() {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<Vec>(5);
  auto y = generate_linear_var_vector<Vec>(5, 10);
  Vec y_val = y.val();

  assign(x, y, "", index_omni());
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
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(4), index_omni());
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(6), index_omni());
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
  assign(x, y, "", index_min(3));
  EXPECT_FLOAT_EQ(x.val()(2), y.val()(0));
  EXPECT_FLOAT_EQ(x.val()(3), y.val()(1));
  EXPECT_FLOAT_EQ(x.val()(4), y.val()(2));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i < 2; };
  check_vector_adjs(check_i, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Vec::Ones(3));
  test_throw_out_of_range(x, y, index_min(0));
  test_throw_out_of_range(x, y, index_min(6));
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(4), index_min(3));
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(2), index_min(3));
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

  assign(x, y, "", index_max(2));
  EXPECT_FLOAT_EQ(y.val()[0], x.val()[0]);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[1]);
  stan::math::sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i > 1; };
  check_vector_adjs(check_i, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Vec::Ones(2));
  test_throw_out_of_range(x, y, index_max(0));
  test_throw_out_of_range(x, y, index_max(6));
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(3), index_max(2));
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(1), index_max(2));
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

  assign(x, y, "", index_min_max(1, 4));
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
  test_throw_out_of_range(x, y, index_min_max(0, 3));
  test_throw_out_of_range(x, y, index_min_max(1, 6));
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(5),
                         index_min_max(1, 4));
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(3),
                         index_min_max(1, 4));
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

  assign(x, y, "", index_min_max(4, 1));
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
  test_throw_out_of_range(x, y, index_min_max(3, 0));
  test_throw_out_of_range(x, y, index_min_max(6, 1));
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(5),
                         index_min_max(4, 1));
  test_throw_invalid_arg(x, generate_linear_var_vector<Vec>(3),
                         index_min_max(4, 1));
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
  assign(xs, y, "", index_uni(2), index_uni(3));
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

  test_throw_out_of_range(xs, y, index_uni(0), index_uni(3));
  test_throw_out_of_range(xs, y, index_uni(2), index_uni(0));
  test_throw_out_of_range(xs, y, index_uni(10), index_uni(3));
  test_throw_out_of_range(xs, y, index_uni(2), index_uni(10));
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
  assign(x, y, "", index_uni(1));
  EXPECT_MATRIX_EQ(y.val().row(0), x.val().row(0));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i != 0; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Eigen::RowVectorXd::Ones(5));
  test_throw_out_of_range(x, y, index_uni(0));
  test_throw_out_of_range(x, y, index_uni(6));
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
  assign(x, y, "", index_uni(2), index_uni(3));
  EXPECT_FLOAT_EQ(y.val(), x.val()(1, 2));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i == 1; };
  auto check_j = [](int j) { return j == 2; };
  check_matrix_adjs(check_i, check_j, x, "lhs", 0);
  EXPECT_FLOAT_EQ(y.adj(), 1);

  test_throw_out_of_range(x, y, index_uni(0), index_uni(3));
  test_throw_out_of_range(x, y, index_uni(2), index_uni(0));
  test_throw_out_of_range(x, y, index_uni(7), index_uni(3));
  test_throw_out_of_range(x, y, index_uni(2), index_uni(7));
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
  assign(x, y, "", index_multi(ns), index_uni(3));
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
  test_throw_out_of_range(x, y, index_multi(ns), index_uni(3));
  ns[ns.size() - 1] = 4;
  test_throw_out_of_range(x, y, index_multi(ns), index_uni(3));
  ns.push_back(2);
  test_throw_invalid_arg(x, y, index_multi(ns), index_uni(3));
}

TEST_F(VarAssign, omni_uni_matrix) {
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_vector<Eigen::VectorXd>(5, 10);
  assign(x, y, "", index_omni(), index_uni(1));
  EXPECT_MATRIX_EQ(y.val(), x.val().col(0));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return true; };
  auto check_j = [](int j) { return j == 0; };
  check_matrix_adjs(check_i, check_j, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::VectorXd::Ones(5));
  test_throw_out_of_range(x, y, index_omni(), index_uni(0));
  test_throw_out_of_range(x, y, index_omni(), index_uni(6));
  test_throw_invalid_arg(x, generate_linear_var_vector<Eigen::VectorXd>(6),
                         index_omni(), index_uni(1));
  test_throw_invalid_arg(x, generate_linear_var_vector<Eigen::VectorXd>(4),
                         index_omni(), index_uni(1));
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

  assign(x, y, "", index_min_max(2, 3), index_uni(4));
  EXPECT_MATRIX_EQ(y.val(), x.val().col(3).segment(1, 2));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return (i == 1 || i == 2); };
  auto check_j = [](int j) { return j == 3; };
  check_matrix_adjs(check_i, check_j, x, "lhs", 0);
  EXPECT_FLOAT_EQ(y.adj()(0), 1);
  EXPECT_FLOAT_EQ(y.adj()(1), 1);

  test_throw_out_of_range(x, y, index_min_max(2, 3), index_uni(0));
  test_throw_out_of_range(x, y, index_min_max(2, 3), index_uni(5));
  test_throw_out_of_range(x, y, index_min_max(0, 1), index_uni(4));
  test_throw_out_of_range(x, y, index_min_max(2, 4), index_uni(4));
  test_throw_invalid_arg(x, generate_linear_var_vector(1, 10),
                         index_min_max(2, 3), index_uni(4));
  test_throw_invalid_arg(x, generate_linear_var_vector(3, 10),
                         index_min_max(2, 3), index_uni(4));
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
  std::vector<int> row_idx{3, 4, 1, 4, 1, 4, 5};
  stan::arena_t<std::vector<int>> x_idx;
  stan::arena_t<std::vector<int>> y_idx;
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = row_idx.size() - 1; i >= 0; --i) {
    if (!check_multi_duplicate(x_idx, row_idx[i] - 1)) {
      y_idx.push_back(i);
      x_idx.push_back(row_idx[i] - 1);
    }
  }
  assign(x, y, "", index_multi(row_idx));
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
  test_throw_invalid_arg(x, generate_linear_var_matrix(8, 5, 10),
                         index_multi(row_idx));
  test_throw_invalid_arg(x, generate_linear_var_matrix(6, 5, 10),
                         index_multi(row_idx));
  test_throw_invalid_arg(x, generate_linear_var_matrix(7, 4, 10),
                         index_multi(row_idx));
  test_throw_invalid_arg(x, generate_linear_var_matrix(7, 6, 10),
                         index_multi(row_idx));
  row_idx[3] = 20;
  test_throw_out_of_range(x, y, index_multi(row_idx));
  row_idx[3] = 2;
  row_idx.push_back(2);
  test_throw_invalid_arg(x, y, index_multi(row_idx));
}

TEST_F(VarAssign, multi_alias_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  std::vector<int> row_idx{2, 3, 1, 3};
  assign(x, x.block(0, 0, 4, 5), "", index_multi(row_idx));
  Eigen::MatrixXd x_val_tmp = x_val;
  x_val_tmp.row(0) = x_val.row(2);
  x_val_tmp.row(1) = x_val.row(0);
  x_val_tmp.row(2) = x_val.row(3);
  EXPECT_MATRIX_EQ(x.val(), x_val_tmp);
  sum(x).grad();
  Eigen::MatrixXd exp_adj = Eigen::MatrixXd::Ones(5, 5);
  exp_adj.row(1) = Eigen::RowVectorXd::Zero(5);
  exp_adj.row(3) = Eigen::RowVectorXd::Constant(5, 2);
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
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

  vector<int> ns{4, 1, 3, 3};
  assign(x, y, "", index_uni(3), index_multi(ns));
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
  test_throw_invalid_arg(x,
                         generate_linear_var_vector<Eigen::RowVectorXd>(5, 10),
                         index_uni(3), index_multi(ns));
  test_throw_invalid_arg(x,
                         generate_linear_var_vector<Eigen::RowVectorXd>(3, 10),
                         index_uni(3), index_multi(ns));
  test_throw_out_of_range(x, y, index_uni(0), index_multi(ns));
  test_throw_out_of_range(x, y, index_uni(6), index_multi(ns));
  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, y, index_uni(3), index_multi(ns));
  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, y, index_uni(3), index_multi(ns));
  ns.push_back(2);
  test_throw_invalid_arg(x, y, index_uni(3), index_multi(ns));
}

TEST_F(VarAssign, uni_multi_alias_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  vector<int> ns{1, 1, 2, 3};
  assign(x, x.row(2).segment(0, 4), "", index_uni(3), index_multi(ns));
  Eigen::MatrixXd x_val_tmp = x_val;
  x_val_tmp(2, 0) = 7;
  x_val_tmp(2, 1) = 12;
  x_val_tmp(2, 2) = 17;
  EXPECT_MATRIX_EQ(x.val(), x_val_tmp);
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  Eigen::MatrixXd exp_adj = Eigen::MatrixXd::Ones(5, 5);
  exp_adj(2, 0) = 0;
  exp_adj(2, 3) = 2;
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
}

TEST_F(VarAssign, multi_multi_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(7, 7, 10);
  std::vector<int> row_idx{3, 4, 1, 4, 1, 4, 5};
  std::vector<int> col_idx{1, 4, 4, 3, 2, 1, 5};

  stan::arena_t<std::vector<std::array<int, 2>>> x_idx;
  stan::arena_t<std::vector<std::array<int, 2>>> y_idx;
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int j = col_idx.size() - 1; j >= 0; --j) {
    for (int i = row_idx.size() - 1; i >= 0; --i) {
      if (!check_multi_duplicate(x_idx, row_idx[i] - 1, col_idx[j] - 1)) {
        y_idx.push_back(std::array<int, 2>{i, j});
        x_idx.push_back(std::array<int, 2>{row_idx[i] - 1, col_idx[j] - 1});
      }
    }
  }
  assign(x, y, "", index_multi(row_idx), index_multi(col_idx));
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

  test_throw_invalid_arg(x, generate_linear_var_matrix(6, 7, 10),
                         index_multi(row_idx), index_multi(col_idx));
  test_throw_invalid_arg(x, generate_linear_var_matrix(8, 7, 10),
                         index_multi(row_idx), index_multi(col_idx));
  test_throw_invalid_arg(x, generate_linear_var_matrix(7, 6, 10),
                         index_multi(row_idx), index_multi(col_idx));
  test_throw_invalid_arg(x, generate_linear_var_matrix(7, 8, 10),
                         index_multi(row_idx), index_multi(col_idx));
  col_idx.pop_back();
  test_throw_invalid_arg(x, y, index_multi(row_idx), index_multi(col_idx));
  col_idx.push_back(22);
  test_throw_out_of_range(x, y, index_multi(row_idx), index_multi(col_idx));
  col_idx.pop_back();
  col_idx.push_back(5);

  row_idx.pop_back();
  test_throw_invalid_arg(x, y, index_multi(row_idx), index_multi(col_idx));
  row_idx.push_back(22);
  test_throw_out_of_range(x, y, index_multi(row_idx), index_multi(col_idx));
}

TEST_F(VarAssign, multi_multi_alias_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  std::vector<int> row_idx{1, 2, 2, 4};
  std::vector<int> col_idx{1, 2, 2, 3};
  assign(x, x.block(0, 0, 4, 4).eval(), "", index_multi(row_idx),
         index_multi(col_idx));
  Eigen::MatrixXd x_val_tmp(5, 5);
  /* clang-format off */
  x_val_tmp << 0, 10, 15, 15, 20,
               2, 12, 17, 16, 21,
               2,  7, 12, 17, 22,
               3, 13, 18, 18, 23,
               4,  9, 14, 19, 24;
  /* clang-format on */

  EXPECT_MATRIX_EQ(x.val(), x_val_tmp);
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  Eigen::MatrixXd exp_adj(5, 5);
  /* clang-format off */
  exp_adj << 1, 0, 1, 2, 1,
             0, 0, 0, 1, 1,
             2, 1, 2, 2, 1,
             1, 0, 1, 2, 1,
             1, 1, 1, 1, 1;
  /* clang-format on */
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
}

TEST_F(VarAssign, minmax_multi_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(3, 4, 25);

  vector<int> ns{4, 1, 3, 3};
  assign(x, y, "", index_min_max(1, 3), index_multi(ns));
  Eigen::MatrixXd x_val_tmp = x_val;
  x_val_tmp.col(0).segment(0, 3) = y.val().col(1);
  x_val_tmp.col(2).segment(0, 3) = y.val().col(3);
  x_val_tmp.col(3).segment(0, 3) = y.val().col(0);
  EXPECT_MATRIX_EQ(x.val(), x_val_tmp);
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i_x = [](int i) { return i < 3; };
  auto check_j_x = [](int j) { return (j == 0 || j == 2 || j == 3); };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 0);
  auto check_i_y = [](int i) { return true; };
  auto check_j_y = [](int j) { return j != 2; };
  check_matrix_adjs(check_i_y, check_j_y, y, "lhs", 1);
  test_throw_invalid_arg(x, generate_linear_var_matrix(3, 5, 10),
                         index_min_max(1, 3), index_multi(ns));
  test_throw_invalid_arg(x, generate_linear_var_matrix(3, 3, 10),
                         index_min_max(1, 3), index_multi(ns));
  test_throw_invalid_arg(x, generate_linear_var_matrix(4, 4, 10),
                         index_min_max(1, 3), index_multi(ns));
  test_throw_invalid_arg(x, generate_linear_var_matrix(2, 4, 10),
                         index_min_max(1, 3), index_multi(ns));

  test_throw_out_of_range(x, y, index_min_max(0, 3), index_multi(ns));
  test_throw_out_of_range(x, y, index_min_max(1, 6), index_multi(ns));
  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, y, index_min_max(1, 3), index_multi(ns));
  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, y, index_min_max(1, 3), index_multi(ns));
  ns.push_back(2);
  test_throw_invalid_arg(x, y, index_min_max(1, 3), index_multi(ns));
}

TEST_F(VarAssign, minmax_multi_alias_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::check_vector_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();

  vector<int> ns{4, 1, 3, 3};
  assign(x, x.block(0, 0, 3, 4), "", index_min_max(1, 3), index_multi(ns));
  Eigen::MatrixXd x_val_tmp = x_val;
  x_val_tmp.col(0).segment(0, 3) = x_val.col(1).segment(0, 3);
  x_val_tmp.col(2).segment(0, 3) = x_val.col(3).segment(0, 3);
  x_val_tmp.col(3).segment(0, 3) = x_val.col(0).segment(0, 3);
  EXPECT_MATRIX_EQ(x.val(), x_val_tmp);
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  Eigen::MatrixXd exp_adj = Eigen::MatrixXd::Ones(5, 5);
  exp_adj.col(1).segment(0, 3).array() = 2;
  exp_adj.col(2).segment(0, 3).array() = 0;
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
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
  assign(x, y, "", index_omni());
  EXPECT_MATRIX_EQ(y.val(), x.val());
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), y.val());
  EXPECT_MATRIX_EQ(x.adj(), Eigen::MatrixXd::Ones(5, 5));
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(5, 5));
  test_throw_invalid_arg(x, generate_linear_var_matrix(5, 6, 10), index_omni());
  test_throw_invalid_arg(x, generate_linear_var_matrix(5, 4, 10), index_omni());
  test_throw_invalid_arg(x, generate_linear_var_matrix(6, 5, 10), index_omni());
  test_throw_invalid_arg(x, generate_linear_var_matrix(4, 5, 10), index_omni());
}

TEST_F(VarAssign, omni_omni_matrix) {
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(5, 5, 10);
  assign(x, y, "", index_omni(), index_omni());
  EXPECT_MATRIX_EQ(y.val(), x.val());
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), y.val());
  EXPECT_MATRIX_EQ(x.adj(), Eigen::MatrixXd::Ones(5, 5));
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(5, 5));
  test_throw_invalid_arg(x, generate_linear_var_matrix(5, 6, 10), index_omni(),
                         index_omni());
  test_throw_invalid_arg(x, generate_linear_var_matrix(5, 4, 10), index_omni(),
                         index_omni());
  test_throw_invalid_arg(x, generate_linear_var_matrix(6, 5, 10), index_omni(),
                         index_omni());
  test_throw_invalid_arg(x, generate_linear_var_matrix(4, 5, 10), index_omni(),
                         index_omni());
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
  assign(x, y, "", index_uni(1), index_omni());
  EXPECT_MATRIX_EQ(y.val().row(0), x.val().row(0));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i != 0; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x, "lhs");
  EXPECT_MATRIX_EQ(y.adj(), Eigen::RowVectorXd::Ones(5));

  test_throw_invalid_arg(x,
                         generate_linear_var_vector<Eigen::RowVectorXd>(4, 10),
                         index_uni(1), index_omni());
  test_throw_invalid_arg(x,
                         generate_linear_var_vector<Eigen::RowVectorXd>(6, 10),
                         index_uni(1), index_omni());
  test_throw_out_of_range(x, y, index_uni(0), index_omni());
  test_throw_out_of_range(x, y, index_uni(6), index_omni());
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

  assign(x, y, "", index_min(2));
  EXPECT_MATRIX_EQ(x.val().bottomRows(2), y.val());
  sum(x).grad();
  // We don't assign to row 1
  auto check_i_x = [](int i) { return i > 0; };
  auto check_j_x = [](int j) { return true; };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(2, 4));
  test_throw_out_of_range(x, y, index_min(0));
  test_throw_out_of_range(x, y, index_min(4));
  test_throw_invalid_arg(x, y, index_min(1));
  var_value<MatrixXd> z(MatrixXd::Ones(1, 2));
  test_throw_invalid_arg(x, z, index_min(2));
}

TEST_F(VarAssign, minmax_min_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(3, 4);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(2, 3, 10);
  assign(x, y, "", index_min_max(2, 3), index_min(2));
  EXPECT_MATRIX_EQ(y.val(), x.val().block(1, 1, 2, 3));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return (i == 1 || i == 2); };
  auto check_j = [](int j) { return j > 0; };
  check_matrix_adjs(check_i, check_j, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), MatrixXd::Ones(2, 3));

  test_throw_out_of_range(x, y, index_min_max(0, 3), index_min(2));
  test_throw_out_of_range(x, y, index_min_max(2, 4), index_min(2));
  test_throw_out_of_range(x, y, index_min_max(2, 3), index_min(0));
  test_throw_out_of_range(x, y, index_min_max(2, 3), index_min(5));
  test_throw_invalid_arg(x, generate_linear_var_matrix(1, 3, 10),
                         index_min_max(2, 3), index_min(2));
  test_throw_invalid_arg(x, generate_linear_var_matrix(2, 5, 10),
                         index_min_max(2, 3), index_min(2));
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

  assign(x, y, "", index_max(2));
  EXPECT_MATRIX_EQ(x.val().topRows(2), y.val());
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i_x = [](int i) { return i < 2; };
  auto check_j_x = [](int j) { return true; };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(2, 4));
  test_throw_out_of_range(x, y, index_max(0));
  test_throw_out_of_range(x, y, index_max(4));
  test_throw_invalid_arg(x, y, index_max(1));
  var_value<MatrixXd> z(MatrixXd::Ones(1, 2));
  test_throw_invalid_arg(x, z, index_max(2));
}

TEST_F(VarAssign, min_max_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_matrix_adjs;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(3, 4);
  Eigen::MatrixXd x_val = x.val();
  auto y = generate_linear_var_matrix(2, 2, 10);

  assign(x, y, "", index_min(2), index_max(2));
  EXPECT_MATRIX_EQ(x.val().block(1, 0, 2, 2), y.val());
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  // We don't assign to row 1
  auto check_i_x = [](int i) { return i > 0; };
  auto check_j_x = [](int j) { return j < 2; };
  check_matrix_adjs(check_i_x, check_j_x, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(2, 2));
  test_throw_out_of_range(x, y, index_min(0), index_max(2));
  test_throw_out_of_range(x, y, index_min(5), index_max(2));
  test_throw_out_of_range(x, y, index_min(2), index_max(0));
  test_throw_out_of_range(x, y, index_min(2), index_max(5));
  test_throw_invalid_arg(x, y, index_min(2), index_max(1));
  var_value<MatrixXd> z(MatrixXd::Ones(1, 4));
  test_throw_invalid_arg(x, z, index_min(2), index_max(2));
  test_throw_invalid_arg(x, z, index_min(2), index_max(3));
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
    assign(x, x_rev.block(0, 0, ii, 5), "", index_min_max(1, ii));
    auto x_val_check = x.val().block(0, 0, ii, 5);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, 5);
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return true; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, 5), index_min_max(0, ii));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, 5),
                            index_min_max(1, ii + x.rows()));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, 5), index_min_max(2, ii));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, 4), index_min_max(1, ii));
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
    assign(x, x_rev.block(0, 0, ii, 5), "", index_min_max(ii, 1));
    auto x_val_check = x.val().block(0, 0, ii, 5);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, 5).colwise().reverse();
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return true; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, 5), index_min_max(ii, 0));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, 5),
                            index_min_max(ii + x.rows(), 1));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, 5), index_min_max(ii, 2));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, 4), index_min_max(1, ii));
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
    assign(x, x_rev.block(0, 0, ii, ii), "", index_min_max(1, ii),
           index_min_max(1, ii));
    auto x_val_check = x.val().block(0, 0, ii, ii);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, ii);
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return jj <= i; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(0, ii),
                            index_min_max(1, ii));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(1, ii),
                            index_min_max(0, ii));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii),
                            index_min_max(1, x.rows() + 1),
                            index_min_max(1, ii));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(1, ii),
                            index_min_max(1, x.rows() + 1));
    // We don't want to go out of bounds when making the eigen block.
    auto ii_range_high = ii == 5 ? 4 : ii + 1;
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii - 1, ii),
                           index_min_max(1, ii), index_min_max(1, ii));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii_range_high, ii),
                           index_min_max(1, ii), index_min_max(1, ii));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, ii - 1),
                           index_min_max(1, ii), index_min_max(1, ii));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, ii_range_high),
                           index_min_max(1, ii), index_min_max(1, ii));
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
    assign(x, x_rev.block(0, 0, ii, ii), "", index_min_max(1, ii),
           index_min_max(ii, 1));
    auto x_val_check = x.val().block(0, 0, ii, ii);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, ii).rowwise().reverse();
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return jj <= i; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(0, ii),
                            index_min_max(ii, 1));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(1, ii),
                            index_min_max(ii, 0));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii),
                            index_min_max(1, x.rows() + 1),
                            index_min_max(1, ii));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(1, ii),
                            index_min_max(1, x.rows() + 1));
    // We don't want to go out of bounds when making the eigen block.
    auto ii_range_high = ii == 5 ? 4 : ii + 1;
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii - 1, ii),
                           index_min_max(1, ii), index_min_max(ii, 1));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii_range_high, ii),
                           index_min_max(1, ii), index_min_max(ii, 1));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, ii - 1),
                           index_min_max(1, ii), index_min_max(ii, 1));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, ii_range_high),
                           index_min_max(1, ii), index_min_max(ii, 1));
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
    assign(x, x_rev.block(0, 0, ii, ii), "", index_min_max(ii, 1),
           index_min_max(1, ii));
    auto x_val_check = x.val().block(0, 0, ii, ii);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, ii).colwise().reverse();
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return jj <= i; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(ii, 0),
                            index_min_max(1, ii));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(ii, 1),
                            index_min_max(0, ii));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii),
                            index_min_max(x.rows() + 1, 1),
                            index_min_max(1, ii));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(ii, 1),
                            index_min_max(1, x.rows() + 1));
    // We don't want to go out of bounds when making the eigen block.
    auto ii_range_high = ii == 5 ? 4 : ii + 1;
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii - 1, ii),
                           index_min_max(ii, 1), index_min_max(1, ii));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii_range_high, ii),
                           index_min_max(ii, 1), index_min_max(1, ii));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, ii - 1),
                           index_min_max(ii, 1), index_min_max(1, ii));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, ii_range_high),
                           index_min_max(ii, 1), index_min_max(1, ii));
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
    assign(x, x_rev.block(0, 0, ii, ii), "", index_min_max(ii, 1),
           index_min_max(ii, 1));
    auto x_val_check = x.val().block(0, 0, ii, ii);
    auto x_rev_val_check = x_rev.val().block(0, 0, ii, ii).reverse();
    EXPECT_MATRIX_EQ(x_val_check, x_rev_val_check);
    sum(x).grad();
    auto check_i = [i](int kk) { return kk <= i; };
    auto check_j = [i](int jj) { return jj <= i; };
    check_matrix_adjs(check_i, check_j, x, "lhs", 0);
    check_matrix_adjs(check_i, check_j, x_rev, "rhs", 1);
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(ii, 0),
                            index_min_max(ii, 1));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(ii, 1),
                            index_min_max(ii, 0));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii),
                            index_min_max(x.rows() + 1, 1),
                            index_min_max(ii, 1));
    test_throw_out_of_range(x, x_rev.block(0, 0, ii, ii), index_min_max(ii, 1),
                            index_min_max(x.rows() + 1, 1));
    // We don't want to go out of bounds when making the eigen block.
    auto ii_range_high = ii == 5 ? 4 : ii + 1;
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii - 1, ii),
                           index_min_max(ii, 1), index_min_max(ii, 1));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii_range_high, ii),
                           index_min_max(ii, 1), index_min_max(ii, 1));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, ii - 1),
                           index_min_max(ii, 1), index_min_max(ii, 1));
    test_throw_invalid_arg(x, x_rev.block(0, 0, ii, ii_range_high),
                           index_min_max(ii, 1), index_min_max(ii, 1));
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
  assign(x, y, "", index_uni(2), index_min_max(2, 4));
  EXPECT_MATRIX_EQ(y.val().segment(0, 3), x.val().row(1).segment(1, 3));
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), x_val);
  auto check_i = [](int i) { return i == 1; };
  auto check_j = [](int j) { return (j > 0 && j < 4); };
  check_matrix_adjs(check_i, check_j, x, "lhs", 0);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::RowVectorXd::Ones(3));
  test_throw_out_of_range(x, y, index_uni(0), index_min_max(2, 4));
  test_throw_out_of_range(x, y, index_uni(6), index_min_max(2, 4));
  test_throw_out_of_range(x, y, index_uni(2), index_min_max(0, 2));
  test_throw_out_of_range(x, y, index_uni(2), index_min_max(1, 6));
  test_throw_invalid_arg(x,
                         generate_linear_var_vector<Eigen::RowVectorXd>(2, 10),
                         index_uni(2), index_min_max(2, 4));
  test_throw_invalid_arg(x,
                         generate_linear_var_vector<Eigen::RowVectorXd>(4, 10),
                         index_uni(2), index_min_max(2, 4));
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
  assign(x, y, "");
  EXPECT_MATRIX_EQ(y.val(), x.val());
  sum(x).grad();
  EXPECT_MATRIX_EQ(x.val(), y.val());
  EXPECT_MATRIX_EQ(x.adj(), Eigen::MatrixXd::Ones(5, 5));
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(5, 5));
}

namespace stan {
namespace model {
namespace test {

template <typename T1, typename I1, typename I2>
inline void assign_tester(T1&& x, const I1& idx1, const I2& idx2) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_matrix;
  auto multi1 = convert_to_multi(idx1, x, false);
  auto multi2 = convert_to_multi(idx2, x, true);
  var_value<std::decay_t<T1>> x1(x);
  var_value<std::decay_t<T1>> x2(x);
  Eigen::MatrixXd y_val
      = generate_linear_matrix(multi1.ns_.size(), multi2.ns_.size(), 10);
  var_value<Eigen::MatrixXd> y(y_val);
  assign(x1, y, "", idx1, idx2);
  assign(x2, y, "", multi1, multi2);
  EXPECT_MATRIX_EQ(x1.val(), x2.val());
  stan::math::sum(stan::math::add(x1, x2)).grad();
  // Since this just moves the pointer x1 omni is diff than
  // multi
  if (!std::is_same<I1, index_omni>::value
      && !std::is_same<I2, index_omni>::value) {
    EXPECT_MATRIX_EQ(x1.val(), x2.val());
    EXPECT_MATRIX_EQ(x1.adj(), x2.adj());
    EXPECT_MATRIX_EQ(y.adj(),
                     Eigen::MatrixXd::Constant(y.rows(), y.cols(), 2).eval());
  }
  stan::math::recover_memory();
}

template <typename T1, typename I1>
inline void assign_tester(T1&& x, const I1& idx1, index_uni idx2) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_vector;
  auto multi1 = convert_to_multi(idx1, x, false);
  auto multi2 = convert_to_multi(idx2, x, true);
  var_value<std::decay_t<T1>> x1(x);
  var_value<std::decay_t<T1>> x2(x);
  Eigen::VectorXd y_val
      = generate_linear_vector<Eigen::VectorXd>(multi1.ns_.size(), 10);
  var_value<Eigen::VectorXd> y(y_val);
  assign(x1, y, "", idx1, idx2);
  assign(x2, y, "", multi1, multi2);
  EXPECT_MATRIX_EQ(x1.val(), x2.val());
  stan::math::sum(stan::math::add(x1, x2)).grad();
  EXPECT_MATRIX_EQ(x1.val(), x2.val());
  EXPECT_MATRIX_EQ(x1.adj(), x2.adj());
  EXPECT_MATRIX_EQ(y.adj(), Eigen::VectorXd::Constant(y.size(), 2).eval());
  stan::math::recover_memory();
}

template <typename T1, typename I2>
inline void assign_tester(T1&& x, index_uni idx1, const I2& idx2) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_vector;
  auto multi1 = convert_to_multi(idx1, x, false);
  auto multi2 = convert_to_multi(idx2, x, true);
  var_value<std::decay_t<T1>> x1(x);
  var_value<std::decay_t<T1>> x2(x);
  Eigen::RowVectorXd y_val
      = generate_linear_vector<Eigen::RowVectorXd>(multi2.ns_.size(), 10);
  var_value<Eigen::RowVectorXd> y(y_val);
  assign(x1, y, "", idx1, idx2);
  assign(x2, y, "", multi1, multi2);
  EXPECT_MATRIX_EQ(x1.val(), x2.val());
  stan::math::sum(stan::math::add(x1, x2)).grad();
  EXPECT_MATRIX_EQ(x1.val(), x2.val());
  EXPECT_MATRIX_EQ(x1.adj(), x2.adj());
  EXPECT_MATRIX_EQ(y.adj(), Eigen::RowVectorXd::Constant(y.size(), 2).eval());
  stan::math::recover_memory();
}

template <typename T1>
inline void all_assign_tests(T1&& x) {
  std::vector<int> multi_ns{1, 2, 3};
  // uni
  // uni uni is explicitly tested and would otherwise need a specialization
  assign_tester(x, index_multi(multi_ns), index_uni(1));
  assign_tester(x, index_omni(), index_uni(1));
  assign_tester(x, index_min(2), index_uni(1));
  assign_tester(x, index_max(2), index_uni(1));
  assign_tester(x, index_min_max(1, 2), index_uni(1));

  // multi
  assign_tester(x, index_uni(1), index_multi(multi_ns));
  assign_tester(x, index_multi(multi_ns), index_multi(multi_ns));
  assign_tester(x, index_omni(), index_multi(multi_ns));
  assign_tester(x, index_min(2), index_multi(multi_ns));
  assign_tester(x, index_max(2), index_multi(multi_ns));
  assign_tester(x, index_min_max(1, 2), index_multi(multi_ns));

  // omni
  assign_tester(x, index_uni(1), index_omni());
  assign_tester(x, index_multi(multi_ns), index_omni());
  assign_tester(x, index_omni(), index_omni());
  assign_tester(x, index_min(2), index_omni());
  assign_tester(x, index_max(2), index_omni());
  assign_tester(x, index_min_max(1, 2), index_omni());

  // min
  assign_tester(x, index_uni(1), index_min(2));
  assign_tester(x, index_multi(multi_ns), index_min(2));
  assign_tester(x, index_omni(), index_min(2));
  assign_tester(x, index_min(2), index_min(2));
  assign_tester(x, index_max(2), index_min(2));
  assign_tester(x, index_min_max(1, 2), index_min(2));

  // max
  assign_tester(x, index_uni(1), index_max(2));
  assign_tester(x, index_multi(multi_ns), index_max(2));
  assign_tester(x, index_omni(), index_max(2));
  assign_tester(x, index_min(2), index_max(2));
  assign_tester(x, index_max(2), index_max(2));
  assign_tester(x, index_min_max(1, 2), index_max(2));

  // min_max
  assign_tester(x, index_uni(1), index_min_max(1, 2));
  assign_tester(x, index_multi(multi_ns), index_min_max(1, 2));
  assign_tester(x, index_omni(), index_min_max(1, 2));
  assign_tester(x, index_min(2), index_min_max(1, 2));
  assign_tester(x, index_max(2), index_min_max(1, 2));
  assign_tester(x, index_min_max(1, 2), index_min_max(1, 2));
}
}  // namespace test
}  // namespace model
}  // namespace stan

TEST_F(VarAssign, all_types) {
  using stan::model::test::all_assign_tests;
  using stan::model::test::generate_linear_matrix;
  Eigen::MatrixXd x = generate_linear_matrix(4, 4);
  all_assign_tests(x);
  Eigen::MatrixXd x_wide = generate_linear_matrix(5, 6);
  all_assign_tests(x_wide);
  Eigen::MatrixXd x_long = generate_linear_matrix(7, 4);
  all_assign_tests(x_long);
}
