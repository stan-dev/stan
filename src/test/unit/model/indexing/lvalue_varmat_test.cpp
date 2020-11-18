#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/lvalue_varmat.hpp>
#include <stan/model/indexing/lvalue.hpp>
#include <stan/model/indexing/rvalue.hpp>
#include <stan/math/rev.hpp>
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

namespace stan {
namespace model {
namespace test {
auto generate_linear_matrix(Eigen::Index n, Eigen::Index m, double start = 0) {
  Eigen::Matrix<double, -1, -1> A(n, m);
  for (Eigen::Index i = 0; i < A.size(); ++i) {
    A(i) = i + start;
  }
  return A;
}
auto generate_linear_var_matrix(Eigen::Index n, Eigen::Index m,
                                double start = 0) {
  using ret_t = stan::math::var_value<Eigen::Matrix<double, -1, -1>>;
  return ret_t(generate_linear_matrix(n, m, start));
}
auto generate_linear_vector(Eigen::Index n, double start = 0) {
  Eigen::Matrix<double, -1, 1> A(n);
  for (Eigen::Index i = 0; i < A.size(); ++i) {
    A(i) = i + start;
  }
  return A;
}
template <bool ColVec = true>
auto generate_linear_var_vector(Eigen::Index n, double start = 0) {
  using ret_t = stan::math::var_value<Eigen::Matrix<double, -1, 1>>;
  return ret_t(generate_linear_vector(n, start));
}

template <>
auto generate_linear_var_vector<false>(Eigen::Index n, double start) {
  using ret_t = stan::math::var_value<Eigen::Matrix<double, 1, -1>>;
  return ret_t(generate_linear_vector(n, start).transpose());
}
}  // namespace test
}  // namespace model
}  // namespace stan

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

template <bool ColVec>
void test_uni_vec() {
  using stan::math::var_value;
  auto x = stan::model::test::generate_linear_var_vector<ColVec>(5);
  stan::math::var y(18);
  assign(x, index_list(index_uni(2)), y);
  EXPECT_FLOAT_EQ(y.val(), x.val()[1]);
  y.adj() = 100;
  x.adj()[1] = 10;
  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj(), 110);
  EXPECT_FLOAT_EQ(x.adj()[0], 0);
  test_throw_out_of_range(x, index_list(index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_uni(6)), y);
}

TEST_F(VarAssign, uni_vec) { test_uni_vec<true>(); }

TEST_F(VarAssign, uni_rowvec) { test_uni_vec<false>(); }

template <bool ColVec>
void test_multi_vec() {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<ColVec>(5);
  auto y = generate_linear_var_vector<ColVec>(3, 10);
  vector<int> ns;
  ns.push_back(2);
  ns.push_back(4);
  ns.push_back(2);
  std::conditional_t<ColVec, Eigen::VectorXd, Eigen::RowVectorXd> x_val
      = x.val();
  assign(x, index_list(index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[3]);
  EXPECT_FLOAT_EQ(y.val()[2], x.val()[1]);

  stan::arena_t<std::vector<int>> x_idx;
  stan::arena_t<std::vector<int>> y_idx;
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = ns.size() - 1; i >= 0; --i) {
    if (!stan::model::internal::check_duplicate(x_idx, ns[i] - 1)) {
      y_idx.push_back(i);
      x_idx.push_back(ns[i] - 1);
    }
  }
  stan::math::sum(x).grad();
  for (int i = 0; i < x.rows(); ++i) {
    EXPECT_FLOAT_EQ(x.val()(i), x_val.coeffRef(i));
    if (stan::model::internal::check_duplicate(x_idx, i)) {
      EXPECT_FLOAT_EQ(x.adj()(i), 0)
          << "Failed for \ni: " << i << " row_idx[i]: " << ns[i] << "\n";
    } else {
      EXPECT_FLOAT_EQ(x.adj()(i), 1)
          << "Failed for \ni: " << i << " row_idx[i]: " << ns[i] << "\n";
    }
  }
  for (int i = 0; i < y_idx.size(); ++i) {
    EXPECT_FLOAT_EQ(y.adj()(y_idx[i]), 1);
  }
  for (int i = 0; i < y.size(); ++i) {
    if (!stan::model::internal::check_duplicate(y_idx, i)) {
      EXPECT_FLOAT_EQ(y.adj()(i), 0);
    } else {
      EXPECT_FLOAT_EQ(y.adj()(i), 1);
    }
  }
}

TEST_F(VarAssign, multi_vec) { test_multi_vec<true>(); }

TEST_F(VarAssign, multi_rowvec) { test_multi_vec<false>(); }

template <bool ColVec>
void test_minmax_vec() {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<ColVec>(5);
  auto y = generate_linear_var_vector<ColVec>(2, 10);

  assign(x, index_list(index_min_max(2, 3)), y);
  EXPECT_FLOAT_EQ(y.val()[0], x.val()[1]);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[2]);
  y.adj()[0] = 10;
  y.adj()[1] = 20;
  x.adj()[1] = 30;
  x.adj()[2] = 40;
  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj()[0], 40);
  EXPECT_FLOAT_EQ(y.adj()[1], 60);
  EXPECT_FLOAT_EQ(x.adj()[1], 0);
  EXPECT_FLOAT_EQ(x.adj()[3], 0);
}

TEST_F(VarAssign, minmax_vec) { test_minmax_vec<true>(); }

TEST_F(VarAssign, minmax_rowvec) { test_minmax_vec<false>(); }

template <bool ColVec>
void test_max_vec() {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<ColVec>(5);
  auto y = generate_linear_var_vector<ColVec>(2, 10);

  assign(x, index_list(index_max(2)), y);
  EXPECT_FLOAT_EQ(y.val()[0], x.val()[0]);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[1]);
  y.adj()[0] = 10;
  y.adj()[1] = 20;
  x.adj()[0] = 30;
  x.adj()[1] = 40;
  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj()[0], 40);
  EXPECT_FLOAT_EQ(y.adj()[1], 60);
  EXPECT_FLOAT_EQ(x.adj()[0], 0);
  EXPECT_FLOAT_EQ(x.adj()[1], 0);
}

TEST_F(VarAssign, max_vec) { test_max_vec<true>(); }

TEST_F(VarAssign, max_rowvec) { test_max_vec<false>(); }

template <bool ColVec>
void test_min_vec() {
  using stan::value_type_t;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_vector;
  auto lhs = generate_linear_var_vector<ColVec>(5);
  auto rhs = generate_linear_var_vector<ColVec>(3, 10);
  value_type_t<decltype(lhs)> lhs_val(lhs.val());
  value_type_t<decltype(rhs)> rhs_val(rhs.val());
  assign(lhs, index_list(index_min(3)), rhs);
  EXPECT_FLOAT_EQ(lhs.val()(2), rhs.val()(0));
  EXPECT_FLOAT_EQ(lhs.val()(3), rhs.val()(1));
  EXPECT_FLOAT_EQ(lhs.val()(4), rhs.val()(2));
  sum(lhs).grad();
  for (Eigen::Index i = 0; i < lhs.size(); ++i) {
    EXPECT_FLOAT_EQ(lhs.val()(i), lhs_val(i))
        << "Failed for (i): (" << i << ")";
    if (i > 1) {
      EXPECT_FLOAT_EQ(lhs.adj()(i), 0) << "Failed for (i): (" << i << ")";
    } else {
      EXPECT_FLOAT_EQ(lhs.adj()(i), 1) << "Failed for (i): (" << i << ")";
    }
  }
  for (Eigen::Index i = 0; i < rhs.size(); ++i) {
    EXPECT_FLOAT_EQ(rhs.val()(i), rhs_val(i))
        << "Failed for (i): (" << i << ")";
    EXPECT_FLOAT_EQ(rhs.adj()(i), 1) << "Failed for (i): (" << i << ")";
  }
  test_throw_out_of_range(lhs, index_list(index_min(0)), rhs);
}
TEST_F(VarAssign, min_vec) { test_min_vec<true>(); }

TEST_F(VarAssign, min_rowvec) { test_min_vec<false>(); }

template <bool ColVec>
void test_omni_vec() {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector<ColVec>(5);
  auto y = generate_linear_var_vector<ColVec>(5, 10);

  assign(x, index_list(index_omni()), y);
  EXPECT_FLOAT_EQ(y.val()[0], x.val()[0]);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[1]);
  EXPECT_FLOAT_EQ(y.val()[2], x.val()[2]);
  EXPECT_FLOAT_EQ(y.val()[3], x.val()[3]);
  EXPECT_FLOAT_EQ(y.val()[4], x.val()[4]);
  x.adj()[0] = 50;
  x.adj()[1] = 40;
  x.adj()[2] = 30;
  x.adj()[3] = 20;
  x.adj()[4] = 10;
  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj()[0], 50);
  EXPECT_FLOAT_EQ(y.adj()[1], 40);
  EXPECT_FLOAT_EQ(y.adj()[2], 30);
  EXPECT_FLOAT_EQ(y.adj()[3], 20);
  EXPECT_FLOAT_EQ(y.adj()[4], 10);
  EXPECT_FLOAT_EQ(x.adj()[0], 50);
  EXPECT_FLOAT_EQ(x.adj()[1], 40);
  EXPECT_FLOAT_EQ(x.adj()[2], 30);
  EXPECT_FLOAT_EQ(x.adj()[3], 20);
  EXPECT_FLOAT_EQ(x.adj()[4], 10);
}

TEST_F(VarAssign, omni_vec) { test_omni_vec<true>(); }

TEST_F(VarAssign, omni_rowvec) { test_omni_vec<false>(); }

template <typename Vec>
void test_eigvec_var_uni_index_seg() {
  using stan::math::sum;
  using stan::math::var;
  using stan::math::var_value;
  Vec lhs_x_val(5);
  lhs_x_val << 0, 1, 2, 3, 4;
  var y = 13;
  var_value<Vec> lhs_x(lhs_x_val);
  assign(lhs_x.segment(0, 5), index_list(index_uni(3)), y);
  EXPECT_FLOAT_EQ(y.val(), lhs_x.val()(2));
  sum(lhs_x).grad();
  EXPECT_FLOAT_EQ(y.adj(), 1);
  for (Eigen::Index i = 0; i < lhs_x.size(); ++i) {
    EXPECT_FLOAT_EQ(lhs_x_val(i), lhs_x.val()(i));
    if (i == 2) {
      EXPECT_FLOAT_EQ(lhs_x.adj()(i), 0);
    } else {
      EXPECT_FLOAT_EQ(lhs_x.adj()(i), 1);
    }
  }
  EXPECT_FLOAT_EQ(y.adj(), 1);
  test_throw_out_of_range(lhs_x, index_list(index_uni(0)), y);
  test_throw_out_of_range(lhs_x, index_list(index_uni(6)), y);
}

TEST_F(VarAssign, uni_vec_segment) {
  test_eigvec_var_uni_index_seg<Eigen::VectorXd>();
}
TEST_F(VarAssign, uni_rowvec_segment) {
  test_eigvec_var_uni_index_seg<Eigen::RowVectorXd>();
}

TEST_F(VarAssign, positive_minmax_vec) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
  Eigen::VectorXd lhs_val(5);
  lhs_val << 1, 2, 3, 4, 5;
  Eigen::VectorXd rhs_val(4);
  rhs_val << 4, 3, 2, 1;
  var_value<Eigen::VectorXd> lhs(lhs_val);
  var_value<Eigen::VectorXd> rhs(rhs_val);

  assign(lhs, index_list(index_min_max(1, 4)), rhs);
  EXPECT_FLOAT_EQ(lhs.val()(0), 4);
  EXPECT_FLOAT_EQ(lhs.val()(1), 3);
  EXPECT_FLOAT_EQ(lhs.val()(2), 2);
  EXPECT_FLOAT_EQ(lhs.val()(3), 1);
  EXPECT_FLOAT_EQ(lhs.val()(4), 5);
  sum(lhs).grad();
  for (Eigen::Index i = 0; i < lhs.size(); ++i) {
    EXPECT_FLOAT_EQ(lhs_val(i), lhs.val()(i));
    if (i < lhs.size() - 1) {
      EXPECT_FLOAT_EQ(lhs.adj()(i), 0);
    } else {
      EXPECT_FLOAT_EQ(lhs.adj()(i), 1);
    }
  }
  for (Eigen::Index i = 0; i < rhs.size(); ++i) {
    EXPECT_FLOAT_EQ(rhs.adj()(i), 1);
  }
  test_throw_out_of_range(lhs, index_list(index_min_max(0, 3)), rhs);
  test_throw_out_of_range(lhs, index_list(index_min_max(1, 8)), rhs);
}
TEST_F(VarAssign, negative_minmax_vec) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
  Eigen::VectorXd lhs_val(5);
  lhs_val << 1, 2, 3, 4, 5;
  Eigen::VectorXd rhs_val(4);
  rhs_val << 1, 2, 3, 4;
  var_value<Eigen::VectorXd> lhs(lhs_val);
  var_value<Eigen::VectorXd> rhs(rhs_val);

  assign(lhs, index_list(index_min_max(4, 1)), rhs);
  EXPECT_FLOAT_EQ(lhs.val()(0), 4);
  EXPECT_FLOAT_EQ(lhs.val()(1), 3);
  EXPECT_FLOAT_EQ(lhs.val()(2), 2);
  EXPECT_FLOAT_EQ(lhs.val()(3), 1);
  EXPECT_FLOAT_EQ(lhs.val()(4), 5);
  sum(lhs).grad();
  for (Eigen::Index i = 0; i < lhs.size(); ++i) {
    EXPECT_FLOAT_EQ(lhs_val(i), lhs.val()(i));
    if (i < lhs.size() - 1) {
      EXPECT_FLOAT_EQ(lhs.adj()(i), 0);
    } else {
      EXPECT_FLOAT_EQ(lhs.adj()(i), 1);
    }
  }
  for (Eigen::Index i = 0; i < rhs.size(); ++i) {
    EXPECT_FLOAT_EQ(rhs.adj()(i), 1);
  }
  test_throw_out_of_range(lhs, index_list(index_min_max(3, 0)), rhs);
  test_throw_out_of_range(lhs, index_list(index_min_max(8, 1)), rhs);
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

// Uni Assigns
TEST_F(VarAssign, uni_matrix) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  auto y = generate_linear_var_vector<false>(5, 10);
  assign(x, index_list(index_uni(1)), y);
  EXPECT_FLOAT_EQ(y.val()(0, 0), x.val()(0, 0));
  EXPECT_FLOAT_EQ(y.val()(0, 1), x.val()(0, 1));
  EXPECT_FLOAT_EQ(y.val()(0, 2), x.val()(0, 2));
  EXPECT_FLOAT_EQ(y.val()(0, 3), x.val()(0, 3));
  EXPECT_FLOAT_EQ(y.val()(0, 4), x.val()(0, 4));
  y.adj()(0, 0) = 10;
  y.adj()(0, 1) = 20;
  y.adj()(0, 2) = 30;
  y.adj()(0, 3) = 40;
  y.adj()(0, 4) = 50;
  x.adj()(0, 0) = 50;
  x.adj()(0, 1) = 40;
  x.adj()(0, 2) = 30;
  x.adj()(0, 3) = 20;
  x.adj()(0, 4) = 10;
  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj()(0, 0), 60);
  EXPECT_FLOAT_EQ(y.adj()(0, 1), 60);
  EXPECT_FLOAT_EQ(y.adj()(0, 2), 60);
  EXPECT_FLOAT_EQ(y.adj()(0, 3), 60);
  EXPECT_FLOAT_EQ(y.adj()(0, 4), 60);
  EXPECT_FLOAT_EQ(x.adj()(0, 0), 0);
  EXPECT_FLOAT_EQ(x.adj()(0, 1), 0);
  EXPECT_FLOAT_EQ(x.adj()(0, 2), 0);
  EXPECT_FLOAT_EQ(x.adj()(0, 3), 0);
  EXPECT_FLOAT_EQ(x.adj()(0, 4), 0);
}

TEST_F(VarAssign, uni_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y_val(3);
  y_val << 10, 11, 12;
  var_value<MatrixXd> x(x_val);
  var_value<RowVectorXd> y(y_val);
  assign(x, index_list(index_uni(2), index_min_max(2, 4)), y);
  EXPECT_FLOAT_EQ(y_val(0), x.val()(1, 1));
  EXPECT_FLOAT_EQ(y_val(1), x.val()(1, 2));
  EXPECT_FLOAT_EQ(y_val(2), x.val()(1, 3));

  sum(x).grad();

  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      EXPECT_FLOAT_EQ(x.val()(i, j), x_val(i, j));
      if (i == 1) {
        if (j > 0 && j < 4) {
          EXPECT_FLOAT_EQ(x.adj()(i, j), 0)
              << "Failed for (i, j): (" << i << ", " << j << ")";
        }
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 1)
            << "Failed for (i, j): (" << i << ", " << j << ")";
      }
    }
  }
  for (Eigen::Index i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(y.adj()(i), 1) << "Failed for (i): (" << i << ")";
  }

  test_throw_out_of_range(x, index_list(index_uni(0), index_min_max(2, 4)), y);
  test_throw_out_of_range(x, index_list(index_uni(5), index_min_max(2, 4)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_min_max(0, 2)), y);
  test_throw_invalid_arg(x, index_list(index_uni(2), index_min_max(2, 5)), y);
}

TEST_F(VarAssign, uni_uni_matrix) {
  using stan::math::sum;
  using stan::math::var;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  var_value<MatrixXd> x(x_val);
  var y = 10.12;
  assign(x, index_list(index_uni(2), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y.val(), x.val()(1, 2));
  sum(x).grad();
  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      EXPECT_FLOAT_EQ(x.val()(i, j), x_val(i, j));
      if (i == 1) {
        if (j == 2) {
          EXPECT_FLOAT_EQ(x.adj()(i, j), 0)
              << "Failed for (i, j): (" << i << ", " << j << ")";
        }
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 1)
            << "Failed for (i, j): (" << i << ", " << j << ")";
      }
    }
  }
  EXPECT_FLOAT_EQ(y.adj(), 1);

  test_throw_out_of_range(x, index_list(index_uni(0), index_uni(3)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_uni(4), index_uni(3)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_uni(5)), y);
}

TEST_F(VarAssign, uni_multi_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y_val(3);
  y_val << 10, 11, 12;

  var_value<MatrixXd> x(x_val);
  var_value<RowVectorXd> y(y_val);

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(x, index_list(index_uni(3), index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y_val(0), x.val()(2, 3));
  EXPECT_FLOAT_EQ(y_val(1), x.val()(2, 0));
  EXPECT_FLOAT_EQ(y_val(2), x.val()(2, 2));

  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_list(index_uni(3), index_multi(ns)), y);

  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, index_list(index_uni(3), index_multi(ns)), y);

  ns.push_back(2);
  test_throw_invalid_arg(x, index_list(index_uni(3), index_multi(ns)), y);

  stan::math::sum(x).grad();
  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      EXPECT_FLOAT_EQ(x.val()(i, j), x_val(i, j))
          << "Failed for (i, j): (" << i << ", " << j << ")";
      if (i == 2) {
        if (j == 0 || j == 2 || j == 3) {
          EXPECT_FLOAT_EQ(x.adj()(i, j), 0)
              << "Failed for (i, j): (" << i << ", " << j << ")";
        }
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 1)
            << "Failed for (i, j): (" << i << ", " << j << ")";
      }
    }
  }
  for (Eigen::Index i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(y.adj()(i), 1) << "Failed for (i): (" << i << ")";
  }
}

// Multi assigns
TEST_F(VarAssign, uni_multi_duplicates_matrix) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_matrix(5, 5);
  auto y = generate_linear_var_vector<false>(7, 12);
  const int row_idx = 3;
  std::vector<int> col_idx;
  col_idx.push_back(1);
  col_idx.push_back(4);
  col_idx.push_back(4);
  col_idx.push_back(3);
  col_idx.push_back(2);
  col_idx.push_back(1);
  col_idx.push_back(5);
  Eigen::MatrixXd x_val = x.val();
  stan::arena_t<std::vector<int>> x_idx;
  stan::arena_t<std::vector<int>> y_idx;
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = col_idx.size() - 1; i >= 0; --i) {
    if (!stan::model::internal::check_duplicate(x_idx, col_idx[i] - 1)) {
      y_idx.push_back(i);
      x_idx.push_back(col_idx[i] - 1);
    }
  }
  assign(x, index_list(index_uni(row_idx), index_multi(col_idx)), y);
  // We use these to check the adjoints
  for (int i = 0; i < x_idx.size(); ++i) {
    EXPECT_FLOAT_EQ(x.val()(row_idx - 1, x_idx[i]), y.val()(y_idx[i]))
        << "Failed for \ni: " << i << "\nx_idx[i][0]: " << x_idx[i]
        << "\ny_idx[i]: " << y_idx[i];
  }
  stan::math::sum(x).grad();
  for (int j = 0; j < x.cols(); ++j) {
    EXPECT_FLOAT_EQ(x.val()(row_idx - 1, j), x_val.coeffRef(row_idx - 1, j));
    if (stan::model::internal::check_duplicate(x_idx, j)) {
      EXPECT_FLOAT_EQ(x.adj()(row_idx - 1, j), 0)
          << "Failed for \ni: " << j << " col_idx[i]: " << col_idx[j] << "\n";
    } else {
      EXPECT_FLOAT_EQ(x.adj()(row_idx - 1, j), 1)
          << "Failed for \ni: " << j << " col_idx[i]: " << col_idx[j] << "\n";
    }
  }
  for (int j = 0; j < y_idx.size(); ++j) {
    EXPECT_FLOAT_EQ(y.adj()(y_idx[j]), 1);
  }
  for (int j = 0; j < y.cols(); ++j) {
    if (!stan::model::internal::check_duplicate(y_idx, j)) {
      EXPECT_FLOAT_EQ(y.adj()(j), 0);
    } else {
      EXPECT_FLOAT_EQ(y.adj()(j), 1);
    }
  }
  test_throw_out_of_range(x, index_list(index_uni(0), index_multi(col_idx)), y);
  col_idx.pop_back();
  col_idx.pop_back();
  test_throw_invalid_arg(
      x, index_list(index_uni(row_idx), index_multi(col_idx)), y);
  col_idx.push_back(19);
  col_idx.push_back(22);
  test_throw_out_of_range(
      x, index_list(index_uni(row_idx), index_multi(col_idx)), y);
}

TEST_F(VarAssign, multi_uni_matrix) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  VectorXd y(2);
  y << 10, 11;
  assign(x, index_list(index_min_max(2, 3), index_uni(4)), y);
  EXPECT_FLOAT_EQ(y(0), x(1, 3));
  EXPECT_FLOAT_EQ(y(1), x(2, 3));

  test_throw_out_of_range(x, index_list(index_min_max(2, 3), index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_min_max(2, 3), index_uni(5)), y);
  test_throw_out_of_range(x, index_list(index_min_max(0, 1), index_uni(4)), y);
  test_throw_invalid_arg(x, index_list(index_min_max(1, 3), index_uni(4)), y);

  vector<int> ns;
  ns.push_back(3);
  ns.push_back(1);
  assign(x, index_list(index_multi(ns), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y(0), x(2, 2));
  EXPECT_FLOAT_EQ(y(1), x(0, 2));

  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_list(index_multi(ns), index_uni(3)), y);

  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, index_list(index_multi(ns), index_uni(3)), y);

  ns.push_back(2);
  test_throw_invalid_arg(x, index_list(index_multi(ns), index_uni(3)), y);
}

TEST_F(VarAssign, multi_multi_matrix) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_matrix;

  auto x = generate_linear_var_matrix(5, 5);
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
  Eigen::MatrixXd x_val = x.val();
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
  stan::math::sum(x).grad();
  for (int j = 0; j < x.cols(); ++j) {
    for (int i = 0; i < x.rows(); ++i) {
      EXPECT_FLOAT_EQ(x.val()(i, j), x_val.coeffRef(i, j));
      if (stan::model::internal::check_duplicate(x_idx, i, j)) {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 0)
            << "Failed for \ni: " << i << " row_idx[i]: " << row_idx[i] << "\n"
            << "j: " << j << " col_idx[i]: " << col_idx[j] << "\n";
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 1)
            << "Failed for \ni: " << i << " row_idx[i]: " << row_idx[i] << "\n"
            << "j: " << j << " col_idx[i]: " << col_idx[j] << "\n";
      }
    }
  }
  for (int i = 0; i < y_idx.size(); ++i) {
    EXPECT_FLOAT_EQ(y.adj()(y_idx[i][0], y_idx[i][1]), 1);
  }
  for (int j = 0; j < y.cols(); ++j) {
    for (int i = 0; i < y.rows(); ++i) {
      if (!stan::model::internal::check_duplicate(y_idx, i, j)) {
        EXPECT_FLOAT_EQ(y.adj()(i, j), 0);
      } else {
        EXPECT_FLOAT_EQ(y.adj()(i, j), 1);
      }
    }
  }
  col_idx.pop_back();
  col_idx.pop_back();
  test_throw_invalid_arg(
      x, index_list(index_multi(row_idx), index_multi(col_idx)), y);
  col_idx.push_back(19);
  col_idx.push_back(22);
  test_throw_out_of_range(
      x, index_list(index_multi(row_idx), index_multi(col_idx)), y);
  col_idx.pop_back();
  col_idx.pop_back();
  col_idx.push_back(1);
  col_idx.push_back(5);

  row_idx.pop_back();
  row_idx.pop_back();
  test_throw_invalid_arg(
      x, index_list(index_multi(row_idx), index_multi(col_idx)), y);
  row_idx.push_back(19);
  row_idx.push_back(22);
  test_throw_out_of_range(
      x, index_list(index_multi(row_idx), index_multi(col_idx)), y);
}

// Min assigns
TEST_F(VarAssign, min_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y_val(2, 4);
  y_val << 10.0, 10.1, 10.2, 10.3, 11.0, 11.1, 11.2, 11.3;
  var_value<MatrixXd> x(x_val);
  var_value<MatrixXd> y(y_val);

  assign(x, index_list(index_min(2)), y);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(y.val()(i, j), x.val()(i + 1, j));
    }
  }
  test_throw_invalid_arg(x, index_list(index_min(1)), y);

  MatrixXd z_val(1, 2);
  z_val << 10, 20;
  var_value<MatrixXd> z(z_val);
  test_throw_invalid_arg(x, index_list(index_min(1)), z);
  test_throw_invalid_arg(x, index_list(index_min(2)), z);
  sum(x).grad();
  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      EXPECT_FLOAT_EQ(x.val()(i, j), x_val(i, j));
      if (i > 0) {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 0)
            << "Failed for (i, j): (" << i << ", " << j << ")";
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 1)
            << "Failed for (i, j): (" << i << ", " << j << ")";
      }
    }
  }
  for (Eigen::Index i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(y.adj()(i), 1) << "Failed for (i): (" << i << ")";
  }
}

// minmax assigns
TEST_F(VarAssign, positive_minmax_positive_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
  Eigen::Matrix<double, -1, -1> x_val(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
    x_rev_val(i) = x_val.size() - i - 1;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    var_value<Eigen::MatrixXd> x_rev(x_rev_val);
    assign(x, index_list(index_min_max(1, i + 1), index_min_max(1, i + 1)),
           x_rev.block(0, 0, i + 1, i + 1));
    for (int kk = 0; kk < i; ++kk) {
      for (int jj = 0; jj < i; ++jj) {
        EXPECT_FLOAT_EQ(x.val()(kk, jj), x_rev.val()(kk, jj))
            << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
            << ")";
      }
    }
    sum(x).grad();
    for (int kk = 0; kk < x.rows(); ++kk) {
      for (int jj = 0; jj < x.cols(); ++jj) {
        EXPECT_FLOAT_EQ(x.val()(kk, jj), x_val(kk, jj));
        if (kk <= i && jj <= i) {
          EXPECT_FLOAT_EQ(x.adj()(kk, jj), 0)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
          EXPECT_FLOAT_EQ(x_rev.adj()(kk, jj), 1)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
        } else {
          EXPECT_FLOAT_EQ(x.adj()(kk, jj), 1)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
          EXPECT_FLOAT_EQ(x_rev.adj()(kk, jj), 0)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
        }
      }
    }
    stan::math::recover_memory();
  }
}

TEST_F(VarAssign, positive_minmax_negative_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
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
    assign(x, index_list(index_min_max(1, i + 1), index_min_max(i + 1, 1)),
           x_rev.block(0, 0, i + 1, i + 1));
    for (int kk = 0; kk < i; ++kk) {
      for (int jj = 0; jj < i; ++jj) {
        EXPECT_FLOAT_EQ(
            x.val()(kk, jj),
            x_rev.val().block(0, 0, i + 1, i + 1).rowwise().reverse()(kk, jj))
            << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
            << ")";
      }
    }
    sum(x).grad();
    for (int kk = 0; kk < x.rows(); ++kk) {
      for (int jj = 0; jj < x.cols(); ++jj) {
        EXPECT_FLOAT_EQ(x.val()(kk, jj), x_val(kk, jj));
        if (kk <= i && jj <= i) {
          EXPECT_FLOAT_EQ(x.adj()(kk, jj), 0)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
          EXPECT_FLOAT_EQ(x_rev.adj()(kk, jj), 1)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
        } else {
          EXPECT_FLOAT_EQ(x.adj()(kk, jj), 1)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
          EXPECT_FLOAT_EQ(x_rev.adj()(kk, jj), 0)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
        }
      }
    }
    stan::math::recover_memory();
  }
}

TEST_F(VarAssign, negative_minmax_positive_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
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
    assign(x, index_list(index_min_max(i + 1, 1), index_min_max(1, i + 1)),
           x_rev.block(0, 0, i + 1, i + 1));
    for (int kk = 0; kk < i; ++kk) {
      for (int jj = 0; jj < i; ++jj) {
        EXPECT_FLOAT_EQ(
            x.val()(kk, jj),
            x_rev.val().block(0, 0, i + 1, i + 1).colwise().reverse()(kk, jj))
            << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
            << ")";
      }
    }
    sum(x).grad();
    for (int kk = 0; kk < x.rows(); ++kk) {
      for (int jj = 0; jj < x.cols(); ++jj) {
        EXPECT_FLOAT_EQ(x.val()(kk, jj), x_val(kk, jj));
        if (kk <= i && jj <= i) {
          EXPECT_FLOAT_EQ(x.adj()(kk, jj), 0)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
          EXPECT_FLOAT_EQ(x_rev.adj()(kk, jj), 1)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
        } else {
          EXPECT_FLOAT_EQ(x.adj()(kk, jj), 1)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
          EXPECT_FLOAT_EQ(x_rev.adj()(kk, jj), 0)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
        }
      }
    }
    stan::math::recover_memory();
  }
}

TEST_F(VarAssign, negative_minmax_negative_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
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
    assign(x, index_list(index_min_max(i + 1, 1), index_min_max(i + 1, 1)),
           x_rev.block(0, 0, i + 1, i + 1));
    for (int kk = 0; kk < i; ++kk) {
      for (int jj = 0; jj < i; ++jj) {
        EXPECT_FLOAT_EQ(x.val()(kk, jj),
                        x_rev.val().block(0, 0, i + 1, i + 1).reverse()(kk, jj))
            << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
            << ")";
      }
    }
    sum(x).grad();
    for (int kk = 0; kk < x.rows(); ++kk) {
      for (int jj = 0; jj < x.cols(); ++jj) {
        EXPECT_FLOAT_EQ(x.val()(kk, jj), x_val(kk, jj));
        if (kk <= i && jj <= i) {
          EXPECT_FLOAT_EQ(x.adj()(kk, jj), 0)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
          EXPECT_FLOAT_EQ(x_rev.adj()(kk, jj), 1)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
        } else {
          EXPECT_FLOAT_EQ(x.adj()(kk, jj), 1)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
          EXPECT_FLOAT_EQ(x_rev.adj()(kk, jj), 0)
              << "Failed for i: (kk, jj): " << i << ": (" << kk << ", " << jj
              << ")";
        }
      }
    }
    stan::math::recover_memory();
  }
}

TEST_F(VarAssign, minmax_uni_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  VectorXd y_val(2);
  y_val << 10, 11;

  var_value<Eigen::MatrixXd> x(x_val);
  var_value<Eigen::VectorXd> y(y_val);

  assign(x, index_list(index_min_max(2, 3), index_uni(4)), y);
  EXPECT_FLOAT_EQ(y.val()(0), x.val()(1, 3));
  EXPECT_FLOAT_EQ(y.val()(1), x.val()(2, 3));
  sum(x).grad();
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x.val()(i), x_val(i));
  }
  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      if (j == 3) {
        if (i == 1 || i == 2) {
          EXPECT_FLOAT_EQ(x.adj()(i, j), 0);
        } else {
          EXPECT_FLOAT_EQ(x.adj()(i, j), 1);
        }
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 1);
      }
    }
  }
  EXPECT_FLOAT_EQ(y.adj()(0), 1);
  EXPECT_FLOAT_EQ(y.adj()(1), 1);

  test_throw_out_of_range(x, index_list(index_min_max(2, 3), index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_min_max(2, 3), index_uni(5)), y);
  test_throw_out_of_range(x, index_list(index_min_max(0, 1), index_uni(4)), y);
  test_throw_invalid_arg(x, index_list(index_min_max(1, 3), index_uni(4)), y);
}

TEST_F(VarAssign, minmax_min_matrix) {
  using stan::math::sum;
  using stan::math::var_value;

  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y_val(2, 3);
  y_val << 10, 11, 12, 20, 21, 22;
  var_value<Eigen::MatrixXd> x(x_val);
  var_value<Eigen::MatrixXd> y(y_val);

  assign(x, index_list(index_min_max(2, 3), index_min(2)), y);
  EXPECT_FLOAT_EQ(y.val()(0, 0), x.val()(1, 1));
  EXPECT_FLOAT_EQ(y.val()(0, 1), x.val()(1, 2));
  EXPECT_FLOAT_EQ(y.val()(0, 2), x.val()(1, 3));
  EXPECT_FLOAT_EQ(y.val()(1, 0), x.val()(2, 1));
  EXPECT_FLOAT_EQ(y.val()(1, 1), x.val()(2, 2));
  EXPECT_FLOAT_EQ(y.val()(1, 2), x.val()(2, 3));
  sum(x).grad();
  for (Eigen::Index i = 0; i < x_val.size(); ++i) {
    EXPECT_FLOAT_EQ(x.val()(i), x_val(i));
  }
  EXPECT_FLOAT_EQ(0, x.adj()(1, 1));
  EXPECT_FLOAT_EQ(0, x.adj()(1, 2));
  EXPECT_FLOAT_EQ(0, x.adj()(1, 3));
  EXPECT_FLOAT_EQ(0, x.adj()(2, 1));
  EXPECT_FLOAT_EQ(0, x.adj()(2, 2));
  EXPECT_FLOAT_EQ(0, x.adj()(2, 3));

  EXPECT_FLOAT_EQ(y.adj()(0, 0), 1);
  EXPECT_FLOAT_EQ(y.adj()(0, 1), 1);
  EXPECT_FLOAT_EQ(y.adj()(0, 2), 1);
  EXPECT_FLOAT_EQ(y.adj()(1, 0), 1);
  EXPECT_FLOAT_EQ(y.adj()(1, 1), 1);
  EXPECT_FLOAT_EQ(y.adj()(1, 2), 1);
}

TEST_F(VarAssign, minmax_min_block_matrix) {
  using stan::math::sum;
  using stan::math::var_value;

  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y_val(2, 3);
  y_val << 10, 11, 12, 20, 21, 22;
  var_value<Eigen::MatrixXd> x(x_val);
  var_value<Eigen::MatrixXd> y(y_val);
  assign(x.block(0, 0, 3, 3), index_list(index_min_max(2, 3), index_min(2)),
         y.block(0, 0, 2, 2));
  EXPECT_FLOAT_EQ(y.val()(0, 0), x.val()(1, 1));
  EXPECT_FLOAT_EQ(y.val()(0, 1), x.val()(1, 2));
  EXPECT_FLOAT_EQ(y.val()(1, 0), x.val()(2, 1));
  EXPECT_FLOAT_EQ(y.val()(1, 1), x.val()(2, 2));

  sum(x).grad();
  auto x_block = x.val().block(0, 0, 3, 3).eval();
  auto x_block_val = x_val.block(0, 0, 3, 3).eval();
  for (Eigen::Index i = 0; i < x_block.size(); ++i) {
    EXPECT_FLOAT_EQ(x_block(i), x_block_val(i));
  }
  EXPECT_FLOAT_EQ(x.adj()(1, 1), 0);
  EXPECT_FLOAT_EQ(x.adj()(1, 2), 0);
  EXPECT_FLOAT_EQ(x.adj()(2, 1), 0);
  EXPECT_FLOAT_EQ(x.adj()(2, 2), 0);

  EXPECT_FLOAT_EQ(y.adj()(0, 0), 1);
  EXPECT_FLOAT_EQ(y.adj()(0, 1), 1);
  EXPECT_FLOAT_EQ(y.adj()(1, 0), 1);
  EXPECT_FLOAT_EQ(y.adj()(1, 1), 1);

  test_throw_invalid_arg(x, index_list(index_min_max(2, 3), index_min(0)), y);
  test_throw_invalid_arg(x, index_list(index_min_max(2, 3), index_min(10)), y);
  test_throw_invalid_arg(x, index_list(index_min_max(1, 3), index_min(2)), y);
}

// omni assigns
TEST_F(VarAssign, omni_uni_matrix) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_vector;

  auto x = generate_linear_var_matrix(5, 5);
  auto y = generate_linear_var_vector<true>(5, 10);
  assign(x, index_list(index_omni(), index_uni(1)), y);
  EXPECT_FLOAT_EQ(y.val()(0), x.val()(0, 0));
  EXPECT_FLOAT_EQ(y.val()(1), x.val()(1, 0));
  EXPECT_FLOAT_EQ(y.val()(2), x.val()(2, 0));
  EXPECT_FLOAT_EQ(y.val()(3), x.val()(3, 0));
  EXPECT_FLOAT_EQ(y.val()(4), x.val()(4, 0));
  y.adj()(0) = 10;
  y.adj()(1) = 20;
  y.adj()(2) = 30;
  y.adj()(3) = 40;
  y.adj()(4) = 50;
  x.adj()(0, 0) = 50;
  x.adj()(1, 0) = 40;
  x.adj()(2, 0) = 30;
  x.adj()(3, 0) = 20;
  x.adj()(4, 0) = 10;
  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj()(0), 60);
  EXPECT_FLOAT_EQ(y.adj()(0), 60);
  EXPECT_FLOAT_EQ(y.adj()(0), 60);
  EXPECT_FLOAT_EQ(y.adj()(0), 60);
  EXPECT_FLOAT_EQ(y.adj()(0), 60);
  EXPECT_FLOAT_EQ(x.adj()(0, 0), 0);
  EXPECT_FLOAT_EQ(x.adj()(1, 0), 0);
  EXPECT_FLOAT_EQ(x.adj()(2, 0), 0);
  EXPECT_FLOAT_EQ(x.adj()(3, 0), 0);
  EXPECT_FLOAT_EQ(x.adj()(4, 0), 0);
}
