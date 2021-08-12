#include <stan/model/indexing.hpp>
#include <stan/math/rev/fun/sum.hpp>
#include <stan/math/prim/fun/eval.hpp>
#include <test/unit/util.hpp>
#include <test/unit/model/indexing/util.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <vector>

using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_uni;
using stan::model::rvalue;

struct RvalueRev : public testing::Test {
  void SetUp() {
    // make sure memory's clean before starting each test
    stan::math::recover_memory();
  }
  void TearDown() {
    // make sure memory's clean before starting each test
    stan::math::recover_memory();
  }
};

template <typename C, typename... I>
void test_throw_out_of_range(C&& c, I&&... idxs) {
  EXPECT_THROW(stan::model::rvalue(c, "", idxs...), std::out_of_range);
}

template <typename T1, typename... I>
void test_throw_invalid_arg(T1&& c, I&&... idxs) {
  EXPECT_THROW(stan::model::rvalue(c, "", idxs...), std::invalid_argument);
}

TEST_F(RvalueRev, nil_vec) {
  using stan::math::var_value;
  Eigen::VectorXd x(3);
  x(0) = 1.1;
  x(1) = 2.2;
  x(2) = 3.3;
  var_value<Eigen::VectorXd> xv(x);
  var_value<Eigen::VectorXd> rx = rvalue(xv, "");
  EXPECT_EQ(3, rx.size());
  EXPECT_FLOAT_EQ(1.1, rx.val()(0));
  EXPECT_FLOAT_EQ(2.2, rx.val()(1));
  EXPECT_FLOAT_EQ(3.3, rx.val()(2));
}

// uni
TEST_F(RvalueRev, uni_nil_vec) {
  using stan::math::var_value;
  Eigen::VectorXd x_val(3);
  x_val(0) = 1.1;
  x_val(1) = 2.2;
  x_val(2) = 3.3;
  var_value<Eigen::VectorXd> x(x_val);

  for (size_t i = 0; i < x.size(); ++i) {
    auto x_uni = rvalue(x, "", index_uni(i + 1));
    EXPECT_EQ(x_val[i], x_uni.val());
    x_uni.grad();
    EXPECT_EQ(x_uni.adj(), x.adj()(i));
    stan::math::zero_adjoints();
  }

  test_throw_out_of_range(x, index_uni(-1));
  test_throw_out_of_range(x, index_uni(0));
  test_throw_out_of_range(x, index_uni(4));
}

// multi
template <typename T>
void test_multi_varvector() {
  using stan::math::var_value;
  T v(5);
  v << 0, 1, 2, 3, 4;
  var_value<T> rv(v);
  std::vector<int> ns{4, 2, 2, 1, 5, 2, 4};
  var_value<T> vi = rvalue(rv, "", index_multi(ns));
  EXPECT_EQ(7, vi.size());
  EXPECT_FLOAT_EQ(3.0, vi.val()(0));
  EXPECT_FLOAT_EQ(1.0, vi.val()(1));
  EXPECT_FLOAT_EQ(1.0, vi.val()(2));
  EXPECT_FLOAT_EQ(0.0, vi.val()(3));
  EXPECT_FLOAT_EQ(4.0, vi.val()(4));
  EXPECT_FLOAT_EQ(1.0, vi.val()(5));
  EXPECT_FLOAT_EQ(3.0, vi.val()(6));

  ns.push_back(0);
  test_throw_out_of_range(rv, index_multi(ns));

  ns[ns.size() - 1] = 15;
  test_throw_out_of_range(rv, index_multi(ns));
  stan::math::sum(vi).grad();
  for (int i = 0; i < vi.size(); ++i) {
    EXPECT_FLOAT_EQ(1, vi.adj()(i));
  }
  // counts are how many times they were accessed
  EXPECT_FLOAT_EQ(1.0, rv.adj()(0));
  EXPECT_FLOAT_EQ(3.0, rv.adj()(1));
  EXPECT_FLOAT_EQ(0.0, rv.adj()(2));
  EXPECT_FLOAT_EQ(2.0, rv.adj()(3));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(4));
}

TEST_F(RvalueRev, multi_vec) { test_multi_varvector<Eigen::VectorXd>(); }

TEST_F(RvalueRev, multi_rowvec) { test_multi_varvector<Eigen::RowVectorXd>(); }

// omni
template <typename T>
void test_omni_varvector() {
  using stan::math::var_value;
  T v(5);
  v << 0, 1, 2, 3, 4;
  var_value<T> rv(v);
  var_value<T> vi = rvalue(rv, "", index_omni());
  EXPECT_EQ(5, vi.size());
  EXPECT_FLOAT_EQ(0, vi.val()(0));
  EXPECT_FLOAT_EQ(2, vi.val()(2));
  EXPECT_FLOAT_EQ(4, vi.val()(4));

  stan::math::sum(vi).grad();
  for (int i = 0; i < vi.size(); ++i) {
    EXPECT_FLOAT_EQ(1, vi.adj()(i));
  }
  // counts are how many times they were accessed
  EXPECT_FLOAT_EQ(1.0, rv.adj()(0));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(1));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(2));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(3));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(4));
}

TEST_F(RvalueRev, omni_vec) { test_omni_varvector<Eigen::VectorXd>(); }

TEST_F(RvalueRev, omni_rowvec) { test_omni_varvector<Eigen::RowVectorXd>(); }

// min
template <typename T>
void test_min_varvector() {
  using stan::math::var_value;
  T v(5);
  v << 0, 1, 2, 3, 4;
  var_value<T> rv(v);
  var_value<T> vi = rvalue(rv, "", index_min(3));
  EXPECT_EQ(3, vi.size());
  EXPECT_FLOAT_EQ(2, vi.val()(0));
  EXPECT_FLOAT_EQ(4, vi.val()(2));
  test_throw_out_of_range(rv, index_min(0));

  stan::math::sum(vi).grad();
  for (int i = 0; i < vi.size(); ++i) {
    EXPECT_FLOAT_EQ(1, vi.adj()(i));
  }

  // counts are how many times they were accessed
  EXPECT_FLOAT_EQ(0.0, rv.adj()(0));
  EXPECT_FLOAT_EQ(0.0, rv.adj()(1));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(2));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(3));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(4));
}

TEST_F(RvalueRev, min_vec) { test_min_varvector<Eigen::VectorXd>(); }

TEST_F(RvalueRev, min_rowvec) { test_min_varvector<Eigen::RowVectorXd>(); }

// max
template <typename T>
void test_max_varvector() {
  using stan::math::var_value;
  T v(5);
  v << 0, 1, 2, 3, 4;
  var_value<T> rv(v);
  var_value<T> vi = rvalue(rv, "", index_max(3));
  EXPECT_EQ(3, vi.size());
  EXPECT_FLOAT_EQ(0, vi.val()(0));
  EXPECT_FLOAT_EQ(2, vi.val()(2));
  test_throw_out_of_range(rv, index_max(15));

  stan::math::sum(vi).grad();
  for (int i = 0; i < vi.size(); ++i) {
    EXPECT_FLOAT_EQ(1, vi.adj()(i));
  }
  // counts are how many times they were accessed
  EXPECT_FLOAT_EQ(1.0, rv.adj()(0));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(1));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(2));
  EXPECT_FLOAT_EQ(0.0, rv.adj()(3));
  EXPECT_FLOAT_EQ(0.0, rv.adj()(4));
}

TEST_F(RvalueRev, max_vec) { test_max_varvector<Eigen::VectorXd>(); }

TEST_F(RvalueRev, max_rowvec) { test_max_varvector<Eigen::RowVectorXd>(); }

// min_max
TEST_F(RvalueRev, min_max_vec) {
  using stan::math::sum;
  using stan::math::var_value;
  Eigen::VectorXd x_val(4);
  x_val(0) = 1.1;
  x_val(1) = 2.2;
  x_val(2) = 3.3;
  x_val(3) = 4.4;

  // min > max
  for (int mn = 0; mn < 4; ++mn) {
    for (int mx = mn; mx < 4; ++mx) {
      var_value<Eigen::VectorXd> x(x_val);
      var_value<Eigen::VectorXd> rx
          = rvalue(x, "", index_min_max(mn + 1, mx + 1));
      EXPECT_FLOAT_EQ(mx - mn + 1, rx.size());
      for (int n = mn; n <= mx; ++n) {
        EXPECT_FLOAT_EQ(x.val()[n], rx.val()[n - mn]);
      }
      sum(rx).grad();
      for (int n = mn; n <= mx; ++n) {
        EXPECT_FLOAT_EQ(x.adj()[n], rx.adj()[n - mn]);
      }
      stan::math::recover_memory();
    }
  }
  var_value<Eigen::VectorXd> x(x_val);
  test_throw_out_of_range(x, index_min_max(0, 2));
  test_throw_out_of_range(x, index_min_max(2, 5));
}

TEST_F(RvalueRev, negative_min_max_vec) {
  using stan::math::sum;
  using stan::math::var_value;
  Eigen::VectorXd x_val(4);
  x_val(0) = 1.1;
  x_val(1) = 2.2;
  x_val(2) = 3.3;
  x_val(3) = 4.4;

  // max > min
  for (int mn = 3; mn > -1; --mn) {
    for (int mx = mn; mx > -1; --mx) {
      var_value<Eigen::VectorXd> x(x_val);
      var_value<Eigen::VectorXd> rx
          = rvalue(x, "", index_min_max(mn + 1, mx + 1));
      EXPECT_FLOAT_EQ(mn - mx + 1, rx.size());
      for (int n = mn; n <= mx; ++n) {
        EXPECT_FLOAT_EQ(x.val()[n], rx.val()[n - mn]);
      }
      sum(rx).grad();
      for (int n = mn; n <= mx; ++n) {
        EXPECT_FLOAT_EQ(x.adj()[n], rx.adj()[n - mn]);
      }
      stan::math::recover_memory();
    }
  }
  var_value<Eigen::VectorXd> x(x_val);
  test_throw_out_of_range(x, index_min_max(2, 0));
  test_throw_out_of_range(x, index_min_max(5, 2));
}

auto make_std_varvec() {
  using stan::math::sum;
  using stan::math::var_value;
  Eigen::Matrix<double, -1, 1> xd0(3);
  xd0(0) = 0.0;
  xd0(1) = 0.1;
  xd0(2) = 0.2;
  var_value<Eigen::Matrix<double, -1, 1>> x0(xd0);

  Eigen::Matrix<double, -1, 1> xd1(3);
  xd1(0) = 1.0;
  xd1(1) = 1.1;
  xd1(2) = 1.2;
  var_value<Eigen::Matrix<double, -1, 1>> x1(xd1);

  Eigen::Matrix<double, -1, 1> xd2(3);
  xd2(0) = 2.0;
  xd2(1) = 2.1;
  xd2(2) = 2.2;
  var_value<Eigen::Matrix<double, -1, 1>> x2(xd2);
  std::vector<var_value<Eigen::Matrix<double, -1, 1>>> x;

  x.push_back(x0);
  x.push_back(x1);
  x.push_back(x2);
  return x;
}

template <typename Check1, typename Check2, typename StdVecVar>
void check_std_vec_adjs(Check1&& i_check, Check2&& j_check,
                        const StdVecVar& x) {
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    for (Eigen::Index j = 0; j < x[i].size(); ++j) {
      if (i_check(i)) {
        if (j_check(j)) {
          EXPECT_FLOAT_EQ(x[i].adj()[j], 1)
              << "Failed for (i, j): (" << i << ", " << j << ")";
        } else {
          EXPECT_FLOAT_EQ(x[i].adj()[j], 0)
              << "Failed for (i, j): (" << i << ", " << j << ")";
        }
      } else {
        EXPECT_FLOAT_EQ(x[i].adj()[j], 0)
            << "Failed for (i, j): (" << i << ", " << j << ")";
      }
    }
  }
}

TEST_F(RvalueRev, uni_stdvec_min_vec) {
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  auto x = make_std_varvec();
  var_value<Eigen::VectorXd> y = rvalue(x, "", index_uni(1), index_min(2));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(0.1, y.val()[0]);
  EXPECT_FLOAT_EQ(0.2, y.val()[1]);
  sum(y).grad();
  auto i_check = [](Eigen::Index i) { return i == 0; };
  auto j_check = [](Eigen::Index j) { return j > 0; };
  check_std_vec_adjs(i_check, j_check, x);
  test_throw_out_of_range(x, index_uni(0), index_min(2));
  test_throw_out_of_range(x, index_uni(1), index_min(0));
}
TEST_F(RvalueRev, uni_stdvec_max_vec) {
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  auto x = make_std_varvec();
  var_value<Eigen::VectorXd> y = rvalue(x, "", index_uni(2), index_max(2));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.0, y.val()[0]);
  EXPECT_FLOAT_EQ(1.1, y.val()[1]);
  sum(y).grad();
  auto i_check = [](Eigen::Index i) { return i == 1; };
  auto j_check = [](Eigen::Index j) { return j < 2; };
  check_std_vec_adjs(i_check, j_check, x);
  test_throw_out_of_range(x, index_uni(0), index_max(2));
  test_throw_out_of_range(x, index_uni(1), index_max(15));
}
TEST_F(RvalueRev, uni_stdvec_positive_minmax_vec) {
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  auto x = make_std_varvec();
  var_value<Eigen::VectorXd> y
      = rvalue(x, "", index_uni(2), index_min_max(2, 3));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.1, y.val()[0]);
  EXPECT_FLOAT_EQ(1.2, y.val()[1]);
  sum(y).grad();
  auto i_check = [](Eigen::Index i) { return i == 1; };
  auto j_check = [](Eigen::Index j) { return j == 1 || j == 2; };
  check_std_vec_adjs(i_check, j_check, x);
  test_throw_out_of_range(x, index_uni(0), index_min_max(2, 3));
  test_throw_out_of_range(x, index_uni(10), index_min_max(2, 15));
}
TEST_F(RvalueRev, uni_stdvec_negative_minmax_vec) {
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  auto x = make_std_varvec();
  var_value<Eigen::VectorXd> y
      = rvalue(x, "", index_uni(2), index_min_max(2, 1));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.1, y.val()[0]);
  sum(y).grad();
  auto i_check = [](Eigen::Index i) { return i == 1; };
  auto j_check = [](Eigen::Index j) { return j == 0 || j == 1; };
  check_std_vec_adjs(i_check, j_check, x);
  test_throw_out_of_range(x, index_uni(1), index_min_max(3, 0));
  test_throw_out_of_range(x, index_uni(1), index_min_max(15, 2));
}
TEST_F(RvalueRev, uni_stdvec_omni_vec) {
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  auto x = make_std_varvec();
  var_value<Eigen::VectorXd> y = rvalue(x, "", index_uni(3), index_omni());
  EXPECT_EQ(3, y.size());
  EXPECT_FLOAT_EQ(2.0, y.val()[0]);
  EXPECT_FLOAT_EQ(2.1, y.val()[1]);
  EXPECT_FLOAT_EQ(2.2, y.val()[2]);
  sum(y).grad();
  auto i_check = [](Eigen::Index i) { return i == 2; };
  auto j_check = [](Eigen::Index j) { return true; };
  check_std_vec_adjs(i_check, j_check, x);
  test_throw_out_of_range(x, index_uni(0), index_omni());
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
TEST_F(RvalueRev, uni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(4, 3);

  var_value<Eigen::RowVectorXd> y = rvalue(x, "", index_uni(1));
  EXPECT_EQ(3, y.size());
  EXPECT_MATRIX_EQ(y.val(), x.val().row(0));
  sum(y).grad();
  Eigen::RowVectorXd y_exp_adj = Eigen::RowVectorXd::Ones(3);
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(4, 3);
  x_exp_adj.row(0) += y_exp_adj;
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);

  test_throw_out_of_range(x, index_uni(0));
  test_throw_out_of_range(x, index_uni(15));
}

TEST_F(RvalueRev, uni_uni_mat) {
  using stan::model::test::check_adjs;
  Eigen::MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  for (int m = 0; m < 3; ++m) {
    for (int n = 0; n < 4; ++n) {
      stan::math::var_value<Eigen::MatrixXd> x(x_val);
      auto x_sub = rvalue(x, "", index_uni(m + 1), index_uni(n + 1));
      EXPECT_FLOAT_EQ(x.val()(m, n), x_sub.val());
      x_sub.grad();
      auto check_i = [m](int i) { return m == i; };
      auto check_j = [n](int j) { return n == j; };
      check_adjs(check_i, check_j, x);
      stan::math::recover_memory();
    }
  }
  stan::math::var_value<Eigen::MatrixXd> x(x_val);
  test_throw_out_of_range(x, index_uni(0), index_uni(1));
  test_throw_out_of_range(x, index_uni(10), index_uni(3));
  test_throw_out_of_range(x, index_uni(1), index_uni(0));
  test_throw_out_of_range(x, index_uni(1), index_uni(10));
}

TEST_F(RvalueRev, omni_uni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  var_value<Eigen::VectorXd> y = rvalue(x, "", index_omni(), index_uni(2));
  EXPECT_EQ(3, y.size());
  EXPECT_MATRIX_EQ(y.val(), x.val().col(1));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(3, 4);
  Eigen::VectorXd y_exp_adj = Eigen::VectorXd::Ones(3);
  x_exp_adj.col(1) += y_exp_adj;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);
  test_throw_out_of_range(x, index_omni(), index_uni(0));
  test_throw_out_of_range(x, index_omni(), index_uni(20));
}

TEST_F(RvalueRev, multi_uni_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::check_adjs;
  using stan::model::test::conditionally_generate_linear_var_matrix;
  using stan::model::test::conditionally_generate_linear_var_vector;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  std::vector<int> ns{3, 1, 1};
  var_value<Eigen::VectorXd> y = rvalue(x, "", index_multi(ns), index_uni(3));
  EXPECT_FLOAT_EQ(y.val()(0), x.val()(2, 2));
  EXPECT_FLOAT_EQ(y.val()(1), x.val()(0, 2));
  EXPECT_FLOAT_EQ(y.val()(2), x.val()(0, 2));
  sum(y).grad();
  EXPECT_MATRIX_EQ(y.adj(), Eigen::VectorXd::Ones(3).eval());
  Eigen::MatrixXd exp_adj = Eigen::MatrixXd::Zero(x.rows(), x.cols());
  exp_adj(0, 2) = 2;
  exp_adj(2, 2) = 1;
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_multi(ns), index_uni(3));
  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, index_multi(ns), index_uni(3));
  ns.push_back(2);
  test_throw_out_of_range(x, index_multi(ns), index_uni(3));
}

TEST_F(RvalueRev, min_uni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  var_value<Eigen::VectorXd> y = rvalue(x, "", index_min(2), index_uni(3));
  EXPECT_EQ(2, y.size());
  EXPECT_MATRIX_EQ(y.val(), x.val().col(2).segment(1, 2));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(3, 4);
  Eigen::VectorXd y_exp_adj = Eigen::VectorXd::Ones(2);
  x_exp_adj.val().col(2).segment(1, 2) += y_exp_adj;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);

  test_throw_out_of_range(x, index_min(0), index_uni(3));
  test_throw_out_of_range(x, index_min(20), index_uni(3));
  test_throw_out_of_range(x, index_min(2), index_uni(0));
  test_throw_out_of_range(x, index_min(2), index_uni(30));
}

TEST_F(RvalueRev, minmax_uni_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  var_value<Eigen::VectorXd> y
      = rvalue(x, "", index_min_max(2, 3), index_uni(4));
  EXPECT_MATRIX_EQ(y.val(), x.val().col(3).segment(1, 2));
  sum(y).grad();
  Eigen::MatrixXd exp_adj = Eigen::MatrixXd::Zero(3, 4);
  exp_adj.col(3).segment(1, 2).array() += 1;
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::VectorXd::Ones(2));
  test_throw_out_of_range(x, index_min_max(2, 3), index_uni(0));
  test_throw_out_of_range(x, index_min_max(2, 3), index_uni(5));
  test_throw_out_of_range(x, index_min_max(0, 1), index_uni(4));
  test_throw_out_of_range(x, index_min_max(1, 6), index_uni(4));
}

TEST_F(RvalueRev, negative_minmax_uni_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  var_value<Eigen::VectorXd> y
      = rvalue(x, "", index_min_max(3, 2), index_uni(4));
  EXPECT_MATRIX_EQ(y.val(), x.val().col(3).segment(1, 2).reverse());
  sum(y).grad();
  Eigen::MatrixXd exp_adj = Eigen::MatrixXd::Zero(3, 4);
  exp_adj.col(3).segment(1, 2).reverse().array() += 1;
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), Eigen::VectorXd::Ones(2));
  test_throw_out_of_range(x, index_min_max(3, 2), index_uni(0));
  test_throw_out_of_range(x, index_min_max(3, 2), index_uni(5));
  test_throw_out_of_range(x, index_min_max(1, 0), index_uni(4));
  test_throw_out_of_range(x, index_min_max(6, 1), index_uni(4));
}

// multi
TEST_F(RvalueRev, multi_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::var_value;

  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(4, 3);
  std::vector<int> row_idx{3, 4, 1, 4, 1, 4, 1};
  var_value<Eigen::MatrixXd> y = rvalue(x, "", index_multi(row_idx));
  EXPECT_FLOAT_EQ(7, y.rows());
  EXPECT_FLOAT_EQ(3, y.cols());
  Eigen::MatrixXd touch_count = Eigen::MatrixXd::Zero(4, 3);
  for (int j = 0; j < y.cols(); ++j) {
    for (int i = 0; i < row_idx.size(); ++i) {
      EXPECT_FLOAT_EQ(y.val()(i, j), x.val()(row_idx[i] - 1, j))
          << "Failed for i: (row_idx[i], j): (" << i << ": (" << row_idx[i]
          << ", " << j << ")";
      touch_count(row_idx[i] - 1, j) += 1;
    }
  }

  sum(y).grad();
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(7, 3).eval());
  for (int j = 0; j < x.cols(); ++j) {
    for (int i = 0; i < x.rows(); ++i) {
      EXPECT_FLOAT_EQ(x.adj()(i, j), touch_count.coeffRef(i, j))
          << "Failed for (i, j): (" << i << ", " << j << ")";
    }
  }
  row_idx.push_back(0);
  test_throw_out_of_range(x, index_multi(row_idx));

  row_idx[row_idx.size() - 1] = 15;
  test_throw_out_of_range(x, index_multi(row_idx));
}

TEST_F(RvalueRev, uni_multi_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;
  using stan::model::test::conditionally_generate_linear_var_vector;

  auto x = conditionally_generate_linear_var_matrix(5, 5);
  std::vector<int> ns{4, 1, 3, 3};
  var_value<Eigen::RowVectorXd> y
      = rvalue(x, "", index_uni(3), index_multi(ns));
  EXPECT_FLOAT_EQ(y.val()(0), x.val()(2, 3));
  EXPECT_FLOAT_EQ(y.val()(1), x.val()(2, 0));
  EXPECT_FLOAT_EQ(y.val()(2), x.val()(2, 2));
  EXPECT_FLOAT_EQ(y.val()(3), x.val()(2, 2));

  sum(y).grad();
  EXPECT_MATRIX_EQ(y.adj(), Eigen::RowVectorXd::Ones(4));
  Eigen::MatrixXd exp_adj = Eigen::MatrixXd::Zero(5, 5);
  exp_adj(2, 0) = 1;
  exp_adj(2, 2) = 2;
  exp_adj(2, 3) = 1;
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_uni(3), index_multi(ns));
  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, index_uni(3), index_multi(ns));
  ns.push_back(2);
  test_throw_out_of_range(x, index_uni(3), index_multi(ns));
}

TEST_F(RvalueRev, multi_multi_mat) {
  Eigen::MatrixXd x(4, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1,
      3.2, 3.3;
  stan::math::var_value<Eigen::MatrixXd> rx(x);

  std::vector<int> row_idx{3, 4, 1, 4, 1, 4, 1};
  std::vector<int> col_idx{4, 4, 3, 2, 1, 4, 1};

  stan::math::var_value<Eigen::MatrixXd> ry
      = rvalue(rx, "", index_multi(row_idx), index_multi(col_idx));
  EXPECT_EQ(7, ry.rows());
  EXPECT_EQ(7, ry.cols());
  // We use these to check the adjoints
  Eigen::MatrixXd touch_count = Eigen::MatrixXd::Zero(4, 4);

  for (int j = 0; j < col_idx.size(); ++j) {
    for (int i = 0; i < row_idx.size(); ++i) {
      EXPECT_FLOAT_EQ(ry.val()(i, j), rx.val()(row_idx[i] - 1, col_idx[j] - 1));
      touch_count(row_idx[i] - 1, col_idx[j] - 1) += 1;
    }
  }

  stan::math::sum(ry).grad();
  EXPECT_MATRIX_EQ(ry.adj(), Eigen::MatrixXd::Ones(7, 7));
  for (int j = 0; j < rx.cols(); ++j) {
    for (int i = 0; i < rx.rows(); ++i) {
      EXPECT_FLOAT_EQ(rx.adj()(i, j), touch_count.coeffRef(i, j));
    }
  }
  row_idx.push_back(19);
  row_idx.push_back(22);
  test_throw_out_of_range(rx, index_multi(row_idx), index_multi(col_idx));
  row_idx.pop_back();
  row_idx.pop_back();
  col_idx.push_back(19);
  col_idx.push_back(22);
  test_throw_out_of_range(rx, index_multi(row_idx), index_multi(col_idx));
}

TEST_F(RvalueRev, minmax_multi_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;
  using stan::model::test::conditionally_generate_linear_var_vector;

  auto x = conditionally_generate_linear_var_matrix(5, 5);
  std::vector<int> ns{4, 1, 3, 3};
  var_value<Eigen::MatrixXd> y
      = rvalue(x, "", index_min_max(1, 3), index_multi(ns));
  EXPECT_MATRIX_EQ(x.val().col(0).segment(0, 3), y.val().col(1));
  EXPECT_MATRIX_EQ(x.val().col(2).segment(0, 3), y.val().col(3));
  EXPECT_MATRIX_EQ(x.val().col(3).segment(0, 3), y.val().col(0));
  sum(y).grad();
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(3, 4));
  Eigen::MatrixXd exp_adj = Eigen::MatrixXd::Zero(5, 5);
  exp_adj.col(0).segment(0, 3).array() += 1;
  exp_adj.col(2).segment(0, 3).array() += 2;
  exp_adj.col(3).segment(0, 3).array() += 1;
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);

  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_min_max(1, 3), index_multi(ns));
  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, index_min_max(1, 3), index_multi(ns));
  ns.push_back(2);
  test_throw_out_of_range(x, index_min_max(1, 3), index_multi(ns));
}

// omni
TEST_F(RvalueRev, omni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(4, 3);
  var_value<Eigen::MatrixXd> y = rvalue(x, "", index_omni());
  EXPECT_EQ(4, y.rows());
  EXPECT_EQ(3, y.cols());
  EXPECT_MATRIX_EQ(y.val(), x.val());
  sum(y).grad();
  Eigen::MatrixXd exp_adj = Eigen::MatrixXd::Ones(4, 3);
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), exp_adj);
}

TEST_F(RvalueRev, uni_omni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  var_value<Eigen::RowVectorXd> y = rvalue(x, "", index_uni(2), index_omni());
  EXPECT_EQ(4, y.size());
  EXPECT_MATRIX_EQ(y.val(), x.val().row(1));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(3, 4);
  Eigen::RowVectorXd y_exp_adj = Eigen::RowVectorXd::Ones(4);
  x_exp_adj.row(1) += y_exp_adj;
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  test_throw_out_of_range(x, index_uni(0), index_omni());
  test_throw_out_of_range(x, index_uni(10), index_omni());
}

TEST_F(RvalueRev, omni_omni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  var_value<Eigen::MatrixXd> y = rvalue(x, "", index_omni(), index_omni());
  EXPECT_EQ(x.rows(), y.rows());
  EXPECT_EQ(x.cols(), y.cols());
  EXPECT_MATRIX_EQ(x.val(), y.val());
  sum(y).grad();
  Eigen::MatrixXd exp_adj = Eigen::MatrixXd::Ones(3, 4);
  EXPECT_MATRIX_EQ(x.adj(), exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), exp_adj);
}

// min
TEST_F(RvalueRev, min_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(4, 3);
  var_value<Eigen::MatrixXd> y = rvalue(x, "", index_min(3));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(3, y.cols());
  EXPECT_MATRIX_EQ(y.val(), x.val().block(2, 0, 2, 3));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(4, 3);
  Eigen::MatrixXd y_exp_adj = Eigen::MatrixXd::Ones(2, 3);
  x_exp_adj.block(2, 0, 2, 3) += y_exp_adj;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);
  test_throw_out_of_range(x, index_min(0));
  test_throw_out_of_range(x, index_min(12));
}

TEST_F(RvalueRev, uni_min_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  var_value<Eigen::RowVectorXd> y = rvalue(x, "", index_uni(3), index_min(2));
  EXPECT_EQ(3, y.size());
  EXPECT_MATRIX_EQ(y.val(), x.val().row(2).segment(1, 3));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(3, 4);
  Eigen::RowVectorXd y_exp_adj = Eigen::RowVectorXd::Ones(3);
  x_exp_adj.row(2).segment(1, 3) = y_exp_adj;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);
  test_throw_out_of_range(x, index_uni(0), index_min(2));
  test_throw_out_of_range(x, index_uni(12), index_min(2));
  test_throw_out_of_range(x, index_uni(1), index_min(0));
  test_throw_out_of_range(x, index_uni(1), index_min(12));
}

TEST_F(RvalueRev, min_min_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;
  auto x = conditionally_generate_linear_var_matrix(3, 4);
  stan::math::var_value<Eigen::MatrixXd> y
      = rvalue(x, "", index_min(2), index_min(3));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_MATRIX_EQ(y.val(), x.val().block(1, 2, 2, 2));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(3, 4);
  Eigen::MatrixXd y_exp_adj = Eigen::MatrixXd::Ones(2, 2);
  x_exp_adj.block(1, 2, 2, 2) = y_exp_adj;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);
  test_throw_out_of_range(x, index_min(0), index_min(3));
  test_throw_out_of_range(x, index_min(12), index_min(3));
  test_throw_out_of_range(x, index_min(2), index_min(0));
  test_throw_out_of_range(x, index_min(2), index_min(12));
}

TEST_F(RvalueRev, minmax_min_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  var_value<Eigen::MatrixXd> y
      = rvalue(x, "", index_min_max(2, 3), index_min(2));
  EXPECT_MATRIX_EQ(y.val(), x.val().block(1, 1, 2, 3));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(3, 4);
  Eigen::MatrixXd y_exp_adj = Eigen::MatrixXd::Ones(2, 3);
  x_exp_adj.block(1, 1, 2, 3) = y_exp_adj;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);
  test_throw_out_of_range(x, index_min_max(0, 3), index_min(2));
  test_throw_out_of_range(x, index_min_max(2, 7), index_min(2));
  test_throw_out_of_range(x, index_min_max(2, 3), index_min(0));
  test_throw_out_of_range(x, index_min_max(2, 3), index_min(7));
}

// max
TEST_F(RvalueRev, max_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(4, 3);
  var_value<Eigen::MatrixXd> y = rvalue(x, "", index_max(2));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(3, y.cols());
  EXPECT_MATRIX_EQ(y.val(), x.val().block(0, 0, 2, 3));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(4, 3);
  Eigen::MatrixXd y_exp_adj = Eigen::MatrixXd::Ones(2, 3);
  x_exp_adj.block(0, 0, 2, 3) = y_exp_adj;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);
  test_throw_out_of_range(x, index_max(0));
  test_throw_out_of_range(x, index_max(15));
}

TEST_F(RvalueRev, min_max_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  var_value<Eigen::MatrixXd> y = rvalue(x, "", index_min(2), index_max(2));
  EXPECT_MATRIX_EQ(y.val(), x.val().block(1, 0, 2, 2));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(3, 4);
  Eigen::MatrixXd y_exp_adj = Eigen::MatrixXd::Ones(2, 2);
  x_exp_adj.block(1, 0, 2, 2) = y_exp_adj;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);
  test_throw_out_of_range(x, index_min(0), index_max(2));
  test_throw_out_of_range(x, index_min(12), index_max(3));
  test_throw_out_of_range(x, index_min(2), index_max(0));
  test_throw_out_of_range(x, index_min(2), index_max(12));
}

// minmax
TEST_F(RvalueRev, positive_min_max_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(4, 3);
  var_value<Eigen::MatrixXd> y = rvalue(x, "", index_min_max(2, 3));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(3, y.cols());
  EXPECT_MATRIX_EQ(y.val(), x.val().block(1, 0, 2, 3));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(4, 3);
  Eigen::MatrixXd y_exp_adj = Eigen::MatrixXd::Ones(2, 3);
  x_exp_adj.block(1, 0, 2, 3) = y_exp_adj;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);

  test_throw_out_of_range(x, index_min_max(1, 15));
  test_throw_out_of_range(x, index_min_max(0, 2));
}

TEST_F(RvalueRev, negative_min_max_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;

  auto x = conditionally_generate_linear_var_matrix(3, 4);
  var_value<Eigen::MatrixXd> y = rvalue(x, "", index_min_max(3, 2));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(4, y.cols());
  EXPECT_MATRIX_EQ(y.val(), x.val().block(1, 0, 2, 4).colwise().reverse());
  sum(y).grad();
  Eigen::MatrixXd x_exp_adj = Eigen::MatrixXd::Zero(3, 4);
  Eigen::MatrixXd y_exp_adj = Eigen::MatrixXd::Ones(2, 4);
  x_exp_adj.block(1, 0, 2, 4).colwise().reverse() = y_exp_adj;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adj);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adj);
  test_throw_out_of_range(x, index_min_max(3, 0));
  test_throw_out_of_range(x, index_min_max(15, 2));
}

TEST_F(RvalueRev, positive_minmax_positive_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;
  Eigen::MatrixXd x_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    const int ii = i + 1;
    var_value<Eigen::MatrixXd> y
        = rvalue(x, "", index_min_max(1, ii), index_min_max(1, ii));
    EXPECT_MATRIX_EQ(y.val(), x.val().block(0, 0, ii, ii));
    sum(y).grad();
    Eigen::MatrixXd x_exp_adjs = Eigen::MatrixXd::Zero(5, 5);
    Eigen::MatrixXd y_exp_adjs = Eigen::MatrixXd::Ones(ii, ii);
    x_exp_adjs.block(0, 0, ii, ii) = y_exp_adjs;
    EXPECT_MATRIX_EQ(x.adj(), x_exp_adjs);
    EXPECT_MATRIX_EQ(y.adj(), y_exp_adjs);
    stan::math::recover_memory();
  }
}

TEST_F(RvalueRev, positive_minmax_negative_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using std::vector;
  Eigen::MatrixXd x_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    const int ii = i + 1;
    var_value<Eigen::MatrixXd> y
        = rvalue(x, "", index_min_max(1, ii), index_min_max(ii, 1));
    EXPECT_MATRIX_EQ(y.val(), x.val().block(0, 0, ii, ii).rowwise().reverse());
    sum(y).grad();
    Eigen::MatrixXd x_exp_adjs = Eigen::MatrixXd::Zero(5, 5);
    Eigen::MatrixXd y_exp_adjs = Eigen::MatrixXd::Ones(ii, ii);
    x_exp_adjs.block(0, 0, ii, ii) = y_exp_adjs;
    EXPECT_MATRIX_EQ(x.adj(), x_exp_adjs);
    EXPECT_MATRIX_EQ(y.adj(), y_exp_adjs);
    stan::math::recover_memory();
  }
}

TEST_F(RvalueRev, negative_minmax_positive_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    const int ii = i + 1;
    var_value<Eigen::MatrixXd> y
        = rvalue(x, "", index_min_max(ii, 1), index_min_max(1, ii));
    EXPECT_MATRIX_EQ(y.val(), x.val().block(0, 0, ii, ii).colwise().reverse());
    sum(y).grad();
    Eigen::MatrixXd x_exp_adjs = Eigen::MatrixXd::Zero(5, 5);
    Eigen::MatrixXd y_exp_adjs = Eigen::MatrixXd::Ones(ii, ii);
    x_exp_adjs.block(0, 0, ii, ii) = y_exp_adjs;
    EXPECT_MATRIX_EQ(x.adj(), x_exp_adjs);
    EXPECT_MATRIX_EQ(y.adj(), y_exp_adjs);
    stan::math::recover_memory();
  }
}

TEST_F(RvalueRev, negative_minmax_negative_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x_val(5, 5);
  for (int i = 0; i < x_val.size(); ++i) {
    x_val(i) = i;
  }

  for (int i = 0; i < x_val.rows(); ++i) {
    var_value<Eigen::MatrixXd> x(x_val);
    const int ii = i + 1;
    var_value<Eigen::MatrixXd> y
        = rvalue(x, "", index_min_max(ii, 1), index_min_max(ii, 1));
    EXPECT_MATRIX_EQ(
        y.val(),
        x.val().block(0, 0, ii, ii).colwise().reverse().rowwise().reverse());
    sum(y).grad();
    Eigen::MatrixXd x_exp_adjs = Eigen::MatrixXd::Zero(5, 5);
    Eigen::MatrixXd y_exp_adjs = Eigen::MatrixXd::Ones(ii, ii);
    x_exp_adjs.block(0, 0, ii, ii) = y_exp_adjs;
    EXPECT_MATRIX_EQ(x.adj(), x_exp_adjs);
    EXPECT_MATRIX_EQ(y.adj(), y_exp_adjs);
    stan::math::recover_memory();
  }
}

TEST_F(RvalueRev, uni_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;
  using stan::model::test::conditionally_generate_linear_var_vector;

  auto x = conditionally_generate_linear_var_matrix(5, 5);
  var_value<Eigen::RowVectorXd> y
      = rvalue(x, "", index_uni(2), index_min_max(2, 4));
  EXPECT_MATRIX_EQ(y.val(), x.val().row(1).segment(1, 3));
  sum(y).grad();
  Eigen::MatrixXd x_exp_adjs = Eigen::MatrixXd::Zero(5, 5);
  Eigen::RowVectorXd y_exp_adjs = Eigen::RowVectorXd::Ones(3);
  x_exp_adjs.row(1).segment(1, 3) = y_exp_adjs;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adjs);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adjs);
  test_throw_out_of_range(x, index_uni(0), index_min_max(2, 4));
  test_throw_out_of_range(x, index_uni(7), index_min_max(2, 4));
  test_throw_out_of_range(x, index_uni(2), index_min_max(0, 2));
  test_throw_out_of_range(x, index_uni(2), index_min_max(2, 12));
}

TEST_F(RvalueRev, uni_negative_minmax_matrix) {
  using stan::math::sum;
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;
  using stan::model::test::conditionally_generate_linear_var_vector;

  auto x = conditionally_generate_linear_var_matrix(5, 5);
  var_value<Eigen::RowVectorXd> y
      = rvalue(x, "", index_uni(2), index_min_max(4, 2));
  EXPECT_MATRIX_EQ(y.val(), x.val().row(1).segment(1, 3).reverse());
  sum(y).grad();
  Eigen::MatrixXd x_exp_adjs = Eigen::MatrixXd::Zero(5, 5);
  Eigen::RowVectorXd y_exp_adjs = Eigen::RowVectorXd::Ones(3);
  x_exp_adjs.row(1).segment(1, 3) = y_exp_adjs;
  EXPECT_MATRIX_EQ(x.adj(), x_exp_adjs);
  EXPECT_MATRIX_EQ(y.adj(), y_exp_adjs);
  test_throw_out_of_range(x, index_uni(0), index_min_max(4, 2));
  test_throw_out_of_range(x, index_uni(7), index_min_max(4, 2));
  test_throw_out_of_range(x, index_uni(2), index_min_max(2, 0));
  test_throw_out_of_range(x, index_uni(2), index_min_max(15, 0));
}

// nil only shows up as a single index
TEST_F(RvalueRev, nil_matrix) {
  using stan::math::var_value;
  using stan::model::test::conditionally_generate_linear_var_matrix;
  using stan::model::test::conditionally_generate_linear_var_vector;

  auto x = conditionally_generate_linear_var_matrix(5, 5);
  Eigen::MatrixXd x_val = x.val();
  auto y = rvalue(x, "");
  EXPECT_MATRIX_EQ(y.val(), x.val());
  sum(y).grad();
  EXPECT_MATRIX_EQ(x.val(), y.val());
  EXPECT_MATRIX_EQ(x.adj(), Eigen::MatrixXd::Ones(5, 5));
  EXPECT_MATRIX_EQ(y.adj(), Eigen::MatrixXd::Ones(5, 5));
}

namespace stan {
namespace model {
namespace test {

template <typename T1, typename I1, typename I2>
inline void rvalue_tester(T1&& x, const I1& idx1, const I2& idx2,
                          const char* name = "ANON", int depth = 0) {
  using stan::math::var_value;
  var_value<std::decay_t<T1>> x1(x);
  var_value<std::decay_t<T1>> x2(x);
  plain_type_t<decltype(rvalue(x1, "", idx1, idx2))> y1
      = rvalue(x1, "", idx1, idx2);
  plain_type_t<decltype(rvalue(x2, "", idx1, idx2))> y2
      = rvalue(x2, "", idx1, idx2);
  EXPECT_MATRIX_EQ(x1.val(), x2.val());
  EXPECT_MATRIX_EQ(y1.val(), y2.val());
  stan::math::sum(stan::math::add(y1, y2)).grad();
  EXPECT_MATRIX_EQ(x1.adj(), x2.adj());
  EXPECT_MATRIX_EQ(y1.adj(), y2.adj());
  stan::math::recover_memory();
}

template <typename T1>
inline void all_rvalue_tests(T1&& x) {
  std::vector<int> multi_ns{1, 2, 3};
  // uni
  // uni uni is explicitly tested and would otherwise need a specialization
  rvalue_tester(x, index_multi(multi_ns), index_uni(1));
  rvalue_tester(x, index_omni(), index_uni(1));
  rvalue_tester(x, index_min(2), index_uni(1));
  rvalue_tester(x, index_max(2), index_uni(1));
  rvalue_tester(x, index_min_max(1, 2), index_uni(1));

  // multi
  rvalue_tester(x, index_uni(1), index_multi(multi_ns));
  rvalue_tester(x, index_multi(multi_ns), index_multi(multi_ns));
  rvalue_tester(x, index_omni(), index_multi(multi_ns));
  rvalue_tester(x, index_min(2), index_multi(multi_ns));
  rvalue_tester(x, index_max(2), index_multi(multi_ns));
  rvalue_tester(x, index_min_max(1, 2), index_multi(multi_ns));

  // omni
  rvalue_tester(x, index_uni(1), index_omni());
  rvalue_tester(x, index_multi(multi_ns), index_omni());
  rvalue_tester(x, index_omni(), index_omni());
  rvalue_tester(x, index_min(2), index_omni());
  rvalue_tester(x, index_max(2), index_omni());
  rvalue_tester(x, index_min_max(1, 2), index_omni());

  // min
  rvalue_tester(x, index_uni(1), index_min(2));
  rvalue_tester(x, index_multi(multi_ns), index_min(2));
  rvalue_tester(x, index_omni(), index_min(2));
  rvalue_tester(x, index_min(2), index_min(2));
  rvalue_tester(x, index_max(2), index_min(2));
  rvalue_tester(x, index_min_max(1, 2), index_min(2));

  // max
  rvalue_tester(x, index_uni(1), index_max(2));
  rvalue_tester(x, index_multi(multi_ns), index_max(2));
  rvalue_tester(x, index_omni(), index_max(2));
  rvalue_tester(x, index_min(2), index_max(2));
  rvalue_tester(x, index_max(2), index_max(2));
  rvalue_tester(x, index_min_max(1, 2), index_max(2));

  // min_max
  rvalue_tester(x, index_uni(1), index_min_max(1, 2));
  rvalue_tester(x, index_multi(multi_ns), index_min_max(1, 2));
  rvalue_tester(x, index_omni(), index_min_max(1, 2));
  rvalue_tester(x, index_min(2), index_min_max(1, 2));
  rvalue_tester(x, index_max(2), index_min_max(1, 2));
  rvalue_tester(x, index_min_max(1, 2), index_min_max(1, 2));
}
}  // namespace test
}  // namespace model
}  // namespace stan

TEST_F(RvalueRev, all_types) {
  using stan::model::test::all_rvalue_tests;
  using stan::model::test::generate_linear_matrix;
  Eigen::MatrixXd x = generate_linear_matrix(4, 4);
  all_rvalue_tests(x);
  Eigen::MatrixXd x_wide = generate_linear_matrix(5, 6);
  all_rvalue_tests(x_wide);
  Eigen::MatrixXd x_long = generate_linear_matrix(7, 4);
  all_rvalue_tests(x_long);
}
