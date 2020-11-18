#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/rvalue.hpp>
#include <stan/model/indexing/rvalue_varmat.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>

using stan::model::cons_index_list;
using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_uni;
using stan::model::nil_index_list;

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

template <typename C, typename I>
void test_out_of_range(C&& c, I&& idxs) {
  EXPECT_THROW(stan::model::rvalue(c, idxs), std::out_of_range);
}

TEST_F(RvalueRev, nil_vec) {
  using stan::math::var_value;
  Eigen::VectorXd x(3);
  x(0) = 1.1;
  x(1) = 2.2;
  x(2) = 3.3;
  var_value<Eigen::VectorXd> xv(x);
  var_value<Eigen::VectorXd> rx = rvalue(xv, nil_index_list());
  EXPECT_EQ(3, rx.size());
  EXPECT_FLOAT_EQ(1.1, rx.val()(0));
  EXPECT_FLOAT_EQ(2.2, rx.val()(1));
  EXPECT_FLOAT_EQ(3.3, rx.val()(2));
}

TEST_F(RvalueRev, uni_nil_vec) {
  using stan::math::var_value;
  Eigen::VectorXd x_val(3);
  x_val(0) = 1.1;
  x_val(1) = 2.2;
  x_val(2) = 3.3;
  var_value<Eigen::VectorXd> x(x_val);

  for (size_t i = 0; i < x.size(); ++i) {
    auto x_uni = rvalue(x, index_list(index_uni(i + 1)));
    EXPECT_EQ(x_val[i], x_uni.val());
    x_uni.grad();
    EXPECT_EQ(x_uni.adj(), x.adj()(i));
    stan::math::zero_adjoints();
  }

  test_out_of_range(x, index_list(index_uni(-1)));
  test_out_of_range(x, index_list(index_uni(0)));
  test_out_of_range(x, index_list(index_uni(4)));
}

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
          = rvalue(x, index_list(index_min_max(mn + 1, mx + 1)));
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
  test_out_of_range(x, index_list(index_min_max(0, 2)));
  test_out_of_range(x, index_list(index_min_max(2, 5)));
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
          = rvalue(x, index_list(index_min_max(mn + 1, mx + 1)));
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
  test_out_of_range(x, index_list(index_min_max(2, 0)));
  test_out_of_range(x, index_list(index_min_max(5, 2)));
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
  var_value<VectorXd> y = rvalue(x, index_list(index_uni(1), index_min(2)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(0.1, y.val()[0]);
  EXPECT_FLOAT_EQ(0.2, y.val()[1]);
  sum(y).grad();
  auto i_check = [](Eigen::Index i) { return i == 0; };
  auto j_check = [](Eigen::Index j) { return j > 0; };
  check_std_vec_adjs(i_check, j_check, x);
  test_out_of_range(x, index_list(index_uni(0), index_min(2)));
  test_out_of_range(x, index_list(index_uni(1), index_min(0)));
}
TEST_F(RvalueRev, uni_stdvec_max_vec) {
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  auto x = make_std_varvec();
  var_value<VectorXd> y = rvalue(x, index_list(index_uni(2), index_max(2)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.0, y.val()[0]);
  EXPECT_FLOAT_EQ(1.1, y.val()[1]);
  sum(y).grad();
  auto i_check = [](Eigen::Index i) { return i == 1; };
  auto j_check = [](Eigen::Index j) { return j < 2; };
  check_std_vec_adjs(i_check, j_check, x);
  test_out_of_range(x, index_list(index_uni(0), index_max(2)));
  test_out_of_range(x, index_list(index_uni(1), index_max(15)));
}
TEST_F(RvalueRev, uni_stdvec_positive_minmax_vec) {
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  auto x = make_std_varvec();
  var_value<VectorXd> y
      = rvalue(x, index_list(index_uni(2), index_min_max(2, 3)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.1, y.val()[0]);
  EXPECT_FLOAT_EQ(1.2, y.val()[1]);
  sum(y).grad();
  auto i_check = [](Eigen::Index i) { return i == 1; };
  auto j_check = [](Eigen::Index j) { return j == 1 || j == 2; };
  check_std_vec_adjs(i_check, j_check, x);
  test_out_of_range(x, index_list(index_uni(0), index_min_max(2, 3)));
  test_out_of_range(x, index_list(index_uni(10), index_min_max(2, 15)));
}
TEST_F(RvalueRev, uni_stdvec_negative_minmax_vec) {
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  auto x = make_std_varvec();
  var_value<VectorXd> y
      = rvalue(x, index_list(index_uni(2), index_min_max(2, 1)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.1, y.val()[0]);
  sum(y).grad();
  auto i_check = [](Eigen::Index i) { return i == 1; };
  auto j_check = [](Eigen::Index j) { return j == 0 || j == 1; };
  check_std_vec_adjs(i_check, j_check, x);
  test_out_of_range(x, index_list(index_uni(1), index_min_max(3, 0)));
  test_out_of_range(x, index_list(index_uni(1), index_min_max(15, 2)));
}
TEST_F(RvalueRev, uni_stdvec_omni_vec) {
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  auto x = make_std_varvec();
  var_value<VectorXd> y = rvalue(x, index_list(index_uni(3), index_omni()));
  EXPECT_EQ(3, y.size());
  EXPECT_FLOAT_EQ(2.0, y.val()[0]);
  EXPECT_FLOAT_EQ(2.1, y.val()[1]);
  EXPECT_FLOAT_EQ(2.2, y.val()[2]);
  sum(y).grad();
  auto i_check = [](Eigen::Index i) { return i == 2; };
  auto j_check = [](Eigen::Index j) { return true; };
  check_std_vec_adjs(i_check, j_check, x);
  test_out_of_range(x, index_list(index_uni(0), index_omni()));
}

template <typename T>
void test_omni_varvector() {
  using stan::math::var_value;
  T v(5);
  v << 0, 1, 2, 3, 4;
  var_value<T> rv(v);
  var_value<T> vi = rvalue(rv, index_list(index_omni()));
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
  var_value<T> vi = rvalue(rv, index_list(index_min(3)));
  EXPECT_EQ(3, vi.size());
  EXPECT_FLOAT_EQ(2, vi.val()(0));
  EXPECT_FLOAT_EQ(4, vi.val()(2));
  test_out_of_range(rv, index_list(index_min(0)));

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
  var_value<T> vi = rvalue(rv, index_list(index_max(3)));
  EXPECT_EQ(3, vi.size());
  EXPECT_FLOAT_EQ(0, vi.val()(0));
  EXPECT_FLOAT_EQ(2, vi.val()(2));
  test_out_of_range(rv, index_list(index_max(15)));

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

// positive minmax
template <typename T>
void test_positive_minmax_varvector() {
  using stan::math::var_value;
  T v(5);
  v << 0, 1, 2, 3, 4;
  var_value<T> rv(v);
  var_value<T> vi = rvalue(rv, index_list(index_min_max(2, 4)));
  EXPECT_EQ(3, vi.size());
  EXPECT_FLOAT_EQ(1, vi.val()(0));
  EXPECT_FLOAT_EQ(3, vi.val()(2));
  test_out_of_range(rv, index_list(index_min_max(0, 4)));
  test_out_of_range(rv, index_list(index_min_max(2, 15)));

  stan::math::sum(vi).grad();
  for (int i = 0; i < vi.size(); ++i) {
    EXPECT_FLOAT_EQ(1, vi.adj()(i));
  }
  // counts are how many times they were accessed
  EXPECT_FLOAT_EQ(0.0, rv.adj()(0));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(1));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(2));
  EXPECT_FLOAT_EQ(1.0, rv.adj()(3));
  EXPECT_FLOAT_EQ(0.0, rv.adj()(4));
}

TEST_F(RvalueRev, positive_minmax_vec) {
  test_positive_minmax_varvector<Eigen::VectorXd>();
}

TEST_F(RvalueRev, positive_minmax_rowvec) {
  test_positive_minmax_varvector<Eigen::RowVectorXd>();
}

// multi
template <typename T>
void test_multi_varvector() {
  using stan::math::var_value;
  T v(5);
  v << 0, 1, 2, 3, 4;
  var_value<T> rv(v);
  std::vector<int> ns;
  ns.push_back(4);
  ns.push_back(2);
  ns.push_back(2);
  ns.push_back(1);
  ns.push_back(5);
  ns.push_back(2);
  ns.push_back(4);
  var_value<T> vi = rvalue(rv, index_list(index_multi(ns)));
  EXPECT_EQ(7, vi.size());
  EXPECT_FLOAT_EQ(3.0, vi.val()(0));
  EXPECT_FLOAT_EQ(1.0, vi.val()(1));
  EXPECT_FLOAT_EQ(1.0, vi.val()(2));
  EXPECT_FLOAT_EQ(0.0, vi.val()(3));
  EXPECT_FLOAT_EQ(4.0, vi.val()(4));
  EXPECT_FLOAT_EQ(1.0, vi.val()(5));
  EXPECT_FLOAT_EQ(3.0, vi.val()(6));

  ns.push_back(0);
  test_out_of_range(rv, index_list(index_multi(ns)));

  ns[ns.size() - 1] = 15;
  test_out_of_range(rv, index_list(index_multi(ns)));
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

template <typename Check1, typename Check2, typename VarMat>
void check_matrix_adjs(Check1&& i_check, Check2&& j_check, const VarMat& x) {
  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      if (i_check(i)) {
        if (j_check(j)) {
          EXPECT_FLOAT_EQ(x.adj()(i, j), 1)
              << "Failed for (i, j): (" << i << ", " << j << ")";
        } else {
          EXPECT_FLOAT_EQ(x.adj()(i, j), 0)
              << "Failed for (i, j): (" << i << ", " << j << ")";
        }
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 0)
            << "Failed for (i, j): (" << i << ", " << j << ")";
      }
    }
  }
}

TEST_F(RvalueRev, uni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;

  MatrixXd m(4, 3);
  m << 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2;
  var_value<MatrixXd> x(m);
  var_value<RowVectorXd> v = rvalue(x, index_list(index_uni(1)));
  EXPECT_EQ(3, v.size());
  EXPECT_FLOAT_EQ(0.0, v.val()(0));
  EXPECT_FLOAT_EQ(0.1, v.val()(1));
  EXPECT_FLOAT_EQ(0.2, v.val()(2));
  sum(v).grad();
  auto check_i = [](int i) { return i == 0; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x);

  test_out_of_range(x, index_list(index_uni(0)));
  test_out_of_range(x, index_list(index_uni(15)));
}

TEST_F(RvalueRev, uni_uni_mat) {
  Eigen::MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  for (int m = 0; m < 3; ++m) {
    for (int n = 0; n < 4; ++n) {
      stan::math::var_value<Eigen::MatrixXd> x(x_val);
      auto x_sub = rvalue(x, index_list(index_uni(m + 1), index_uni(n + 1)));
      EXPECT_FLOAT_EQ(m + n / 10.0, x_sub.val());
      x_sub.grad();
      auto check_i = [m](int i) { return m == i; };
      auto check_j = [n](int j) { return n == j; };
      check_matrix_adjs(check_i, check_j, x);
      stan::math::recover_memory();
    }
  }
  stan::math::var_value<Eigen::MatrixXd> x(x_val);
  test_out_of_range(x, index_list(index_uni(0), index_uni(1)));
  test_out_of_range(x, index_list(index_uni(0), index_uni(10)));
  test_out_of_range(x, index_list(index_uni(1), index_uni(0)));
  test_out_of_range(x, index_list(index_uni(1), index_uni(10)));
}

TEST_F(RvalueRev, uni_omni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  var_value<MatrixXd> x(x_val);
  var_value<RowVectorXd> y = rvalue(x, index_list(index_uni(2), index_omni()));
  EXPECT_EQ(4, y.size());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(y.val()(i), x.val()(1, i));
  }
  sum(y).grad();
  auto check_i = [](int i) { return 1 == i; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x);

  test_out_of_range(x, index_list(index_uni(0), index_omni()));
  test_out_of_range(x, index_list(index_uni(10), index_omni()));
}

TEST_F(RvalueRev, uni_min_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  var_value<MatrixXd> x(x_val);
  var_value<RowVectorXd> y = rvalue(x, index_list(index_uni(3), index_min(2)));
  EXPECT_EQ(3, y.size());
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(y.val()(i), x.val()(2, i + 1));
  }
  sum(y).grad();
  auto check_i = [](int i) { return 2 == i; };
  auto check_j = [](int j) { return j > 0; };
  check_matrix_adjs(check_i, check_j, x);
  test_out_of_range(x, index_list(index_uni(0), index_min(2)));
  test_out_of_range(x, index_list(index_uni(1), index_min(0)));
}

TEST_F(RvalueRev, min_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(4, 3);
  x_val << 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2;
  var_value<MatrixXd> x(x_val);
  var_value<MatrixXd> y = rvalue(x, index_list(index_min(3)));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(3, y.cols());
  EXPECT_FLOAT_EQ(2.0, y.val()(0, 0));
  EXPECT_FLOAT_EQ(2.1, y.val()(0, 1));
  EXPECT_FLOAT_EQ(2.2, y.val()(0, 2));
  EXPECT_FLOAT_EQ(3.0, y.val()(1, 0));
  EXPECT_FLOAT_EQ(3.1, y.val()(1, 1));
  EXPECT_FLOAT_EQ(3.2, y.val()(1, 2));
  sum(y).grad();
  auto check_i = [](int i) { return i > 1; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x);
  test_out_of_range(x, index_list(index_min(0)));
}

TEST_F(RvalueRev, min_uni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  var_value<MatrixXd> x(x_val);
  var_value<VectorXd> y = rvalue(x, index_list(index_min(2), index_uni(3)));
  EXPECT_EQ(2, y.size());
  for (int j = 0; j < 2; ++j) {
    EXPECT_EQ(1 + j + 0.2, y.val()(j));
  }
  sum(y).grad();
  auto check_i = [](int i) { return i > 0; };
  auto check_j = [](int j) { return j == 2; };
  check_matrix_adjs(check_i, check_j, x);

  test_out_of_range(x, index_list(index_min(0), index_uni(3)));
  test_out_of_range(x, index_list(index_min(2), index_uni(0)));
  test_out_of_range(x, index_list(index_min(2), index_uni(30)));
}

TEST_F(RvalueRev, min_min_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  stan::math::var_value<Eigen::MatrixXd> x(x_val);
  stan::math::var_value<Eigen::MatrixXd> y
      = rvalue(x, index_list(index_min(2), index_min(3)));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(2, y.cols());
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_FLOAT_EQ(i + 1 + (j + 2) / 10.0, y.val()(i, j));
    }
  }
  sum(y).grad();
  auto check_i = [](int i) { return i > 0; };
  auto check_j = [](int j) { return j > 1; };
  check_matrix_adjs(check_i, check_j, x);
  test_out_of_range(x, index_list(index_min(0), index_min(3)));
  test_out_of_range(x, index_list(index_min(2), index_min(0)));
}

TEST_F(RvalueRev, max_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(4, 3);
  x_val << 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2;
  var_value<MatrixXd> x(x_val);
  var_value<MatrixXd> y = rvalue(x, index_list(index_max(2)));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(3, y.cols());
  EXPECT_FLOAT_EQ(0.0, y.val()(0, 0));
  EXPECT_FLOAT_EQ(0.1, y.val()(0, 1));
  EXPECT_FLOAT_EQ(0.2, y.val()(0, 2));
  EXPECT_FLOAT_EQ(1.0, y.val()(1, 0));
  EXPECT_FLOAT_EQ(1.1, y.val()(1, 1));
  EXPECT_FLOAT_EQ(1.2, y.val()(1, 2));
  sum(y).grad();
  auto check_i = [](int i) { return i < 2; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x);
  test_out_of_range(x, index_list(index_max(15)));
}

TEST_F(RvalueRev, positive_min_max_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(4, 3);
  x_val << 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2;
  var_value<MatrixXd> x(x_val);
  var_value<MatrixXd> y = rvalue(x, index_list(index_min_max(2, 3)));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(3, y.cols());
  EXPECT_FLOAT_EQ(1.0, y.val()(0, 0));
  EXPECT_FLOAT_EQ(1.1, y.val()(0, 1));
  EXPECT_FLOAT_EQ(1.2, y.val()(0, 2));
  EXPECT_FLOAT_EQ(2.0, y.val()(1, 0));
  EXPECT_FLOAT_EQ(2.1, y.val()(1, 1));
  EXPECT_FLOAT_EQ(2.2, y.val()(1, 2));
  sum(y).grad();
  auto check_i = [](int i) { return i > 0 && i < 3; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x);
  test_out_of_range(x, index_list(index_min_max(1, 15)));
  test_out_of_range(x, index_list(index_min_max(0, 2)));
}

TEST_F(RvalueRev, negative_min_max_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(4, 3);
  x_val << 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2;
  var_value<MatrixXd> x(x_val);
  var_value<MatrixXd> y = rvalue(x, index_list(index_min_max(3, 2)));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(3, y.cols());
  EXPECT_FLOAT_EQ(2, y.val()(0, 0));
  EXPECT_FLOAT_EQ(2.1, y.val()(0, 1));
  EXPECT_FLOAT_EQ(2.2, y.val()(0, 2));
  EXPECT_FLOAT_EQ(1, y.val()(1, 0));
  EXPECT_FLOAT_EQ(1.1, y.val()(1, 1));
  EXPECT_FLOAT_EQ(1.2, y.val()(1, 2));
  sum(y).grad();
  auto check_i = [](int i) { return i > 0 && i < 3; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x);
  test_out_of_range(x, index_list(index_min_max(3, 0)));
  test_out_of_range(x, index_list(index_min_max(15, 2)));
}

TEST_F(RvalueRev, omni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(4, 3);
  x_val << 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2;
  var_value<MatrixXd> x(x_val);
  var_value<MatrixXd> y = rvalue(x, index_list(index_omni()));
  EXPECT_EQ(4, y.rows());
  EXPECT_EQ(3, y.cols());
  EXPECT_FLOAT_EQ(0.0, y.val()(0, 0));
  EXPECT_FLOAT_EQ(0.1, y.val()(0, 1));
  EXPECT_FLOAT_EQ(0.2, y.val()(0, 2));
  EXPECT_FLOAT_EQ(3.0, y.val()(3, 0));
  EXPECT_FLOAT_EQ(3.1, y.val()(3, 1));
  EXPECT_FLOAT_EQ(3.2, y.val()(3, 2));
  sum(y).grad();
  auto check_i = [](int i) { return true; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x);
}

TEST_F(RvalueRev, omni_uni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  var_value<MatrixXd> x(x_val);
  var_value<VectorXd> y = rvalue(x, index_list(index_omni(), index_uni(2)));
  EXPECT_EQ(3, y.size());
  for (int j = 0; j < 3; ++j) {
    EXPECT_FLOAT_EQ(j + 0.1, y.val()(j));
  }
  sum(y).grad();
  auto check_i = [](int i) { return true; };
  auto check_j = [](int j) { return j == 1; };
  check_matrix_adjs(check_i, check_j, x);
  test_out_of_range(x, index_list(index_omni(), index_uni(0)));
  test_out_of_range(x, index_list(index_omni(), index_uni(20)));
}

TEST_F(RvalueRev, omni_omni_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::sum;
  using stan::math::var_value;
  MatrixXd x_val(3, 4);
  x_val << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  var_value<MatrixXd> x(x_val);
  var_value<MatrixXd> y = rvalue(x, index_list(index_omni(), index_omni()));
  EXPECT_EQ(x.rows(), y.rows());
  EXPECT_EQ(x.cols(), y.cols());
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.cols(); ++j) {
      EXPECT_FLOAT_EQ(x.val()(i, j), y.val()(i, j));
    }
  }
  sum(y).grad();
  auto check_i = [](int i) { return true; };
  auto check_j = [](int j) { return true; };
  check_matrix_adjs(check_i, check_j, x);
}

TEST_F(RvalueRev, multi_mat) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::var_value;
  MatrixXd x_val(4, 3);
  x_val << 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2;
  var_value<MatrixXd> x(x_val);
  std::vector<int> row_idx;
  row_idx.push_back(3);
  row_idx.push_back(4);
  row_idx.push_back(1);
  row_idx.push_back(4);
  row_idx.push_back(1);
  row_idx.push_back(4);
  row_idx.push_back(1);
  var_value<MatrixXd> y = rvalue(x, index_list(index_multi(row_idx)));
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
  for (int j = 0; j < x.cols(); ++j) {
    for (int i = 0; i < x.rows(); ++i) {
      EXPECT_FLOAT_EQ(x.adj()(i, j), touch_count.coeffRef(i, j))
          << "Failed for (i, j): (" << i << ", " << j << ")";
    }
  }
  row_idx.push_back(0);
  test_out_of_range(x, index_list(index_multi(row_idx)));

  row_idx[row_idx.size() - 1] = 15;
  test_out_of_range(x, index_list(index_multi(row_idx)));
}

TEST_F(RvalueRev, multi_multi_mat) {
  Eigen::MatrixXd x(4, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1,
      3.2, 3.3;
  stan::math::var_value<Eigen::MatrixXd> rx(x);

  std::vector<int> row_idx;
  std::vector<int> col_idx;
  row_idx.push_back(3);
  col_idx.push_back(4);
  row_idx.push_back(4);
  col_idx.push_back(4);
  row_idx.push_back(1);
  col_idx.push_back(3);
  row_idx.push_back(4);
  col_idx.push_back(2);
  row_idx.push_back(1);
  col_idx.push_back(1);
  row_idx.push_back(4);
  col_idx.push_back(4);
  row_idx.push_back(1);
  col_idx.push_back(1);

  stan::math::var_value<Eigen::MatrixXd> ry
      = rvalue(rx, index_list(index_multi(row_idx), index_multi(col_idx)));
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
  for (int j = 0; j < rx.cols(); ++j) {
    for (int i = 0; i < rx.rows(); ++i) {
      EXPECT_FLOAT_EQ(rx.adj()(i, j), touch_count.coeffRef(i, j));
    }
  }
  row_idx.push_back(19);
  row_idx.push_back(22);
  test_out_of_range(rx, index_list(index_multi(row_idx), index_multi(col_idx)));
  col_idx.push_back(19);
  col_idx.push_back(22);
  test_out_of_range(rx, index_list(index_multi(row_idx), index_multi(col_idx)));
}
