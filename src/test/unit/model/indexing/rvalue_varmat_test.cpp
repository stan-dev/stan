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

TEST_F(RvalueRev, rvalue_vector_nil) {
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

TEST_F(RvalueRev, rvalue_vector_uni_nil) {
  using stan::math::var_value;
  Eigen::VectorXd x(3);
  x(0) = 1.1;
  x(1) = 2.2;
  x(2) = 3.3;
  var_value<Eigen::VectorXd> xv(x);

  for (size_t k = 0; k < x.size(); ++k)
    EXPECT_EQ(x[k], rvalue(xv, index_list(index_uni(k + 1))).val());

  test_out_of_range(x, index_list(index_uni(-1)));
  test_out_of_range(x, index_list(index_uni(0)));
  test_out_of_range(x, index_list(index_uni(4)));
  stan::math::recover_memory();
}

TEST_F(RvalueRev, rvalue_vector_varmat_min_max_nil) {
  using stan::math::var_value;
  Eigen::VectorXd x(4);
  x(0) = 1.1;
  x(1) = 2.2;
  x(2) = 3.3;
  x(3) = 4.4;
  var_value<Eigen::VectorXd> xv(x);

  // min > max
  for (int mn = 0; mn < 4; ++mn) {
    for (int mx = mn; mx < 4; ++mx) {
      var_value<Eigen::VectorXd> rx
          = rvalue(xv, index_list(index_min_max(mn + 1, mx + 1)));
      EXPECT_FLOAT_EQ(mx - mn + 1, rx.size());
      for (int n = mn; n <= mx; ++n)
        EXPECT_FLOAT_EQ(x[n], rx.val()[n - mn]);
    }
  }

  // max > min
  for (int mn = 3; mn > -1; --mn) {
    for (int mx = mn; mx > -1; --mx) {
      var_value<Eigen::VectorXd> rx
          = rvalue(xv, index_list(index_min_max(mn + 1, mx + 1)));
      EXPECT_FLOAT_EQ(mn - mx + 1, rx.size());
      for (int n = mn; n <= mx; ++n)
        EXPECT_FLOAT_EQ(x[n], rx.val()[n - mn]);
    }
  }

  test_out_of_range(xv, index_list(index_min_max(0, 2)));
  test_out_of_range(xv, index_list(index_min_max(2, 5)));
  stan::math::recover_memory();
}

TEST_F(RvalueRev, rvalue_varmat_uni_multi) {
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
  var_value<Eigen::Matrix<double, -1, 1>> y
      = rvalue(x, index_list(index_uni(1), index_min(2)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(0.1, y.val()[0]);
  EXPECT_FLOAT_EQ(0.2, y.val()[1]);
  test_out_of_range(x, index_list(index_uni(0), index_min(2)));
  test_out_of_range(x, index_list(index_uni(1), index_min(0)));
  // TODO: Test reverse pass for adjoint propogration
  y = rvalue(x, index_list(index_uni(2), index_max(2)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.0, y.val()[0]);
  EXPECT_FLOAT_EQ(1.1, y.val()[1]);
  test_out_of_range(x, index_list(index_uni(0), index_max(2)));
  test_out_of_range(x, index_list(index_uni(1), index_max(15)));

  y = rvalue(x, index_list(index_uni(2), index_min_max(2, 3)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.1, y.val()[0]);
  EXPECT_FLOAT_EQ(1.2, y.val()[1]);
  test_out_of_range(x, index_list(index_uni(0), index_min_max(2, 3)));
  test_out_of_range(x, index_list(index_uni(10), index_min_max(2, 3)));
  test_out_of_range(x, index_list(index_uni(1), index_min_max(0, 3)));
  test_out_of_range(x, index_list(index_uni(1), index_min_max(2, 15)));

  y = rvalue(x, index_list(index_uni(2), index_min_max(2, 2)));
  EXPECT_EQ(1, y.size());
  EXPECT_FLOAT_EQ(1.1, y.val()[0]);

  y = rvalue(x, index_list(index_uni(3), index_omni()));
  EXPECT_EQ(3, y.size());
  EXPECT_FLOAT_EQ(2.0, y.val()[0]);
  EXPECT_FLOAT_EQ(2.1, y.val()[1]);
  EXPECT_FLOAT_EQ(2.2, y.val()[2]);
  test_out_of_range(x, index_list(index_uni(0), index_omni()));
}

template <typename T>
void varvector_uni_test() {
  T v(3);
  v << 0, 1, 2;
  stan::math::var_value<T> rv(v);
  EXPECT_FLOAT_EQ(0, rvalue(rv, index_list(index_uni(1))).val());
  EXPECT_FLOAT_EQ(1, rvalue(rv, index_list(index_uni(2))).val());
  EXPECT_FLOAT_EQ(2, rvalue(rv, index_list(index_uni(3))).val());

  test_out_of_range(rv, index_list(index_uni(0)));
  test_out_of_range(rv, index_list(index_uni(20)));
}

TEST_F(RvalueRev, rvalueVectorUni) { varvector_uni_test<Eigen::VectorXd>(); }

TEST_F(RvalueRev, rvalueRowVectorUni) {
  varvector_uni_test<Eigen::RowVectorXd>();
}

template <typename T>
void varvector_multi_test() {
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
  stan::math::set_zero_all_adjoints();

  vi = rvalue(rv, index_list(index_min(3)));
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
  stan::math::set_zero_all_adjoints();

  vi = rvalue(rv, index_list(index_max(3)));
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
  stan::math::set_zero_all_adjoints();

  vi = rvalue(rv, index_list(index_min_max(2, 4)));
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
  stan::math::set_zero_all_adjoints();

  std::vector<int> ns;
  ns.push_back(4);
  ns.push_back(2);
  ns.push_back(2);
  ns.push_back(1);
  ns.push_back(5);
  ns.push_back(2);
  ns.push_back(4);
  std::cout << "\n before vi_val: \n" << vi.val() << "\n";
  std::cout << "\n before rv_val: \n" << rv.val() << "\n";
  vi = rvalue(rv, index_list(index_multi(ns)));
  std::cout << "\n after vi_val: \n" << vi.val() << "\n";
  std::cout << "\n after rv_val: \n" << rv.val() << "\n";
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
  std::cout << "\n before vi_adj: \n" << vi.adj() << "\n";
  std::cout << "\n before rv_adj: \n" << rv.adj() << "\n";
  stan::math::sum(vi).grad();
  std::cout << "\n after vi_adj: \n" << vi.adj() << "\n";
  std::cout << "\n after rv_adj: \n" << rv.adj() << "\n";
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

TEST_F(RvalueRev, rvalueVectorMulti) {
  varvector_multi_test<Eigen::VectorXd>();
}

TEST_F(RvalueRev, rvalueRowVectorMulti) {
  varvector_multi_test<Eigen::RowVectorXd>();
}

TEST(ModelIndexing, rvalueMatrixUni) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;

  MatrixXd m(4, 3);
  m << 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2;
  RowVectorXd v = rvalue(m, index_list(index_uni(1)));
  EXPECT_EQ(3, v.size());
  EXPECT_FLOAT_EQ(0.0, v(0));
  EXPECT_FLOAT_EQ(0.1, v(1));
  EXPECT_FLOAT_EQ(0.2, v(2));
  test_out_of_range(m, index_list(index_uni(0)));
  test_out_of_range(m, index_list(index_uni(15)));

  v = rvalue(m, index_list(index_uni(2)));
  EXPECT_EQ(3, v.size());
  EXPECT_FLOAT_EQ(1.0, v(0));
  EXPECT_FLOAT_EQ(1.1, v(1));
  EXPECT_FLOAT_EQ(1.2, v(2));
}

TEST_F(RvalueRev, rvalueMatrixUni) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::var_value;

  MatrixXd m(4, 3);
  m << 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2;
  var_value<MatrixXd> rm(m);
  var_value<RowVectorXd> v = rvalue(rm, index_list(index_uni(1)));
  EXPECT_EQ(3, v.size());
  EXPECT_FLOAT_EQ(0.0, v.val()(0));
  EXPECT_FLOAT_EQ(0.1, v.val()(1));
  EXPECT_FLOAT_EQ(0.2, v.val()(2));
  test_out_of_range(rm, index_list(index_uni(0)));
  test_out_of_range(rm, index_list(index_uni(15)));

  v = rvalue(rm, index_list(index_uni(2)));
  EXPECT_EQ(3, v.size());
  EXPECT_FLOAT_EQ(1.0, v.val()(0));
  EXPECT_FLOAT_EQ(1.1, v.val()(1));
  EXPECT_FLOAT_EQ(1.2, v.val()(2));
}

TEST_F(RvalueRev, rvalueMatrixMulti) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  using stan::math::var_value;
  MatrixXd m(4, 3);
  m << 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2;
  var_value<MatrixXd> rm(m);
  var_value<MatrixXd> a = rvalue(rm, index_list(index_min(3)));
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(2.0, a.val()(0, 0));
  EXPECT_FLOAT_EQ(2.1, a.val()(0, 1));
  EXPECT_FLOAT_EQ(2.2, a.val()(0, 2));
  EXPECT_FLOAT_EQ(3.0, a.val()(1, 0));
  EXPECT_FLOAT_EQ(3.1, a.val()(1, 1));
  EXPECT_FLOAT_EQ(3.2, a.val()(1, 2));
  test_out_of_range(rm, index_list(index_min(0)));

  a = rvalue(rm, index_list(index_max(2)));
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(0.0, a.val()(0, 0));
  EXPECT_FLOAT_EQ(0.1, a.val()(0, 1));
  EXPECT_FLOAT_EQ(0.2, a.val()(0, 2));
  EXPECT_FLOAT_EQ(1.0, a.val()(1, 0));
  EXPECT_FLOAT_EQ(1.1, a.val()(1, 1));
  EXPECT_FLOAT_EQ(1.2, a.val()(1, 2));
  test_out_of_range(rm, index_list(index_max(15)));

  a = rvalue(rm, index_list(index_min_max(2, 3)));
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(1.0, a.val()(0, 0));
  EXPECT_FLOAT_EQ(1.1, a.val()(0, 1));
  EXPECT_FLOAT_EQ(1.2, a.val()(0, 2));
  EXPECT_FLOAT_EQ(2.0, a.val()(1, 0));
  EXPECT_FLOAT_EQ(2.1, a.val()(1, 1));
  EXPECT_FLOAT_EQ(2.2, a.val()(1, 2));
  std::cout << "\n before rm_val: \n" << rm.val() << "\n";
  std::cout << "\n before a_val: \n" << a.val() << "\n";

  a = rvalue(rm, index_list(index_min_max(3, 2)));
  std::cout << "\n after rm_val: \n" << rm.val() << "\n";
  std::cout << "\n after a_val: \n" << a.val() << "\n";
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(2, a.val()(0, 0));
  EXPECT_FLOAT_EQ(2.1, a.val()(0, 1));
  EXPECT_FLOAT_EQ(2.2, a.val()(0, 2));
  EXPECT_FLOAT_EQ(1, a.val()(1, 0));
  EXPECT_FLOAT_EQ(1.1, a.val()(1, 1));
  EXPECT_FLOAT_EQ(1.2, a.val()(1, 2));

  test_out_of_range(rm, index_list(index_min_max(0, 3)));
  test_out_of_range(rm, index_list(index_min_max(2, 15)));

  a = rvalue(rm, index_list(index_omni()));
  EXPECT_EQ(4, a.rows());
  EXPECT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(0.0, a.val()(0, 0));
  EXPECT_FLOAT_EQ(0.1, a.val()(0, 1));
  EXPECT_FLOAT_EQ(0.2, a.val()(0, 2));
  EXPECT_FLOAT_EQ(3.0, a.val()(3, 0));
  EXPECT_FLOAT_EQ(3.1, a.val()(3, 1));
  EXPECT_FLOAT_EQ(3.2, a.val()(3, 2));

  std::vector<int> ns;
  ns.push_back(3);
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(4);
  ns.push_back(1);
  a = rvalue(rm, index_list(index_multi(ns)));
  EXPECT_FLOAT_EQ(7, a.rows());
  EXPECT_FLOAT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(2.0, a.val()(0, 0));
  EXPECT_FLOAT_EQ(2.1, a.val()(0, 1));
  EXPECT_FLOAT_EQ(2.2, a.val()(0, 2));
  EXPECT_FLOAT_EQ(3.0, a.val()(5, 0));
  EXPECT_FLOAT_EQ(3.1, a.val()(5, 1));
  EXPECT_FLOAT_EQ(3.2, a.val()(5, 2));
  EXPECT_FLOAT_EQ(0.0, a.val()(6, 0));
  EXPECT_FLOAT_EQ(0.1, a.val()(6, 1));
  EXPECT_FLOAT_EQ(0.2, a.val()(6, 2));

  ns.push_back(0);
  test_out_of_range(rm, index_list(index_multi(ns)));

  ns[ns.size() - 1] = 15;
  test_out_of_range(rm, index_list(index_multi(ns)));
}

TEST_F(RvalueRev, rvalueMatrixSingleSingle) {
  Eigen::MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  stan::math::var_value<Eigen::MatrixXd> rx(x);
  for (int m = 0; m < 3; ++m)
    for (int n = 0; n < 4; ++n)
      EXPECT_FLOAT_EQ(
          m + n / 10.0,
          rvalue(rx, index_list(index_uni(m + 1), index_uni(n + 1))).val());
  test_out_of_range(rx, index_list(index_uni(0), index_uni(1)));
  test_out_of_range(rx, index_list(index_uni(0), index_uni(10)));
  test_out_of_range(rx, index_list(index_uni(1), index_uni(0)));
  test_out_of_range(rx, index_list(index_uni(1), index_uni(10)));
}

TEST_F(RvalueRev, rvalueMatrixSingleMulti) {
  using stan::math::var_value;
  Eigen::MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  var_value<Eigen::MatrixXd> vx(x);
  var_value<Eigen::RowVectorXd> vr
      = rvalue(vx, index_list(index_uni(2), index_omni()));
  EXPECT_EQ(4, vr.size());
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(vr.val()(i), vx.val()(1, i));
  test_out_of_range(vx, index_list(index_uni(0), index_omni()));
  test_out_of_range(vx, index_list(index_uni(10), index_omni()));

  vr = rvalue(vx, index_list(index_uni(3), index_min(2)));
  EXPECT_EQ(3, vr.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(vr.val()(i), vx.val()(2, i + 1));
  test_out_of_range(vx, index_list(index_uni(0), index_min(2)));
  test_out_of_range(vx, index_list(index_uni(1), index_min(0)));
}

TEST_F(RvalueRev, rvalueMatrixMultiSingle) {
  Eigen::MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  stan::math::var_value<Eigen::MatrixXd> rx(x);
  stan::math::var_value<Eigen::VectorXd> rv
      = rvalue(x, index_list(index_omni(), index_uni(2)));
  EXPECT_EQ(3, rv.size());
  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(j + 0.1, rv.val()(j));
  test_out_of_range(rx, index_list(index_omni(), index_uni(0)));
  test_out_of_range(rx, index_list(index_omni(), index_uni(20)));

  rv = rvalue(rx, index_list(index_min(2), index_uni(3)));
  EXPECT_EQ(2, rv.size());
  for (int j = 0; j < 2; ++j)
    EXPECT_EQ(1 + j + 0.2, rv.val()(j));
  test_out_of_range(rx, index_list(index_min(0), index_uni(3)));
  test_out_of_range(rx, index_list(index_min(2), index_uni(0)));
  test_out_of_range(rx, index_list(index_min(2), index_uni(30)));
}

TEST_F(RvalueRev, rvalueMatrixOmniOmni) {
  Eigen::MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  stan::math::var_value<Eigen::MatrixXd> rx(x);
  stan::math::var_value<Eigen::MatrixXd> ry
      = rvalue(rx, index_list(index_omni(), index_omni()));
  EXPECT_EQ(rx.rows(), ry.rows());
  EXPECT_EQ(rx.cols(), ry.cols());
  for (int i = 0; i < rx.rows(); ++i)
    for (int j = 0; j < rx.cols(); ++j)
      EXPECT_FLOAT_EQ(rx.val()(i, j), ry.val()(i, j));
}

TEST_F(RvalueRev, rvalueMatrixMinMaxMaxMin) {
  Eigen::MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  stan::math::var_value<Eigen::MatrixXd> rx(x);
  stan::math::var_value<Eigen::MatrixXd> ry = rvalue(rx, index_list(index_min(2), index_min(3)));
  EXPECT_EQ(2, ry.rows());
  EXPECT_EQ(2, ry.cols());
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      EXPECT_FLOAT_EQ(i + 1 + (j + 2) / 10.0, ry.val()(i, j));
  test_out_of_range(rx, index_list(index_min(0), index_min(3)));
  test_out_of_range(rx, index_list(index_min(2), index_min(0)));
}

TEST_F(RvalueRev, rvalueMatrixMultiMulti) {
  Eigen::MatrixXd x(4, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3;
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

  stan::math::var_value<Eigen::MatrixXd> ry = rvalue(rx, index_list(index_multi(row_idx), index_multi(col_idx)));
  std::cout << "\n after rx.val(): \n" << rx.val() << "\n";
  std::cout << "\n after ry.val(): \n" << ry.val() << "\n";
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
  std::cout << "\n after rx.adj(): \n" << rx.adj() << "\n";
  std::cout << "\n after ry.adj(): \n" << ry.adj() << "\n";
  std::cout << "\n after touches: \n" << touch_count << "\n";
  for (int j = 0; j < rx.cols(); ++j) {
    for (int i = 0; i < rx.rows(); ++i) {
      EXPECT_FLOAT_EQ(rx.adj()(i, j), touch_count.coeffRef(i, j));
    }
  }
  puts("Pass1");
  row_idx.push_back(19);
  row_idx.push_back(22);
  test_out_of_range(rx, index_list(index_multi(row_idx), index_multi(col_idx)));
  row_idx.pop_back();
  row_idx.pop_back();
  col_idx.push_back(19);
  col_idx.push_back(22);
  test_out_of_range(rx, index_list(index_multi(row_idx), index_multi(col_idx)));
}
