#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/rvalue.hpp>
#include <gtest/gtest.h>

using stan::model::nil_index_list;
using stan::model::cons_index_list;
using stan::model::index_uni;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_min;
using stan::model::index_max;
using stan::model::index_min_max;

template <typename C, typename I>
void test_out_of_range(const C& c, const I& idxs) {
  EXPECT_THROW(stan::model::rvalue(c, idxs), std::out_of_range);
}

TEST(ModelIndexing, rvalue_vector_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);

  std::vector<double> rx = rvalue(x, nil_index_list());
  EXPECT_EQ(2,rx.size());
  EXPECT_FLOAT_EQ(1.1, rx[0]);
  EXPECT_FLOAT_EQ(2.2, rx[1]);
}

TEST(ModelIndexing, rvalue_vector_uni_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  for (size_t k = 0; k < x.size(); ++k)
    EXPECT_EQ(x[k], rvalue(x, index_list(index_uni(k + 1))));

  test_out_of_range(x, index_list(index_uni(-1)));
  test_out_of_range(x, index_list(index_uni(0)));
  test_out_of_range(x, index_list(index_uni(4)));
}

TEST(ModelIndexing, rvalue_vector_multi_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);
  x.push_back(4.4);
  x.push_back(5.5);
  x.push_back(6.6);

  std::vector<int> idxs;
  idxs.push_back(1);
  idxs.push_back(4);
  idxs.push_back(5);
  
  std::vector<double> rx = rvalue(x, index_list(index_multi(idxs)));
  EXPECT_FLOAT_EQ(3, rx.size());
  EXPECT_FLOAT_EQ(x[0], rx[0]);
  EXPECT_FLOAT_EQ(x[3], rx[1]);
  EXPECT_FLOAT_EQ(x[4], rx[2]);

  idxs.push_back(0);
  test_out_of_range(x, index_list(index_multi(idxs)));
  idxs[idxs.size()-1] = 7;
  test_out_of_range(x, index_list(index_multi(idxs)));
}

TEST(ModelIndexing, rvalue_vector_omni_nil) {
  std::vector<double> y;
  std::vector<double> ry = rvalue(y, index_list(index_omni()));
  EXPECT_EQ(0U, y.size());

  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  std::vector<double> rx = rvalue(x, index_list(index_omni()));
  EXPECT_FLOAT_EQ(3, rx.size());
  for (size_t n = 0; n < rx.size(); ++n)
    EXPECT_FLOAT_EQ(x[n], rx[n]);
}

TEST(ModelIndexing, rvalue_vector_min_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  for (int k = 0; k < 4; ++k) {
    std::vector<double> rx = rvalue(x, index_list(index_min(k + 1)));
    EXPECT_FLOAT_EQ(3 - k, rx.size());
    for (size_t n = 0; n < rx.size(); ++n)
      EXPECT_FLOAT_EQ(x[n + k], rx[n]);
  }

  std::vector<double> ry = rvalue(x, index_list(index_min(7)));
  EXPECT_EQ(0U, ry.size());

  test_out_of_range(x,  index_list(index_min(0)));
}

TEST(ModelIndexing, rvalue_vector_max_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  for (int k = 0; k < 3; ++k) {
    std::vector<double> rx = rvalue(x, index_list(index_max(k + 1)));
    EXPECT_FLOAT_EQ(k + 1, rx.size());
    for (size_t n = 0; n < rx.size(); ++n)
      EXPECT_FLOAT_EQ(x[n], rx[n]);
  }

  std::vector<double> ry = rvalue(x, index_list(index_max(0)));
  EXPECT_EQ(0U, ry.size());

  test_out_of_range(x, index_list(index_max(4)));
}

TEST(ModelIndexing, rvalue_vector_min_max_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);
  x.push_back(4.4);

  for (int mn = 0; mn < 4; ++mn) {
    for (int mx = mn; mx < 4; ++mx) {
      std::vector<double> rx 
        = rvalue(x, index_list(index_min_max(mn + 1, mx + 1)));
      EXPECT_FLOAT_EQ(mx - mn + 1, rx.size());
      for (int n = mn; n <= mx; ++n)
        EXPECT_FLOAT_EQ(x[n], rx[n - mn]);
    }
  }

  test_out_of_range(x, index_list(index_min_max(0, 2)));
  test_out_of_range(x, index_list(index_min_max(2, 5)));
}

TEST(ModelIndexing, rvalue_doubless_uni_uni) {
  using std::vector;
  
  vector<double> x0;
  x0.push_back(0.0);
  x0.push_back(0.1);

  vector<double> x1;
  x1.push_back(1.0);
  x1.push_back(1.1);

  vector<vector<double> > x;
  x.push_back(x0);
  x.push_back(x1);
  
  for (int m = 0; m < 2; ++m)
    for (int n = 0; n < 2; ++n)
      EXPECT_FLOAT_EQ(m + n / 10.0, 
                  rvalue(x, index_list(index_uni(m + 1), index_uni(n + 1))));

  test_out_of_range(x, index_list(index_uni(0), index_uni(1)));
  test_out_of_range(x, index_list(index_uni(5), index_uni(1)));
  test_out_of_range(x, index_list(index_uni(1), index_uni(0)));
  test_out_of_range(x, index_list(index_uni(1), index_uni(5)));
}

TEST(ModelIndexing, rvalue_doubless_uni_multi) {
  using std::vector;
  
  vector<double> x0;
  x0.push_back(0.0);
  x0.push_back(0.1);
  x0.push_back(0.2);

  vector<double> x1;
  x1.push_back(1.0);
  x1.push_back(1.1);
  x1.push_back(1.2);

  vector<double> x2;
  x2.push_back(2.0);
  x2.push_back(2.1);
  x2.push_back(2.2);

  vector<vector<double> > x;
  x.push_back(x0);
  x.push_back(x1);
  x.push_back(x2);

  vector<double> y = rvalue(x, index_list(index_uni(1), index_min(2)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(0.1, y[0]);
  EXPECT_FLOAT_EQ(0.2, y[1]);
  test_out_of_range(x, index_list(index_uni(0), index_min(2)));
  test_out_of_range(x, index_list(index_uni(1), index_min(0)));

  y = rvalue(x, index_list(index_uni(2), index_max(2)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.0, y[0]);
  EXPECT_FLOAT_EQ(1.1, y[1]);
  test_out_of_range(x, index_list(index_uni(0), index_max(2)));
  test_out_of_range(x, index_list(index_uni(1), index_max(15)));

  y = rvalue(x, index_list(index_uni(2), index_min_max(2,3)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.1, y[0]);
  EXPECT_FLOAT_EQ(1.2, y[1]);
  test_out_of_range(x, index_list(index_uni(0), index_min_max(2,3)));
  test_out_of_range(x, index_list(index_uni(10), index_min_max(2,3)));
  test_out_of_range(x, index_list(index_uni(1), index_min_max(0,3)));
  test_out_of_range(x, index_list(index_uni(1), index_min_max(2,15)));

  y = rvalue(x, index_list(index_uni(2), index_min_max(2,2)));
  EXPECT_EQ(1, y.size());
  EXPECT_FLOAT_EQ(1.1, y[0]);

  y = rvalue(x, index_list(index_uni(3), index_omni()));
  EXPECT_EQ(3, y.size());
  EXPECT_FLOAT_EQ(2.0, y[0]);
  EXPECT_FLOAT_EQ(2.1, y[1]);
  EXPECT_FLOAT_EQ(2.2, y[2]);
  test_out_of_range(x, index_list(index_uni(0), index_omni()));

  vector<int> ns;
  ns.push_back(3);
  ns.push_back(1);
  y = rvalue(x, index_list(index_uni(1), index_multi(ns)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(0.2, y[0]);
  EXPECT_FLOAT_EQ(0.0, y[1]);
  test_out_of_range(x, index_list(index_uni(0), index_multi(ns)));
  test_out_of_range(x, index_list(index_uni(10), index_multi(ns)));

  ns.push_back(0);
  test_out_of_range(x, index_list(index_uni(1), index_multi(ns)));

  ns[ns.size() - 1] = 20;
  test_out_of_range(x, index_list(index_uni(1), index_multi(ns)));
}

TEST(ModelIndexing, rvalue_doubless_multi_uni) {
  using std::vector;
  
  vector<double> x0;
  x0.push_back(0.0);
  x0.push_back(0.1);
  x0.push_back(0.2);

  vector<double> x1;
  x1.push_back(1.0);
  x1.push_back(1.1);
  x1.push_back(1.2);

  vector<double> x2;
  x2.push_back(2.0);
  x2.push_back(2.1);
  x2.push_back(2.2);

  vector<vector<double> > x;
  x.push_back(x0);
  x.push_back(x1);
  x.push_back(x2);

  vector<double> y = rvalue(x, index_list(index_min(2), index_uni(1)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.0, y[0]);
  EXPECT_FLOAT_EQ(2.0, y[1]);
  test_out_of_range(x, index_list(index_min(0), index_uni(1)));
  test_out_of_range(x, index_list(index_min(2), index_uni(0)));
  test_out_of_range(x, index_list(index_min(2), index_uni(10)));

  y = rvalue(x, index_list(index_max(2), index_uni(3)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(0.2, y[0]);
  EXPECT_FLOAT_EQ(1.2, y[1]);
  test_out_of_range(x, index_list(index_max(10), index_uni(3)));
  test_out_of_range(x, index_list(index_max(2), index_uni(0)));
  test_out_of_range(x, index_list(index_max(2), index_uni(15)));

  y = rvalue(x, index_list(index_min_max(2,3), index_uni(2)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.1, y[0]);
  EXPECT_FLOAT_EQ(2.1, y[1]);
  test_out_of_range(x, index_list(index_min_max(0,3), index_uni(2)));
  test_out_of_range(x, index_list(index_min_max(2,15), index_uni(2)));
  test_out_of_range(x, index_list(index_min_max(2,3), index_uni(0)));
  test_out_of_range(x, index_list(index_min_max(2,3), index_uni(10)));

  y = rvalue(x, index_list(index_min_max(2,2), index_uni(2)));
  EXPECT_EQ(1, y.size());
  EXPECT_FLOAT_EQ(1.1, y[0]);
  test_out_of_range(x, index_list(index_min_max(0,2), index_uni(2)));
  test_out_of_range(x, index_list(index_min_max(2,12), index_uni(2)));
  test_out_of_range(x, index_list(index_min_max(2,2), index_uni(0)));
  test_out_of_range(x, index_list(index_min_max(2,2), index_uni(15)));

  y = rvalue(x, index_list(index_omni(), index_uni(3)));
  EXPECT_EQ(3, y.size());
  EXPECT_FLOAT_EQ(0.2, y[0]);
  EXPECT_FLOAT_EQ(1.2, y[1]);
  EXPECT_FLOAT_EQ(2.2, y[2]);
  test_out_of_range(x, index_list(index_omni(), index_uni(0)));
  test_out_of_range(x, index_list(index_omni(), index_uni(10)));

  vector<int> ns;
  ns.push_back(3);
  ns.push_back(1);
  y = rvalue(x, index_list(index_multi(ns), index_uni(1)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(2.0, y[0]);
  EXPECT_FLOAT_EQ(0.0, y[1]);

  ns.push_back(0);
  test_out_of_range(x, index_list(index_multi(ns), index_uni(1)));

  ns[ns.size() - 1] = 15;
  test_out_of_range(x, index_list(index_multi(ns), index_uni(1)));
}

TEST(ModelIndexing, rvalue_doubless_multi_multi) {
  using std::vector;
  
  vector<double> x0;
  x0.push_back(0.0);
  x0.push_back(0.1);
  x0.push_back(0.2);

  vector<double> x1;
  x1.push_back(1.0);
  x1.push_back(1.1);
  x1.push_back(1.2);

  vector<double> x2;
  x2.push_back(2.0);
  x2.push_back(2.1);
  x2.push_back(2.2);

  vector<vector<double> > x;
  x.push_back(x0);
  x.push_back(x1);
  x.push_back(x2);

  vector<vector<double> > y = rvalue(x, index_list(index_max(2), index_min(2)));
  EXPECT_EQ(2, y.size());
  EXPECT_EQ(2, y[0].size());
  EXPECT_EQ(2, y[1].size());
  EXPECT_FLOAT_EQ(0.1, y[0][0]);
  EXPECT_FLOAT_EQ(0.2, y[0][1]);
  EXPECT_FLOAT_EQ(1.1, y[1][0]);
  EXPECT_FLOAT_EQ(1.2, y[1][1]);
  test_out_of_range(x, index_list(index_max(20), index_min(2)));
  test_out_of_range(x, index_list(index_max(2), index_min(0)));
}

template <typename T>
void vector_uni_test() {
  T v(3);
  v << 0, 1, 2;
  
  EXPECT_FLOAT_EQ(0, rvalue(v, index_list(index_uni(1))));
  EXPECT_FLOAT_EQ(1, rvalue(v, index_list(index_uni(2))));
  EXPECT_FLOAT_EQ(2, rvalue(v, index_list(index_uni(3))));
  
  test_out_of_range(v, index_list(index_uni(0)));
  test_out_of_range(v, index_list(index_uni(20)));
}

TEST(ModelIndexing, rvalueVectorUni) {
  vector_uni_test<Eigen::VectorXd>();
}

TEST(ModelIndexing, rvalueRowVectorUni) {
  vector_uni_test<Eigen::RowVectorXd>();
}

template <typename T>
void vector_multi_test() {
  T v(5);
  v << 0, 1, 2, 3, 4;
  
  T vi = rvalue(v, index_list(index_omni()));
  EXPECT_EQ(5, vi.size());
  EXPECT_FLOAT_EQ(0, vi(0));
  EXPECT_FLOAT_EQ(2, vi(2));
  EXPECT_FLOAT_EQ(4, vi(4));
  
  vi = rvalue(v, index_list(index_min(3)));
  EXPECT_EQ(3, vi.size());
  EXPECT_FLOAT_EQ(2, vi(0));
  EXPECT_FLOAT_EQ(4, vi(2));
  test_out_of_range(v, index_list(index_min(0)));

  vi = rvalue(v, index_list(index_max(3)));
  EXPECT_EQ(3, vi.size());
  EXPECT_FLOAT_EQ(0, vi(0));
  EXPECT_FLOAT_EQ(2, vi(2));
  test_out_of_range(v, index_list(index_max(15)));

  vi = rvalue(v, index_list(index_min_max(2,4)));
  EXPECT_EQ(3, vi.size());
  EXPECT_FLOAT_EQ(1, vi(0));
  EXPECT_FLOAT_EQ(3, vi(2));
  test_out_of_range(v, index_list(index_min_max(0,4)));
  test_out_of_range(v, index_list(index_min_max(2,15)));

  std::vector<int> ns;
  ns.push_back(4);
  ns.push_back(2);
  ns.push_back(2);
  ns.push_back(1);
  ns.push_back(5);
  ns.push_back(2);
  ns.push_back(4);
  
  vi = rvalue(v, index_list(index_multi(ns)));
  EXPECT_EQ(7, vi.size());
  EXPECT_FLOAT_EQ(3.0, vi(0));
  EXPECT_FLOAT_EQ(1.0, vi(2));
  EXPECT_FLOAT_EQ(4.0, vi(4));
  EXPECT_FLOAT_EQ(3.0, vi(6));

  ns.push_back(0);
  test_out_of_range(v, index_list(index_multi(ns)));
  
  ns[ns.size() - 1] = 15;
  test_out_of_range(v, index_list(index_multi(ns)));
}

TEST(ModelIndexing, rvalueVectorMulti) {
  vector_multi_test<Eigen::VectorXd>();
}

TEST(ModelIndexing, rvalueRowVectorMulti) {
  vector_multi_test<Eigen::RowVectorXd>();
}

TEST(ModelIndexing, rvalueMatrixUni) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  
  MatrixXd m(4,3);
  m << 
    0.0, 0.1, 0.2, 
    1.0, 1.1, 1.2,
    2.0, 2.1, 2.2,
    3.0, 3.1, 3.2;

  // FIXME
  
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

TEST(ModelIndexing, rvalueMatrixMulti) {
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;
  
  MatrixXd m(4,3);
  m << 
    0.0, 0.1, 0.2, 
    1.0, 1.1, 1.2,
    2.0, 2.1, 2.2,
    3.0, 3.1, 3.2;
  
  MatrixXd a = rvalue(m, index_list(index_min(3)));
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(2.0, a(0,0));
  EXPECT_FLOAT_EQ(2.1, a(0,1));
  EXPECT_FLOAT_EQ(2.2, a(0,2));
  EXPECT_FLOAT_EQ(3.0, a(1,0));
  EXPECT_FLOAT_EQ(3.1, a(1,1));
  EXPECT_FLOAT_EQ(3.2, a(1,2));
  test_out_of_range(m, index_list(index_min(0)));

  a = rvalue(m, index_list(index_max(2)));
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(0.0, a(0,0));
  EXPECT_FLOAT_EQ(0.1, a(0,1));
  EXPECT_FLOAT_EQ(0.2, a(0,2));
  EXPECT_FLOAT_EQ(1.0, a(1,0));
  EXPECT_FLOAT_EQ(1.1, a(1,1));
  EXPECT_FLOAT_EQ(1.2, a(1,2));
  test_out_of_range(m, index_list(index_max(15)));

  a = rvalue(m, index_list(index_min_max(2,3)));
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(1.0, a(0,0));
  EXPECT_FLOAT_EQ(1.1, a(0,1));
  EXPECT_FLOAT_EQ(1.2, a(0,2));
  EXPECT_FLOAT_EQ(2.0, a(1,0));
  EXPECT_FLOAT_EQ(2.1, a(1,1));
  EXPECT_FLOAT_EQ(2.2, a(1,2));
  test_out_of_range(m, index_list(index_min_max(0,3)));
  test_out_of_range(m, index_list(index_min_max(2,15)));

  a = rvalue(m, index_list(index_omni()));
  EXPECT_EQ(4, a.rows());
  EXPECT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(0.0, a(0,0));
  EXPECT_FLOAT_EQ(0.1, a(0,1));
  EXPECT_FLOAT_EQ(0.2, a(0,2));
  EXPECT_FLOAT_EQ(3.0, a(3,0));
  EXPECT_FLOAT_EQ(3.1, a(3,1));
  EXPECT_FLOAT_EQ(3.2, a(3,2));

  std::vector<int> ns;
  ns.push_back(3);
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(4);
  ns.push_back(1);
  a = rvalue(m, index_list(index_multi(ns)));
  EXPECT_FLOAT_EQ(7, a.rows());
  EXPECT_FLOAT_EQ(3, a.cols());
  EXPECT_FLOAT_EQ(2.0, a(0,0));
  EXPECT_FLOAT_EQ(2.1, a(0,1));
  EXPECT_FLOAT_EQ(2.2, a(0,2));
  EXPECT_FLOAT_EQ(3.0, a(5,0));
  EXPECT_FLOAT_EQ(3.1, a(5,1));
  EXPECT_FLOAT_EQ(3.2, a(5,2));
  EXPECT_FLOAT_EQ(0.0, a(6,0));
  EXPECT_FLOAT_EQ(0.1, a(6,1));
  EXPECT_FLOAT_EQ(0.2, a(6,2));

  ns.push_back(0);
  test_out_of_range(m, index_list(index_multi(ns)));
  
  ns[ns.size() - 1] = 15;
  test_out_of_range(m, index_list(index_multi(ns)));
}

TEST(ModelIndexing, rvalueMatrixSingleSingle) {
  Eigen::MatrixXd x(3,4);
  x << 
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;

  for (int m = 0; m < 3; ++m)
    for (int n = 0; n < 4; ++n)
      EXPECT_FLOAT_EQ(m + n / 10.0, 
                  rvalue(x, index_list(index_uni(m + 1), index_uni(n + 1))));
  test_out_of_range(x, index_list(index_uni(0), index_uni(1)));
  test_out_of_range(x, index_list(index_uni(0), index_uni(10)));
  test_out_of_range(x, index_list(index_uni(1), index_uni(0)));
  test_out_of_range(x, index_list(index_uni(1), index_uni(10)));
}

TEST(ModelIndexing, rvalueMatrixSingleMulti) {
  Eigen::MatrixXd x(3,4);
  x << 
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;

  Eigen::RowVectorXd v = rvalue(x, index_list(index_uni(2), index_omni()));
  EXPECT_EQ(4, v.size());
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(v(i), x(1,i));
  test_out_of_range(x, index_list(index_uni(0), index_omni()));
  test_out_of_range(x, index_list(index_uni(10), index_omni()));

  v = rvalue(x, index_list(index_uni(3), index_min(2)));
  EXPECT_EQ(3, v.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(v(i), x(2,i + 1));
  test_out_of_range(x, index_list(index_uni(0), index_min(2)));
  test_out_of_range(x, index_list(index_uni(1), index_min(0)));
}

TEST(ModelIndexing, rvalueMatrixMultiSingle) {
  Eigen::MatrixXd x(3,4);
  x << 
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;
  
  Eigen::VectorXd v = rvalue(x, index_list(index_omni(), index_uni(2)));
  EXPECT_EQ(3, v.size());
  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(j + 0.1, v(j));
  test_out_of_range(x, index_list(index_omni(), index_uni(0)));
  test_out_of_range(x, index_list(index_omni(), index_uni(20)));

  v = rvalue(x, index_list(index_min(2), index_uni(3)));
  EXPECT_EQ(2, v.size());
  for (int j = 0; j < 2; ++j)
    EXPECT_EQ(1 + j + 0.2, v(j));
  test_out_of_range(x, index_list(index_min(0), index_uni(3)));
  test_out_of_range(x, index_list(index_min(2), index_uni(0)));
  test_out_of_range(x, index_list(index_min(2), index_uni(30)));
}

TEST(ModelIndexing, rvalueMatrixMultiMulti) {
  Eigen::MatrixXd x(3,4);
  x << 
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;

  Eigen::MatrixXd y = rvalue(x, index_list(index_omni(), index_omni()));
  EXPECT_EQ(x.rows(), y.rows());
  EXPECT_EQ(x.cols(), y.cols());
  for (int i = 0; i < x.rows(); ++i)
    for (int j = 0; j < x.cols(); ++j)
      EXPECT_FLOAT_EQ(x(i,j), y(i,j));

  y = rvalue(x, index_list(index_min(2), index_min(3)));
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(2, y.cols());
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      EXPECT_FLOAT_EQ(i + 1 + (j + 2) / 10.0, y(i,j));
  test_out_of_range(x, index_list(index_min(0), index_min(3)));
  test_out_of_range(x, index_list(index_min(2), index_min(0)));
}


  
