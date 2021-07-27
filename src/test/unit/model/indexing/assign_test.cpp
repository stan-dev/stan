#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/assign.hpp>
#include <stan/model/indexing/rvalue.hpp>
#include <stan/math/rev.hpp>
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

template <typename T1, typename T2, typename... I>
void test_throw(T1& lhs, const T2& rhs, const I&... idxs) {
  EXPECT_THROW(stan::model::assign(lhs, rhs, "", idxs...), std::out_of_range);
}

template <typename T1, typename T2, typename... I>
void test_throw_ia(T1& lhs, const T2& rhs, const I&... idxs) {
  EXPECT_THROW(stan::model::assign(lhs, rhs, "", idxs...),
               std::invalid_argument);
}

TEST(ModelIndexing, lvalueNil) {
  double x = 3;
  double y = 5;
  assign(x, y, "");
  EXPECT_FLOAT_EQ(5, x);

  vector<double> xs;
  xs.push_back(3);
  xs.push_back(5);
  vector<double> ys;
  ys.push_back(13);
  ys.push_back(15);
  assign(xs, ys, "");
  EXPECT_FLOAT_EQ(ys[0], xs[0]);
  EXPECT_FLOAT_EQ(ys[1], xs[1]);
}

TEST(ModelIndexing, lvalueUni) {
  vector<double> xs;
  xs.push_back(3);
  xs.push_back(5);
  xs.push_back(7);
  double y = 15;
  assign(xs, y, "", index_uni(2));
  EXPECT_FLOAT_EQ(y, xs[1]);

  test_throw(xs, y, index_uni(0));
  test_throw(xs, y, index_uni(4));
}

TEST(ModelIndexing, lvalueUniEigen) {
  Eigen::VectorXd xs(3);
  xs << 3, 5, 7;
  double y = 15;
  assign(xs, y, "", index_uni(2));
  EXPECT_FLOAT_EQ(y, xs[1]);
  double z = 10;
  assign(xs.segment(0, 3), z, "", index_uni(2));
  EXPECT_FLOAT_EQ(z, xs[1]);
  assign(xs.segment(0, 3).array(), z, "", index_uni(2));
  EXPECT_FLOAT_EQ(z, xs[1]);

  test_throw(xs, y, index_uni(0));
  test_throw(xs, y, index_uni(4));
}

TEST(model_indexing, assign_eigvec_scalar_uni_index_segment) {
  VectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  double y = 13;
  assign(lhs_x.segment(0, 5), y, "", index_uni(3));
  EXPECT_FLOAT_EQ(y, lhs_x(2));

  test_throw(lhs_x, y, index_uni(0));
  test_throw(lhs_x, y, index_uni(6));
}

TEST(model_indexing, assign_eigrowvec_scalar_uni_index_segment) {
  RowVectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  double y = 13;
  assign(lhs_x.segment(0, 5), y, "", index_uni(3));
  EXPECT_FLOAT_EQ(y, lhs_x(2));
  test_throw(lhs_x, y, index_uni(0));
  test_throw(lhs_x, y, index_uni(6));
}

TEST(ModelIndexing, lvalueUniUni) {
  vector<double> xs0;
  xs0.push_back(0.0);
  xs0.push_back(0.1);
  xs0.push_back(0.2);

  vector<double> xs1;
  xs1.push_back(1.0);
  xs1.push_back(1.1);
  xs1.push_back(1.2);

  vector<vector<double> > xs;
  xs.push_back(xs0);
  xs.push_back(xs1);

  double y = 15;
  assign(xs, y, "", index_uni(2), index_uni(3));
  EXPECT_FLOAT_EQ(y, xs[1][2]);

  test_throw(xs, y, index_uni(0), index_uni(3));
  test_throw(xs, y, index_uni(2), index_uni(0));
  test_throw(xs, y, index_uni(10), index_uni(3));
  test_throw(xs, y, index_uni(2), index_uni(10));
}

TEST(ModelIndexing, lvalueUniUniEigen) {
  Eigen::VectorXd xs0(3);
  xs0 << 0.0, 0.1, 0.2;

  Eigen::VectorXd xs1(3);
  xs1 << 1.0, 1.1, 1.2;

  vector<Eigen::VectorXd> xs;
  xs.push_back(xs0);
  xs.push_back(xs1);

  double y = 15;
  assign(xs, y, "", index_uni(2), index_uni(3));
  EXPECT_FLOAT_EQ(y, xs[1][2]);

  test_throw(xs, y, index_uni(0), index_uni(3));
  test_throw(xs, y, index_uni(2), index_uni(0));
  test_throw(xs, y, index_uni(10), index_uni(3));
  test_throw(xs, y, index_uni(2), index_uni(10));
}

TEST(ModelIndexing, lvalueMulti) {
  vector<double> x;
  for (int i = 0; i < 10; ++i)
    x.push_back(i);

  vector<double> y;
  y.push_back(8.1);
  y.push_back(9.1);

  assign(x, y, "", index_min(9));
  EXPECT_FLOAT_EQ(y[0], x[8]);
  EXPECT_FLOAT_EQ(y[1], x[9]);
  test_throw_ia(x, y, index_min(0));

  assign(x, y, "", index_max(2));
  EXPECT_FLOAT_EQ(y[0], x[0]);
  EXPECT_FLOAT_EQ(y[1], x[1]);
  EXPECT_FLOAT_EQ(2, x[2]);
  test_throw_ia(x, y, index_max(10));

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(6);
  assign(x, y, "", index_multi(ns));
  EXPECT_FLOAT_EQ(y[0], x[3]);
  EXPECT_FLOAT_EQ(y[1], x[5]);

  ns[0] = 0;
  test_throw(x, y, index_multi(ns));

  ns[0] = 11;
  test_throw(x, y, index_multi(ns));

  ns.push_back(3);
  test_throw_ia(x, y, index_multi(ns));
}

TEST(ModelIndexing, lvalueMultiEigen) {
  Eigen::VectorXd x(10);
  for (int i = 0; i < 10; ++i) {
    x(i) = i;
  }

  Eigen::VectorXd y(2);
  y << 8.1, 9.1;

  assign(x, y, "", index_min(9));
  EXPECT_FLOAT_EQ(y[0], x[8]);
  EXPECT_FLOAT_EQ(y[1], x[9]);
  test_throw(x, y, index_min(0));

  assign(x, y, "", index_max(2));
  EXPECT_FLOAT_EQ(y[0], x[0]);
  EXPECT_FLOAT_EQ(y[1], x[1]);
  EXPECT_FLOAT_EQ(2, x[2]);
  test_throw_ia(x, y, index_max(10));

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(6);
  assign(x, y, "", index_multi(ns));
  EXPECT_FLOAT_EQ(y[0], x[3]);
  EXPECT_FLOAT_EQ(y[1], x[5]);

  ns[0] = 0;
  test_throw(x, y, index_multi(ns));

  ns[0] = 11;
  test_throw(x, y, index_multi(ns));

  ns.push_back(3);
  test_throw_ia(x, y, index_multi(ns));
}

TEST(ModelIndexing, lvalueMultiMulti) {
  vector<vector<double> > xs;
  for (int i = 0; i < 10; ++i) {
    vector<double> xsi;
    for (int j = 0; j < 20; ++j)
      xsi.push_back(i + j / 10.0);
    xs.push_back(xsi);
  }

  vector<vector<double> > ys;
  for (int i = 0; i < 2; ++i) {
    vector<double> ysi;
    for (int j = 0; j < 3; ++j)
      ysi.push_back(10 + i + j / 10.0);
    ys.push_back(ysi);
  }

  assign(xs, ys, "", index_min(9), index_max(3));

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ(ys[i][j], xs[8 + i][j]);

  test_throw_ia(xs, ys, index_min(7), index_max(3));
  test_throw_ia(xs, ys, index_min(9), index_max(2));
}

TEST(ModelIndexing, lvalueMultiMultiEigen) {
  vector<Eigen::VectorXd> xs;
  for (int i = 0; i < 10; ++i) {
    Eigen::VectorXd xsi(20);
    for (int j = 0; j < 20; ++j) {
      xsi(j) = (i + j / 10.0);
    }
    xs.push_back(xsi);
  }

  vector<Eigen::VectorXd> ys;
  for (int i = 0; i < 2; ++i) {
    Eigen::VectorXd ysi(3);
    for (int j = 0; j < 3; ++j) {
      ysi(j) = (10 + i + j / 10.0);
    }
    ys.push_back(ysi);
  }

  assign(xs, ys, "", index_min(9), index_max(3));

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ(ys[i][j], xs[8 + i][j]);

  test_throw_ia(xs, ys, index_min(7), index_max(3));
  test_throw_ia(xs, ys, index_min(9), index_max(2));
}

TEST(ModelIndexing, lvalueUniMulti) {
  vector<vector<double> > xs;
  for (int i = 0; i < 10; ++i) {
    vector<double> xsi;
    for (int j = 0; j < 20; ++j)
      xsi.push_back(i + j / 10.0);
    xs.push_back(xsi);
  }

  vector<double> ys;
  for (int i = 0; i < 3; ++i)
    ys.push_back(10 + i);

  assign(xs, ys, "", index_uni(4), index_min_max(3, 5));

  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(ys[j], xs[3][j + 2]);

  test_throw(xs, ys, index_uni(0), index_min_max(3, 5));
  test_throw(xs, ys, index_uni(11), index_min_max(3, 5));
  test_throw_ia(xs, ys, index_uni(4), index_min_max(2, 5));
}

TEST(ModelIndexing, lvalueMultiUni) {
  vector<vector<double> > xs;
  for (int i = 0; i < 10; ++i) {
    vector<double> xsi;
    for (int j = 0; j < 20; ++j)
      xsi.push_back(i + j / 10.0);
    xs.push_back(xsi);
  }

  vector<double> ys;
  for (int i = 0; i < 3; ++i)
    ys.push_back(10 + i);

  assign(xs, ys, "", index_min_max(5, 7), index_uni(8));

  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(ys[j], xs[j + 4][7]);

  test_throw_ia(xs, ys, index_min_max(3, 6), index_uni(7));
  test_throw(xs, ys, index_min_max(4, 6), index_uni(0));
  test_throw(xs, ys, index_min_max(4, 6), index_uni(30));
}

TEST(ModelIndexing, lvalueVecUni) {
  VectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  double y = 13;
  assign(xs, y, "", index_uni(3));
  EXPECT_FLOAT_EQ(y, xs(2));

  test_throw(xs, y, index_uni(0));
  test_throw(xs, y, index_uni(6));
}

TEST(ModelIndexing, lvalueRowVecUni) {
  RowVectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  double y = 13;
  assign(xs, y, "", index_uni(3));
  EXPECT_FLOAT_EQ(y, xs(2));
  test_throw(xs, y, index_uni(0));
  test_throw(xs, y, index_uni(6));
}

TEST(ModelIndexing, lvalueVecMulti) {
  VectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  VectorXd ys(3);
  ys << 10, 11, 12;
  assign(xs, ys, "", index_min(3));
  EXPECT_FLOAT_EQ(ys(0), xs(2));
  EXPECT_FLOAT_EQ(ys(1), xs(3));
  EXPECT_FLOAT_EQ(ys(2), xs(4));
  test_throw(xs, ys, index_min(0));
  test_throw_ia(xs, VectorXd::Ones(7), index_min(1));

  xs << 0, 1, 2, 3, 4;
  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(xs, ys, "", index_multi(ns));
  EXPECT_FLOAT_EQ(ys(0), xs(3));
  EXPECT_FLOAT_EQ(ys(1), xs(0));
  EXPECT_FLOAT_EQ(ys(2), xs(2));
  test_throw_ia(xs, VectorXd::Ones(7), index_multi(ns));

  ns[ns.size() - 1] = 0;
  test_throw(xs, ys, index_multi(ns));

  ns[ns.size() - 1] = 10;
  test_throw(xs, ys, index_multi(ns));

  ns[ns.size() - 1] = 3;
  ns.push_back(1);
  test_throw_ia(xs, ys, index_multi(ns));
}

TEST(ModelIndexing, lvalueRowVecMulti) {
  RowVectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  RowVectorXd ys(3);
  ys << 10, 11, 12;
  assign(xs, ys, "", index_min(3));
  EXPECT_FLOAT_EQ(ys(0), xs(2));
  EXPECT_FLOAT_EQ(ys(1), xs(3));
  EXPECT_FLOAT_EQ(ys(2), xs(4));
  test_throw_ia(xs, ys, index_min(2));
  test_throw_ia(xs, RowVectorXd::Ones(4), index_min(3));
  test_throw(xs, ys, index_min(0));

  xs << 0, 1, 2, 3, 4;
  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(xs, ys, "", index_multi(ns));
  EXPECT_FLOAT_EQ(ys(0), xs(3));
  EXPECT_FLOAT_EQ(ys(1), xs(0));
  EXPECT_FLOAT_EQ(ys(2), xs(2));

  ns[ns.size() - 1] = 0;
  test_throw(xs, ys, index_multi(ns));

  ns[ns.size() - 1] = 10;
  test_throw(xs, ys, index_multi(ns));

  ns[ns.size() - 1] = 3;
  ns.push_back(1);
  test_throw_ia(xs, ys, index_multi(ns));
}

TEST(ModelIndexing, lvalueMatrixUni) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(4);
  y << 10.0, 10.1, 10.2, 10.3;

  assign(x, y, "", index_uni(3));
  for (int j = 0; j < 4; ++j)
    EXPECT_FLOAT_EQ(x(2, j), y(j));

  test_throw(x, y, index_uni(0));
  test_throw(x, y, index_uni(5));
}

TEST(ModelIndexing, lvalueMatrixMin) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y(2, 4);
  y << 10.0, 10.1, 10.2, 10.3, 11.0, 11.1, 11.2, 11.3;

  assign(x, y, "", index_min(2));
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(y(i, j), x(i + 1, j));
  test_throw_ia(x, y, index_min(1));

  MatrixXd z(1, 2);
  z << 10, 20;
  test_throw_ia(x, z, index_min(1));
  test_throw_ia(x, z, index_min(2));
}

TEST(ModelIndexing, lvalueMatrixMax) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y(2, 4);
  y << 10.0, 10.1, 10.2, 10.3, 11.0, 11.1, 11.2, 11.3;

  assign(x, y, "", index_max(2));
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(y(i, j), x(i, j));
  test_throw(x, y, index_max(0));
  test_throw(x, y, index_max(8));
  test_throw_ia(x, y, index_max(1));

  MatrixXd z(1, 2);
  z << 10, 20;
  test_throw_ia(x, z, index_max(1));
  test_throw_ia(x, z, index_max(2));
}

TEST(ModelIndexing, lvalueMatrixUniUni) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  double y = 10.12;
  assign(x, y, "", index_uni(2), index_uni(3));
  EXPECT_FLOAT_EQ(y, x(1, 2));

  test_throw(x, y, index_uni(0), index_uni(3));
  test_throw(x, y, index_uni(2), index_uni(0));
  test_throw(x, y, index_uni(4), index_uni(3));
  test_throw(x, y, index_uni(2), index_uni(5));
}

TEST(ModelIndexing, lvalueMatrixUniMulti) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(3);
  y << 10, 11, 12;
  assign(x, y, "", index_uni(2), index_min_max(2, 4));
  EXPECT_FLOAT_EQ(y(0), x(1, 1));
  EXPECT_FLOAT_EQ(y(1), x(1, 2));
  EXPECT_FLOAT_EQ(y(2), x(1, 3));

  test_throw(x, y, index_uni(0), index_min_max(2, 4));
  test_throw(x, y, index_uni(5), index_min_max(2, 4));
  test_throw(x, y, index_uni(2), index_min_max(0, 2));
  test_throw(x, y, index_uni(2), index_min_max(2, 5));
  test_throw_ia(x, RowVectorXd::Ones(4), index_uni(2), index_min_max(2, 4));

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(x, y, "", index_uni(3), index_multi(ns));
  EXPECT_FLOAT_EQ(y(0), x(2, 3));
  EXPECT_FLOAT_EQ(y(1), x(2, 0));
  EXPECT_FLOAT_EQ(y(2), x(2, 2));

  ns[ns.size() - 1] = 0;
  test_throw(x, y, index_uni(3), index_multi(ns));

  ns[ns.size() - 1] = 20;
  test_throw(x, y, index_uni(3), index_multi(ns));

  ns.push_back(2);
  test_throw_ia(x, y, index_uni(3), index_multi(ns));
}

TEST(ModelIndexing, lvalueMatrixMinMaxRow) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y = MatrixXd::Ones(2, 4);
  assign(x, y, "", index_min_max(1, 2));
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(x(i, j), y(i, j));
    }
  }
  test_throw(x, y, index_min_max(2, 4));
  test_throw(x, y, index_min_max(0, 1));
  test_throw_ia(x, y, index_min_max(1, 3));
}

TEST(ModelIndexing, lvalueMatrixNegativeMinMaxRow) {
  MatrixXd x(5, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1,
      3.2, 3.3, 4.0, 4.1, 4.2, 4.3;

  MatrixXd y = MatrixXd::Ones(3, 4);
  assign(x, y, "", index_min_max(3, 1));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(x(i, j), y(i, j));
    }
  }
  test_throw(x, y, index_min_max(2, 6));
  test_throw(x, y, index_min_max(1, 0));
  test_throw_ia(x, y, index_min_max(2, 1));
}

TEST(ModelIndexing, lvalueMatrixMultiUni) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  VectorXd y(2);
  y << 10, 11;
  assign(x, y, "", index_min_max(2, 3), index_uni(4));
  EXPECT_FLOAT_EQ(y(0), x(1, 3));
  EXPECT_FLOAT_EQ(y(1), x(2, 3));

  test_throw(x, y, index_min_max(2, 3), index_uni(0));
  test_throw(x, y, index_min_max(2, 3), index_uni(5));
  test_throw(x, y, index_min_max(0, 1), index_uni(4));
  test_throw_ia(x, y, index_min_max(1, 3), index_uni(4));

  vector<int> ns;
  ns.push_back(3);
  ns.push_back(1);
  assign(x, y, "", index_multi(ns), index_uni(3));
  EXPECT_FLOAT_EQ(y(0), x(2, 2));
  EXPECT_FLOAT_EQ(y(1), x(0, 2));

  ns[ns.size() - 1] = 0;
  test_throw(x, y, index_multi(ns), index_uni(3));

  ns[ns.size() - 1] = 20;
  test_throw(x, y, index_multi(ns), index_uni(3));

  ns.push_back(2);
  test_throw_ia(x, y, index_multi(ns), index_uni(3));
}

TEST(ModelIndexing, lvalueMatrixMultiMulti) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y(2, 3);
  y << 10, 11, 12, 20, 21, 22;
  assign(x, y, "", index_min_max(2, 3), index_min(2));
  EXPECT_FLOAT_EQ(y(0, 0), x(1, 1));
  EXPECT_FLOAT_EQ(y(0, 1), x(1, 2));
  EXPECT_FLOAT_EQ(y(0, 2), x(1, 3));
  EXPECT_FLOAT_EQ(y(1, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y(1, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y(1, 2), x(2, 3));

  test_throw(x, y, index_min_max(2, 3), index_min(0));
  test_throw(x, y, index_min_max(2, 3), index_min(10));
  test_throw_ia(x, y, index_min_max(1, 3), index_min(2));

  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
  vector<int> ms;
  ms.push_back(3);
  ms.push_back(1);

  vector<int> ns;
  ns.push_back(2);
  ns.push_back(3);
  ns.push_back(1);
  assign(x, y, "", index_multi(ms), index_multi(ns));
  EXPECT_FLOAT_EQ(y(0, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y(0, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y(0, 2), x(2, 0));
  EXPECT_FLOAT_EQ(y(1, 0), x(0, 1));
  EXPECT_FLOAT_EQ(y(1, 1), x(0, 2));
  EXPECT_FLOAT_EQ(y(1, 2), x(0, 0));

  ms[ms.size() - 1] = 0;
  test_throw(x, y, index_multi(ms), index_multi(ns));

  ms[ms.size() - 1] = 10;
  test_throw(x, y, index_multi(ms), index_multi(ns));

  ms[ms.size() - 1] = 1;  // back to original valid value
  ns[ns.size() - 1] = 0;
  test_throw(x, y, index_multi(ms), index_multi(ns));

  ns[ns.size() - 1] = 10;
  test_throw(x, y, index_multi(ms), index_multi(ns));
}
TEST(ModelIndexing, doubleToVar) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  using stan::model::assign;
  using stan::model::index_omni;
  using std::vector;

  vector<double> xs;
  xs.push_back(1);
  xs.push_back(2);
  xs.push_back(3);
  vector<vector<double> > xss;
  xss.push_back(xs);

  vector<var> ys(3);
  vector<vector<var> > yss;
  yss.push_back(ys);

  assign(yss, xss, "foo", index_omni());

  // test both cases where matrix indexed by rows
  // case 1: double matrix with single multi-index on LHS, var matrix on RHS
  Matrix<var, Dynamic, Dynamic> a(4, 3);
  for (int i = 0; i < 12; ++i)
    a(i) = -(i + 1);

  Matrix<double, Dynamic, Dynamic> b(2, 3);
  b << 1, 2, 3, 4, 5, 6;

  vector<int> is;
  is.push_back(2);
  is.push_back(3);
  assign(a, b, "", index_multi(is));
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ(a(i + 1, j).val(), b(i, j));

  // case 2: double matrix with single multi-index on LHS, row vector
  // on RHS
  Matrix<var, Dynamic, Dynamic> c(4, 3);
  for (int i = 0; i < 12; ++i)
    c(i) = -(i + 1);
  Matrix<double, 1, Dynamic> d(3);
  d << 100, 101, 102;
  assign(c, d, "", index_uni(2));
  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(c(1, j).val(), d(j));
}
TEST(ModelIndexing, resultSizeNegIndexing) {
  using stan::model::assign;
  using stan::model::index_min_max;
  using std::vector;

  vector<double> rhs;
  rhs.push_back(2);
  rhs.push_back(5);
  rhs.push_back(-125);

  vector<double> lhs;
  assign(rhs, lhs, "", index_min_max(1, 0));
  EXPECT_EQ(0, lhs.size());
}

TEST(ModelIndexing, resultSizeIndexingEigen) {
  using stan::model::assign;
  using stan::model::index_min_max;
  using std::vector;
  Eigen::VectorXd lhs(5);
  lhs << 1, 2, 3, 4, 5;
  Eigen::VectorXd rhs(4);
  rhs << 4, 3, 2, 1;
  assign(lhs, rhs, "", index_min_max(1, 4));
  EXPECT_FLOAT_EQ(lhs(0), 4);
  EXPECT_FLOAT_EQ(lhs(1), 3);
  EXPECT_FLOAT_EQ(lhs(2), 2);
  EXPECT_FLOAT_EQ(lhs(3), 1);
  EXPECT_FLOAT_EQ(lhs(4), 5);
}

TEST(ModelIndexing, resultSizeNegIndexingEigen) {
  using stan::model::assign;
  using stan::model::index_min_max;
  using std::vector;
  Eigen::VectorXd lhs(5);
  lhs << 1, 2, 3, 4, 5;
  Eigen::VectorXd rhs(4);
  rhs << 1, 2, 3, 4;
  assign(lhs, rhs, "", index_min_max(4, 1));
  EXPECT_FLOAT_EQ(lhs(0), 4);
  EXPECT_FLOAT_EQ(lhs(1), 3);
  EXPECT_FLOAT_EQ(lhs(2), 2);
  EXPECT_FLOAT_EQ(lhs(3), 1);
  EXPECT_FLOAT_EQ(lhs(4), 5);
}

TEST(ModelIndexing, resultSizePosMinMaxPosMinMaxEigenMatrix) {
  using stan::model::assign;
  using stan::model::index_min_max;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev(5, 5);
  for (int i = 0; i < x.size(); ++i) {
    x(i) = i;
    x_rev(i) = x.size() - i - 1;
  }

  for (int i = 0; i < x.rows(); ++i) {
    Eigen::MatrixXd x_colwise_rev = x_rev.block(0, 0, i + 1, i + 1);
    assign(x, x_rev.block(0, 0, i + 1, i + 1), "", index_min_max(1, i + 1),
           index_min_max(1, i + 1));
    for (int kk = 0; kk < i; ++kk) {
      for (int jj = 0; jj < i; ++jj) {
        EXPECT_FLOAT_EQ(x(kk, jj), x_rev(kk, jj));
      }
    }
    for (int j = 0; j < x.size(); ++j) {
      x(j) = j;
    }
  }
}

TEST(ModelIndexing, resultSizePosMinMaxNegMinMaxEigenMatrix) {
  using stan::model::assign;
  using stan::model::index_min_max;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev(5, 5);
  for (int i = 0; i < x.size(); ++i) {
    x(i) = i;
    x_rev(i) = x.size() - i - 1;
  }

  for (int i = 0; i < x.rows(); ++i) {
    Eigen::MatrixXd x_rowwise_reverse
        = x_rev.block(0, 0, i + 1, i + 1).rowwise().reverse();
    assign(x, x_rev.block(0, 0, i + 1, i + 1), "", index_min_max(1, i + 1),
           index_min_max(i + 1, 1));
    for (int kk = 0; kk < i; ++kk) {
      for (int jj = 0; jj < i; ++jj) {
        EXPECT_FLOAT_EQ(x(kk, jj), x_rowwise_reverse(kk, jj));
      }
    }
    for (int j = 0; j < x.size(); ++j) {
      x(j) = j;
    }
  }
}

TEST(ModelIndexing, resultSizeNigMinMaxPosMinMaxEigenMatrix) {
  using stan::model::assign;
  using stan::model::index_min_max;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev(5, 5);
  for (int i = 0; i < x.size(); ++i) {
    x(i) = i;
    x_rev(i) = x.size() - i - 1;
  }

  for (int i = 0; i < x.rows(); ++i) {
    Eigen::MatrixXd x_colwise_reverse
        = x_rev.block(0, 0, i + 1, i + 1).colwise().reverse();
    assign(x, x_rev.block(0, 0, i + 1, i + 1), "", index_min_max(i + 1, 1),
           index_min_max(1, i + 1));
    for (int kk = 0; kk < i; ++kk) {
      for (int jj = 0; jj < i; ++jj) {
        EXPECT_FLOAT_EQ(x(kk, jj), x_colwise_reverse(kk, jj));
      }
    }
    for (int j = 0; j < x.size(); ++j) {
      x(j) = j;
    }
  }
}

TEST(ModelIndexing, resultSizeNegMinMaxNegMinMaxEigenMatrix) {
  using stan::model::assign;
  using stan::model::index_min_max;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev(5, 5);
  for (int i = 0; i < x.size(); ++i) {
    x(i) = i;
    x_rev(i) = x.size() - i - 1;
  }

  for (int i = 0; i < x.rows(); ++i) {
    Eigen::MatrixXd x_reverse = x_rev.block(0, 0, i + 1, i + 1).reverse();
    assign(x, x_rev.block(0, 0, i + 1, i + 1), "", index_min_max(i + 1, 1),
           index_min_max(i + 1, 1));
    for (int kk = 0; kk < i; ++kk) {
      for (int jj = 0; jj < i; ++jj) {
        EXPECT_FLOAT_EQ(x(kk, jj), x_reverse(kk, jj));
      }
    }
    for (int j = 0; j < x.size(); ++j) {
      x(j) = j;
    }
  }
}

TEST(modelIndexing, doubleToVarSimple) {
  using stan::math::var;
  typedef Eigen::MatrixXd mat_d;
  typedef Eigen::Matrix<var, -1, -1> mat_v;

  mat_d a(2, 2);
  a << 1, 2, 3, 4;
  mat_v b;
  assign(b, a, "");
  for (int i = 0; i < a.size(); ++i)
    EXPECT_FLOAT_EQ(a(i), b(i).val());
}

TEST(model_indexing, assign_eigvec_eigvec_index_min) {
  VectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  VectorXd rhs_y(3);
  rhs_y << 10, 11, 12;
  assign(lhs_x, rhs_y, "", index_min(3));
  EXPECT_FLOAT_EQ(rhs_y(0), lhs_x(2));
  EXPECT_FLOAT_EQ(rhs_y(1), lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(2), lhs_x(4));
  test_throw(lhs_x, rhs_y, index_min(0));

  assign(lhs_x, rhs_y.array() + 1.0, "", index_min(3));
  EXPECT_FLOAT_EQ(rhs_y(0) + 1.0, lhs_x(2));
  EXPECT_FLOAT_EQ(rhs_y(1) + 1.0, lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(2) + 1.0, lhs_x(4));
}

TEST(model_indexing, assign_eigvec_eigvec_index_multi) {
  VectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  VectorXd rhs_y(3);
  rhs_y << 10, 11, 12;

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(lhs_x, rhs_y, "", index_multi(ns));
  EXPECT_FLOAT_EQ(rhs_y(0), lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(1), lhs_x(0));
  EXPECT_FLOAT_EQ(rhs_y(2), lhs_x(2));

  assign(lhs_x, rhs_y.array() + 4, "", index_multi(ns));
  EXPECT_FLOAT_EQ(rhs_y(0) + 4, lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(1) + 4, lhs_x(0));
  EXPECT_FLOAT_EQ(rhs_y(2) + 4, lhs_x(2));

  ns[ns.size() - 1] = 0;
  test_throw(lhs_x, rhs_y, index_multi(ns));

  ns[ns.size() - 1] = 10;
  test_throw(lhs_x, rhs_y, index_multi(ns));

  ns[ns.size() - 1] = 3;
  ns.push_back(1);
  test_throw_ia(lhs_x, rhs_y, index_multi(ns));
}

TEST(model_indexing, assign_eigrowvec_eigrowvec_index_min) {
  RowVectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  RowVectorXd rhs_y(3);
  rhs_y << 10, 11, 12;
  assign(lhs_x, rhs_y, "", index_min(3));
  EXPECT_FLOAT_EQ(rhs_y(0), lhs_x(2));
  EXPECT_FLOAT_EQ(rhs_y(1), lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(2), lhs_x(4));
  test_throw(lhs_x, rhs_y, index_min(0));

  assign(lhs_x, rhs_y.array() + 1.0, "", index_min(3));
  EXPECT_FLOAT_EQ(rhs_y(0) + 1.0, lhs_x(2));
  EXPECT_FLOAT_EQ(rhs_y(1) + 1.0, lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(2) + 1.0, lhs_x(4));
}

TEST(model_indexing, assign_eigrowvec_eigrowvec_index_multi) {
  RowVectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  RowVectorXd rhs_y(3);
  rhs_y << 10, 11, 12;

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(lhs_x, rhs_y, "", index_multi(ns));
  EXPECT_FLOAT_EQ(rhs_y(0), lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(1), lhs_x(0));
  EXPECT_FLOAT_EQ(rhs_y(2), lhs_x(2));

  assign(lhs_x, rhs_y.array() + 4, "", index_multi(ns));
  EXPECT_FLOAT_EQ(rhs_y(0) + 4, lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(1) + 4, lhs_x(0));
  EXPECT_FLOAT_EQ(rhs_y(2) + 4, lhs_x(2));

  ns[ns.size() - 1] = 0;
  test_throw(lhs_x, rhs_y, index_multi(ns));

  ns[ns.size() - 1] = 10;
  test_throw(lhs_x, rhs_y, index_multi(ns));

  ns[ns.size() - 1] = 3;
  ns.push_back(1);
  test_throw_ia(lhs_x, rhs_y, index_multi(ns));
}

TEST(model_indexing, assign_densemat_rowvec_uni_index) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(4);
  y << 10.0, 10.1, 10.2, 10.3;

  assign(x, y.array() + 3, "", index_uni(3));
  for (int j = 0; j < 4; ++j)
    EXPECT_FLOAT_EQ(x(2, j), y(j) + 3);

  test_throw(x, y, index_uni(0));
  test_throw(x, y, index_uni(5));
}

TEST(model_indexing, assign_densemat_densemat_index_min) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y(2, 4);
  y << 10.0, 10.1, 10.2, 10.3, 11.0, 11.1, 11.2, 11.3;

  assign(x, y, "", index_min(2));
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(y(i, j), x(i + 1, j));
    }
  }
  assign(x, y.transpose().transpose(), "", index_min(2));
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(y(i, j), x(i + 1, j));
    }
  }
  test_throw_ia(x, y, index_min(1));

  MatrixXd z(1, 2);
  z << 10, 20;
  test_throw_ia(x, z, index_min(1));
  test_throw_ia(x, z, index_min(2));
}

TEST(model_indexing, assign_densemat_scalar_index_uni) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  double y = 10.12;
  assign(x, y, "", index_uni(2), index_uni(3));
  EXPECT_FLOAT_EQ(y, x(1, 2));

  test_throw(x, y, index_uni(0), index_uni(3));
  test_throw(x, y, index_uni(2), index_uni(0));
  test_throw(x, y, index_uni(4), index_uni(3));
  test_throw(x, y, index_uni(2), index_uni(5));
}

TEST(model_indexing, assign_densemat_eigrowvec_uni_index_min_max_index) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(3);
  y << 10, 11, 12;
  assign(x, y, "", index_uni(2), index_min_max(2, 4));
  EXPECT_FLOAT_EQ(y(0), x(1, 1));
  EXPECT_FLOAT_EQ(y(1), x(1, 2));
  EXPECT_FLOAT_EQ(y(2), x(1, 3));

  assign(x, y.array() + 2, "", index_uni(2), index_min_max(2, 4));
  EXPECT_FLOAT_EQ(y(0) + 2, x(1, 1));
  EXPECT_FLOAT_EQ(y(1) + 2, x(1, 2));
  EXPECT_FLOAT_EQ(y(2) + 2, x(1, 3));

  test_throw(x, y, index_uni(0), index_min_max(2, 4));
  test_throw(x, y, index_uni(5), index_min_max(2, 4));
  test_throw(x, y, index_uni(2), index_min_max(0, 2));
  test_throw(x, y, index_uni(2), index_min_max(2, 5));

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(x, y, "", index_uni(3), index_multi(ns));
  EXPECT_FLOAT_EQ(y(0), x(2, 3));
  EXPECT_FLOAT_EQ(y(1), x(2, 0));
  EXPECT_FLOAT_EQ(y(2), x(2, 2));

  assign(x, y.array() + 2, "", index_uni(3), index_multi(ns));
  EXPECT_FLOAT_EQ(y(0) + 2, x(2, 3));
  EXPECT_FLOAT_EQ(y(1) + 2, x(2, 0));
  EXPECT_FLOAT_EQ(y(2) + 2, x(2, 2));

  ns[ns.size() - 1] = 0;
  test_throw(x, y, index_uni(3), index_multi(ns));

  ns[ns.size() - 1] = 20;
  test_throw(x, y, index_uni(3), index_multi(ns));

  ns.push_back(2);
  test_throw_ia(x, y, index_uni(3), index_multi(ns));
}

TEST(model_indexing, assign_densemat_eigvec_min_max_index_uni_index) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  VectorXd y(2);
  y << 10, 11;

  assign(x, y, "", index_min_max(2, 3), index_uni(4));
  EXPECT_FLOAT_EQ(y(0), x(1, 3));
  EXPECT_FLOAT_EQ(y(1), x(2, 3));

  assign(x, y.array() + 2, "", index_min_max(2, 3), index_uni(4));
  EXPECT_FLOAT_EQ(y(0) + 2, x(1, 3));
  EXPECT_FLOAT_EQ(y(1) + 2, x(2, 3));

  test_throw(x, y, index_min_max(2, 3), index_uni(0));
  test_throw(x, y, index_min_max(2, 3), index_uni(5));
  test_throw(x, y, index_min_max(0, 1), index_uni(4));
  test_throw_ia(x, y, index_min_max(1, 3), index_uni(4));

  vector<int> ns;
  ns.push_back(3);
  ns.push_back(1);
  assign(x, y, "", index_multi(ns), index_uni(3));
  EXPECT_FLOAT_EQ(y(0), x(2, 2));
  EXPECT_FLOAT_EQ(y(1), x(0, 2));

  assign(x.block(0, 0, 3, 3), y.array() + 2, "", index_multi(ns), index_uni(3));
  EXPECT_FLOAT_EQ(y(0) + 2, x(2, 2));
  EXPECT_FLOAT_EQ(y(1) + 2, x(0, 2));

  ns[ns.size() - 1] = 0;
  test_throw(x, y, index_multi(ns), index_uni(3));

  ns[ns.size() - 1] = 20;
  test_throw(x, y, index_multi(ns), index_uni(3));

  ns.push_back(2);
  test_throw_ia(x, y, index_multi(ns), index_uni(3));
}

TEST(model_indexing, assign_densemat_densemat_min_max_index_min_index) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y(2, 3);
  y << 10, 11, 12, 20, 21, 22;

  assign(x, y, "", index_min_max(2, 3), index_min(2));
  EXPECT_FLOAT_EQ(y(0, 0), x(1, 1));
  EXPECT_FLOAT_EQ(y(0, 1), x(1, 2));
  EXPECT_FLOAT_EQ(y(0, 2), x(1, 3));
  EXPECT_FLOAT_EQ(y(1, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y(1, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y(1, 2), x(2, 3));

  assign(x.block(0, 0, 3, 3), y.block(0, 0, 2, 2), "", index_min_max(2, 3),
         index_min(2));
  EXPECT_FLOAT_EQ(y(0, 0), x(1, 1));
  EXPECT_FLOAT_EQ(y(0, 1), x(1, 2));
  EXPECT_FLOAT_EQ(y(0, 2), x(1, 3));
  EXPECT_FLOAT_EQ(y(1, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y(1, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y(1, 2), x(2, 3));

  test_throw(x, y, index_min_max(2, 3), index_min(0));
  test_throw(x, y, index_min_max(2, 3), index_min(10));
  test_throw_ia(x, y, index_min_max(1, 3), index_min(2));
}

TEST(model_indexing, assign_densemat_densemat_multi_index_multi_index) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y(2, 3);
  y << 10, 11, 12, 20, 21, 22;
  vector<int> ms;
  ms.push_back(3);
  ms.push_back(1);

  vector<int> ns;
  ns.push_back(2);
  ns.push_back(3);
  ns.push_back(1);
  assign(x, y, "", index_multi(ms), index_multi(ns));
  EXPECT_FLOAT_EQ(y(0, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y(0, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y(0, 2), x(2, 0));
  EXPECT_FLOAT_EQ(y(1, 0), x(0, 1));
  EXPECT_FLOAT_EQ(y(1, 1), x(0, 2));
  EXPECT_FLOAT_EQ(y(1, 2), x(0, 0));

  MatrixXd y2 = y.array() + 2;
  assign(x.block(0, 0, 3, 4), y.array() + 2, "", index_multi(ms),
         index_multi(ns));
  EXPECT_FLOAT_EQ(y2(0, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y2(0, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y2(0, 2), x(2, 0));
  EXPECT_FLOAT_EQ(y2(1, 0), x(0, 1));
  EXPECT_FLOAT_EQ(y2(1, 1), x(0, 2));
  EXPECT_FLOAT_EQ(y2(1, 2), x(0, 0));

  ms[ms.size() - 1] = 0;
  test_throw(x, y, index_multi(ms), index_multi(ns));

  ms[ms.size() - 1] = 10;
  test_throw(x, y, index_multi(ms), index_multi(ns));

  ms[ms.size() - 1] = 1;  // back to original valid value
  ns[ns.size() - 1] = 0;
  test_throw(x, y, index_multi(ms), index_multi(ns));

  ns[ns.size() - 1] = 10;
  test_throw(x, y, index_multi(ms), index_multi(ns));
}
