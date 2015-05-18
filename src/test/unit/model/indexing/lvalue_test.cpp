#include <iostream>
#include <vector>
#include <stan/model/indexing/lvalue.hpp>
#include <gtest/gtest.h>

using stan::model::nil_index_list;
using stan::model::cons_index_list;
using stan::model::index_uni;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_min;
using stan::model::index_max;
using stan::model::index_min_max;
using stan::model::assign;
using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

TEST(ModelIndexing, lvalueNil) {
  double x = 3;
  double y = 5;
  assign(x, nil_index_list(), y);
  EXPECT_FLOAT_EQ(5, x);

  vector<double> xs;
  xs.push_back(3);
  xs.push_back(5);
  vector<double> ys;
  ys.push_back(13);
  ys.push_back(15);
  assign(xs, nil_index_list(), ys);
  EXPECT_FLOAT_EQ(ys[0], xs[0]);
  EXPECT_FLOAT_EQ(ys[1], xs[1]);
}
TEST(ModelIndexing, lvalueUni) {
  vector<double> xs;
  xs.push_back(3);
  xs.push_back(5);
  xs.push_back(7);
  double y = 15;
  assign(xs, index_list(index_uni(1)), y);
  EXPECT_FLOAT_EQ(y, xs[1]);
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
  assign(xs, index_list(index_uni(1), index_uni(2)), y);
  EXPECT_FLOAT_EQ(y, xs[1][2]);
}
TEST(ModelIndexing, lvalueMulti) {
  vector<double> x;
  for (int i = 0; i < 10; ++i)
    x.push_back(i);

  vector<double> y;
  y.push_back(8.1);
  y.push_back(9.1);
  
  assign(x, index_list(index_min(8)), y);
  EXPECT_FLOAT_EQ(y[0], x[8]);
  EXPECT_FLOAT_EQ(y[1], x[9]);

  assign(x, index_list(index_max(1)), y);
  EXPECT_FLOAT_EQ(y[0], x[0]);
  EXPECT_FLOAT_EQ(y[1], x[1]);
  EXPECT_FLOAT_EQ(2, x[2]);
  
  vector<int> ns;
  ns.push_back(3);
  ns.push_back(5);
  assign(x, index_list(index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y[0], x[3]);
  EXPECT_FLOAT_EQ(y[1], x[5]);
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

  assign(xs, index_list(index_min(8), index_max(2)), ys);
  
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ(ys[i][j], xs[8 + i][j]);
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

  assign(xs, index_list(index_uni(3), index_min_max(2, 4)), ys);
  
  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(ys[j], xs[3][j + 2]);
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

  assign(xs, index_list(index_min_max(4, 6), index_uni(7)), ys);

  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(ys[j], xs[j + 4][7]);
}
TEST(ModelIndexing, lvalueVecUni) {
  VectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  double y = 13;
  assign(xs, index_list(index_uni(2)), y);
  EXPECT_FLOAT_EQ(y, xs(2));
}
TEST(ModelIndexing, lvalueRowVecUni) {
  RowVectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  double y = 13;
  assign(xs, index_list(index_uni(2)), y);
  EXPECT_FLOAT_EQ(y, xs(2));
}
TEST(ModelIndexing, lvalueVecMulti) {
  VectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  VectorXd ys(3);
  ys << 10, 11, 12;
  assign(xs, index_list(index_min(2)), ys);
  EXPECT_FLOAT_EQ(ys(0), xs(2));
  EXPECT_FLOAT_EQ(ys(1), xs(3));
  EXPECT_FLOAT_EQ(ys(2), xs(4));

  xs << 0, 1, 2, 3, 4;
  vector<int> ns;
  ns.push_back(3);
  ns.push_back(0);
  ns.push_back(2);
  assign(xs, index_list(index_multi(ns)), ys);
  EXPECT_FLOAT_EQ(ys(0), xs(3));
  EXPECT_FLOAT_EQ(ys(1), xs(0));
  EXPECT_FLOAT_EQ(ys(2), xs(2));
} 
TEST(ModelIndexing, lvalueRowVecMulti) {
  RowVectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  RowVectorXd ys(3);
  ys << 10, 11, 12;
  assign(xs, index_list(index_min(2)), ys);
  EXPECT_FLOAT_EQ(ys(0), xs(2));
  EXPECT_FLOAT_EQ(ys(1), xs(3));
  EXPECT_FLOAT_EQ(ys(2), xs(4));

  xs << 0, 1, 2, 3, 4;
  vector<int> ns;
  ns.push_back(3);
  ns.push_back(0);
  ns.push_back(2);
  assign(xs, index_list(index_multi(ns)), ys);
  EXPECT_FLOAT_EQ(ys(0), xs(3));
  EXPECT_FLOAT_EQ(ys(1), xs(0));
  EXPECT_FLOAT_EQ(ys(2), xs(2));
}  
TEST(ModelIndexing, lvalueMatrixUni) {
  MatrixXd x(3,4);
  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;
  
  RowVectorXd y(4);
  y << 10.0, 10.1, 10.2, 10.3;
  
  assign(x, index_list(index_uni(2)), y);
  for (int j = 0; j < 4; ++j)
    EXPECT_FLOAT_EQ(x(2,j), y(j));
}
TEST(ModelIndexing, lvalueMatrixMulti) {
  MatrixXd x(3,4);
  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;
  
  MatrixXd y(2, 4);
  y << 
    10.0, 10.1, 10.2, 10.3, 
    11.0, 11.1, 11.2, 11.3;
  
  assign(x, index_list(index_min(1)), y);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(y(i, j), x(i + 1, j));
}
TEST(ModelIndexing, lvalueMatrixUniUni) {
  MatrixXd x(3,4);
  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;

  double y = 10.12;
  assign(x, index_list(index_uni(1), index_uni(2)), y);
  EXPECT_FLOAT_EQ(y, x(1,2));
}
TEST(ModelIndexing, lvalueMatrixUniMulti) {
  MatrixXd x(3,4);
  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(3);
  y << 10, 11, 12;

  assign(x, index_list(index_uni(1), index_min_max(1,3)), y);
  EXPECT_FLOAT_EQ(y(0), x(1,1));
  EXPECT_FLOAT_EQ(y(1), x(1,2));
  EXPECT_FLOAT_EQ(y(2), x(1,3));

  vector<int> ns;
  ns.push_back(3);
  ns.push_back(0);
  ns.push_back(2);
  assign(x, index_list(index_uni(2), index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y(0), x(2, 3));
  EXPECT_FLOAT_EQ(y(1), x(2, 0));
  EXPECT_FLOAT_EQ(y(2), x(2, 2));
}
TEST(ModelIndexing, lvalueMatrixMultiUni) {
  MatrixXd x(3,4);
  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;

  VectorXd y(2);
  y << 10, 11;

  assign(x, index_list(index_min_max(1,2), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y(0), x(1,3));
  EXPECT_FLOAT_EQ(y(1), x(2,3));

  vector<int> ns;
  ns.push_back(2);
  ns.push_back(0);
  assign(x, index_list(index_multi(ns), index_uni(2)), y);
  EXPECT_FLOAT_EQ(y(0), x(2, 2));
  EXPECT_FLOAT_EQ(y(1), x(0, 2));
}

TEST(ModelIndexing, lvalueMatrixMultiMulti) {
  MatrixXd x(3,4);
  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;

  MatrixXd y(2,3);
  y << 
    10, 11, 12,
    20, 21, 22;

  assign(x, index_list(index_min_max(1,2), index_min(1)), y);
  EXPECT_FLOAT_EQ(y(0,0), x(1,1));
  EXPECT_FLOAT_EQ(y(0,1), x(1,2));
  EXPECT_FLOAT_EQ(y(0,2), x(1,3));
  EXPECT_FLOAT_EQ(y(1,0), x(2,1));
  EXPECT_FLOAT_EQ(y(1,1), x(2,2));
  EXPECT_FLOAT_EQ(y(1,2), x(2,3));

  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;
  vector<int> ms;
  ms.push_back(2);
  ms.push_back(0);

  vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(0);
  assign(x, index_list(index_multi(ms), index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y(0,0), x(2,1));
  EXPECT_FLOAT_EQ(y(0,1), x(2,2));
  EXPECT_FLOAT_EQ(y(0,2), x(2,0));
  EXPECT_FLOAT_EQ(y(1,0), x(0,1));
  EXPECT_FLOAT_EQ(y(1,1), x(0,2));
  EXPECT_FLOAT_EQ(y(1,2), x(0,0));

}
