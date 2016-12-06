#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/lvalue.hpp>
#include <stan/math/rev/mat.hpp>
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

template <typename T1, typename I, typename T2>
void test_throw(T1& lhs, const I& idxs, const T2& rhs) {
  EXPECT_THROW(stan::model::assign(lhs, idxs, rhs), std::out_of_range);
}

template <typename T1, typename I, typename T2>
void test_throw_ia(T1& lhs, const I& idxs, const T2& rhs) {
  EXPECT_THROW(stan::model::assign(lhs, idxs, rhs), std::invalid_argument);
}


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
  assign(xs, index_list(index_uni(2)), y);
  EXPECT_FLOAT_EQ(y, xs[1]);
  
  test_throw(xs, index_list(index_uni(0)), y);
  test_throw(xs, index_list(index_uni(4)), y);
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
  assign(xs, index_list(index_uni(2), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y, xs[1][2]);
  
  test_throw(xs, index_list(index_uni(0), index_uni(3)), y);
  test_throw(xs, index_list(index_uni(2), index_uni(0)), y);
  test_throw(xs, index_list(index_uni(10), index_uni(3)), y);
  test_throw(xs, index_list(index_uni(2), index_uni(10)), y);
}

TEST(ModelIndexing, lvalueMulti) {
  vector<double> x;
  for (int i = 0; i < 10; ++i)
    x.push_back(i);

  vector<double> y;
  y.push_back(8.1);
  y.push_back(9.1);
  
  assign(x, index_list(index_min(9)), y);
  EXPECT_FLOAT_EQ(y[0], x[8]);
  EXPECT_FLOAT_EQ(y[1], x[9]);
  test_throw_ia(x, index_list(index_min(0)), y);

  assign(x, index_list(index_max(2)), y);
  EXPECT_FLOAT_EQ(y[0], x[0]);
  EXPECT_FLOAT_EQ(y[1], x[1]);
  EXPECT_FLOAT_EQ(2, x[2]);
  test_throw_ia(x, index_list(index_max(10)), y);
  
  vector<int> ns;
  ns.push_back(4);
  ns.push_back(6);
  assign(x, index_list(index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y[0], x[3]);
  EXPECT_FLOAT_EQ(y[1], x[5]);

  ns[0] = 0;
  test_throw(x, index_list(index_multi(ns)), y);

  ns[0] = 11;
  test_throw(x, index_list(index_multi(ns)), y);

  ns.push_back(3);
  test_throw_ia(x, index_list(index_multi(ns)), y);
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

  assign(xs, index_list(index_min(9), index_max(3)), ys);
  
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ(ys[i][j], xs[8 + i][j]);

  test_throw_ia(xs, index_list(index_min(7), index_max(3)), ys);
  test_throw_ia(xs, index_list(index_min(9), index_max(2)), ys);
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

  assign(xs, index_list(index_uni(4), index_min_max(3, 5)), ys);
  
  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(ys[j], xs[3][j + 2]);

  test_throw(xs, index_list(index_uni(0), index_min_max(3, 5)), ys);
  test_throw(xs, index_list(index_uni(11), index_min_max(3, 5)), ys);
  test_throw_ia(xs, index_list(index_uni(4), index_min_max(2, 5)), ys);

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

  assign(xs, index_list(index_min_max(5, 7), index_uni(8)), ys);

  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(ys[j], xs[j + 4][7]);

  test_throw_ia(xs, index_list(index_min_max(3, 6), index_uni(7)), ys);
  test_throw(xs, index_list(index_min_max(4, 6), index_uni(0)), ys);
  test_throw(xs, index_list(index_min_max(4, 6), index_uni(30)), ys);
}

TEST(ModelIndexing, lvalueVecUni) {
  VectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  double y = 13;
  assign(xs, index_list(index_uni(3)), y);
  EXPECT_FLOAT_EQ(y, xs(2));

  test_throw(xs, index_list(index_uni(0)), y);
  test_throw(xs, index_list(index_uni(6)), y);
}

TEST(ModelIndexing, lvalueRowVecUni) {
  RowVectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  double y = 13;
  assign(xs, index_list(index_uni(3)), y);
  EXPECT_FLOAT_EQ(y, xs(2));
  test_throw(xs, index_list(index_uni(0)), y);
  test_throw(xs, index_list(index_uni(6)), y);
}

TEST(ModelIndexing, lvalueVecMulti) {
  VectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  VectorXd ys(3);
  ys << 10, 11, 12;
  assign(xs, index_list(index_min(3)), ys);
  EXPECT_FLOAT_EQ(ys(0), xs(2));
  EXPECT_FLOAT_EQ(ys(1), xs(3));
  EXPECT_FLOAT_EQ(ys(2), xs(4));
  test_throw_ia(xs, index_list(index_min(0)), ys);

  xs << 0, 1, 2, 3, 4;
  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(xs, index_list(index_multi(ns)), ys);
  EXPECT_FLOAT_EQ(ys(0), xs(3));
  EXPECT_FLOAT_EQ(ys(1), xs(0));
  EXPECT_FLOAT_EQ(ys(2), xs(2));

  ns[ns.size() - 1] = 0;
  test_throw(xs, index_list(index_multi(ns)), ys);

  ns[ns.size() - 1] = 10;
  test_throw(xs, index_list(index_multi(ns)), ys);

  ns[ns.size() - 1] = 3;
  ns.push_back(1);
  test_throw_ia(xs, index_list(index_multi(ns)), ys);
} 

TEST(ModelIndexing, lvalueRowVecMulti) {
  RowVectorXd xs(5);
  xs << 0, 1, 2, 3, 4;
  RowVectorXd ys(3);
  ys << 10, 11, 12;
  assign(xs, index_list(index_min(3)), ys);
  EXPECT_FLOAT_EQ(ys(0), xs(2));
  EXPECT_FLOAT_EQ(ys(1), xs(3));
  EXPECT_FLOAT_EQ(ys(2), xs(4));
  test_throw_ia(xs, index_list(index_min(2)), ys);
  test_throw_ia(xs, index_list(index_min(0)), ys);

  xs << 0, 1, 2, 3, 4;
  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(xs, index_list(index_multi(ns)), ys);
  EXPECT_FLOAT_EQ(ys(0), xs(3));
  EXPECT_FLOAT_EQ(ys(1), xs(0));
  EXPECT_FLOAT_EQ(ys(2), xs(2));

  ns[ns.size() - 1] = 0;
  test_throw(xs, index_list(index_multi(ns)), ys);

  ns[ns.size() - 1] = 10;
  test_throw(xs, index_list(index_multi(ns)), ys);

  ns[ns.size() - 1] = 3;
  ns.push_back(1);
  test_throw_ia(xs, index_list(index_multi(ns)), ys);
}  

TEST(ModelIndexing, lvalueMatrixUni) {
  MatrixXd x(3,4);
  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;
  
  RowVectorXd y(4);
  y << 10.0, 10.1, 10.2, 10.3;
  
  assign(x, index_list(index_uni(3)), y);
  for (int j = 0; j < 4; ++j)
    EXPECT_FLOAT_EQ(x(2,j), y(j));

  test_throw(x, index_list(index_uni(0)), y);
  test_throw(x, index_list(index_uni(5)), y);
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
  
  assign(x, index_list(index_min(2)), y);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(y(i, j), x(i + 1, j));
  test_throw_ia(x, index_list(index_min(1)), y);

  
  MatrixXd z(1,2);
  z << 10, 20;
  test_throw_ia(x, index_list(index_min(1)), z);
  test_throw_ia(x, index_list(index_min(2)), z);

}

TEST(ModelIndexing, lvalueMatrixUniUni) {
  MatrixXd x(3,4);
  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;
  
  double y = 10.12;
  assign(x, index_list(index_uni(2), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y, x(1,2));

  test_throw(x, index_list(index_uni(0), index_uni(3)), y);
  test_throw(x, index_list(index_uni(2), index_uni(0)), y);
  test_throw(x, index_list(index_uni(4), index_uni(3)), y);
  test_throw(x, index_list(index_uni(2), index_uni(5)), y);
}

TEST(ModelIndexing, lvalueMatrixUniMulti) {
  MatrixXd x(3,4);
  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(3);
  y << 10, 11, 12;

  assign(x, index_list(index_uni(2), index_min_max(2,4)), y);
  EXPECT_FLOAT_EQ(y(0), x(1,1));
  EXPECT_FLOAT_EQ(y(1), x(1,2));
  EXPECT_FLOAT_EQ(y(2), x(1,3));

  test_throw(x, index_list(index_uni(0), index_min_max(2,4)), y);
  test_throw(x, index_list(index_uni(5), index_min_max(2,4)), y);
  test_throw(x, index_list(index_uni(2), index_min_max(0,2)), y);
  test_throw_ia(x, index_list(index_uni(2), index_min_max(2,5)), y);

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(x, index_list(index_uni(3), index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y(0), x(2, 3));
  EXPECT_FLOAT_EQ(y(1), x(2, 0));
  EXPECT_FLOAT_EQ(y(2), x(2, 2));

  ns[ns.size() - 1] = 0;
  test_throw(x, index_list(index_uni(3), index_multi(ns)), y);

  ns[ns.size() - 1] = 20;
  test_throw(x, index_list(index_uni(3), index_multi(ns)), y);

  ns.push_back(2);
  test_throw_ia(x, index_list(index_uni(3), index_multi(ns)), y);
}

TEST(ModelIndexing, lvalueMatrixMultiUni) {
  MatrixXd x(3,4);
  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;

  VectorXd y(2);
  y << 10, 11;

  assign(x, index_list(index_min_max(2,3), index_uni(4)), y);
  EXPECT_FLOAT_EQ(y(0), x(1,3));
  EXPECT_FLOAT_EQ(y(1), x(2,3));

  test_throw(x, index_list(index_min_max(2,3), index_uni(0)), y);
  test_throw(x, index_list(index_min_max(2,3), index_uni(5)), y);
  test_throw(x, index_list(index_min_max(0,1), index_uni(4)), y);
  test_throw_ia(x, index_list(index_min_max(1,3), index_uni(4)), y);

  vector<int> ns;
  ns.push_back(3);
  ns.push_back(1);
  assign(x, index_list(index_multi(ns), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y(0), x(2, 2));
  EXPECT_FLOAT_EQ(y(1), x(0, 2));
 
  ns[ns.size() - 1] = 0;
  test_throw(x, index_list(index_multi(ns), index_uni(3)), y);

  ns[ns.size() - 1] = 20;
  test_throw(x, index_list(index_multi(ns), index_uni(3)), y);

  ns.push_back(2);
  test_throw_ia(x, index_list(index_multi(ns), index_uni(3)), y);
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

  assign(x, index_list(index_min_max(2,3), index_min(2)), y);
  EXPECT_FLOAT_EQ(y(0,0), x(1,1));
  EXPECT_FLOAT_EQ(y(0,1), x(1,2));
  EXPECT_FLOAT_EQ(y(0,2), x(1,3));
  EXPECT_FLOAT_EQ(y(1,0), x(2,1));
  EXPECT_FLOAT_EQ(y(1,1), x(2,2));
  EXPECT_FLOAT_EQ(y(1,2), x(2,3));

  test_throw_ia(x, index_list(index_min_max(2,3), index_min(0)), y);
  test_throw_ia(x, index_list(index_min_max(2,3), index_min(10)), y);
  test_throw_ia(x, index_list(index_min_max(1,3), index_min(2)), y);

  x <<
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    2.0, 2.1, 2.2, 2.3;
  vector<int> ms;
  ms.push_back(3);
  ms.push_back(1);

  vector<int> ns;
  ns.push_back(2);
  ns.push_back(3);
  ns.push_back(1);
  assign(x, index_list(index_multi(ms), index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y(0,0), x(2,1));
  EXPECT_FLOAT_EQ(y(0,1), x(2,2));
  EXPECT_FLOAT_EQ(y(0,2), x(2,0));
  EXPECT_FLOAT_EQ(y(1,0), x(0,1));
  EXPECT_FLOAT_EQ(y(1,1), x(0,2));
  EXPECT_FLOAT_EQ(y(1,2), x(0,0));

  ms[ms.size() - 1] = 0;
  test_throw(x, index_list(index_multi(ms), index_multi(ns)), y);

  ms[ms.size() - 1] = 10;
  test_throw(x, index_list(index_multi(ms), index_multi(ns)), y);

  ms[ms.size() - 1] = 1;  // back to original valid value
  ns[ns.size() - 1] = 0;
  test_throw(x, index_list(index_multi(ms), index_multi(ns)), y);

  ns[ns.size() - 1] = 10;
  test_throw(x, index_list(index_multi(ms), index_multi(ns)), y);
}
TEST(ModelIndexing, doubleToVar) {
  using stan::math::var;
  using std::vector;
  using stan::model::assign;
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::model::cons_list;
  using stan::model::index_omni;
  using stan::model::nil_index_list;

  vector<double> xs;
  xs.push_back(1);
  xs.push_back(2);
  xs.push_back(3);
  vector<vector<double> > xss;
  xss.push_back(xs);

  vector<var> ys(3);
  vector<vector<var> > yss;
  yss.push_back(ys);

  assign(yss, cons_list(index_omni(), nil_index_list()),
         xss, "foo");

  // test both cases where matrix indexed by rows
  // case 1: double matrix with single multi-index on LHS, var matrix on RHS
  Matrix<var, Dynamic, Dynamic> a(4, 3);
  for (int i = 0; i < 12; ++i) a(i) = -(i + 1);

  Matrix<double, Dynamic, Dynamic> b(2,3);
  b << 1, 2, 3, 4, 5, 6;
  
  vector<int> is;
  is.push_back(2);
  is.push_back(3);
  assign(a, cons_list(index_multi(is), nil_index_list()), b);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ(a(i + 1, j).val(), b(i, j));
                      
  // case 2: double matrix with single multi-index on LHS, row vector
  // on RHS
  Matrix<var, Dynamic, Dynamic> c(4, 3);
  for (int i = 0; i < 12; ++i) c(i) = -(i + 1);
  Matrix<double, 1, Dynamic> d(3);
  d << 100, 101, 102;
  assign(c, cons_list(index_uni(2), nil_index_list()), d);
  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(c(1, j).val(), d(j));
}
TEST(ModelIndexing, resultSizeNegIndexing) {
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
  using std::vector;

  vector<double> rhs;
  rhs.push_back(2);
  rhs.push_back(5);
  rhs.push_back(-125);

  vector<double> lhs;
  assign(rhs, cons_list(index_min_max(1, 0), nil_index_list()), lhs);
  EXPECT_EQ(0, lhs.size());
}

