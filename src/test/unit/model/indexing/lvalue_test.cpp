#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/lvalue.hpp>
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

/**
 * Checks that an out of range error occurs
 */
template <typename T1, typename I, typename T2>
void test_throw(T1& lhs, const I& idlhs_x, const T2& rhs) {
  EXPECT_THROW(stan::model::assign(lhs, idlhs_x, rhs), std::out_of_range);
}

/**
 * Checks that an invalid argument is given to assignment
 */
template <typename T1, typename I, typename T2>
void test_throw_ia(T1& lhs, const I& idlhs_x, const T2& rhs) {
  EXPECT_THROW(stan::model::assign(lhs, idlhs_x, rhs), std::invalid_argument);
}

TEST(model_indexing, assign_scalar_nilindex) {
  double x = 3;
  double y = 5;
  assign(x, nil_index_list(), y);
  EXPECT_FLOAT_EQ(5, x);
}

TEST(model_indexing, assign_stdvec_nil_index) {
  vector<double> lhs_x;
  lhs_x.push_back(3);
  lhs_x.push_back(5);
  vector<double> rhs_y;
  rhs_y.push_back(13);
  rhs_y.push_back(15);
  assign(lhs_x, nil_index_list(), rhs_y);
  EXPECT_FLOAT_EQ(rhs_y[0], lhs_x[0]);
  EXPECT_FLOAT_EQ(rhs_y[1], lhs_x[1]);
}

TEST(model_indexing, assign_stdvec_scalar_uni_index) {
  vector<double> lhs_x;
  lhs_x.push_back(3);
  lhs_x.push_back(5);
  lhs_x.push_back(7);
  double y = 15;
  assign(lhs_x, index_list(index_uni(2)), y);
  EXPECT_FLOAT_EQ(y, lhs_x[1]);

  test_throw(lhs_x, index_list(index_uni(0)), y);
  test_throw(lhs_x, index_list(index_uni(4)), y);
}

TEST(model_indexing, assign_stdvec_scalar_uni_index_uniindex) {
  vector<double> lhs_x0;
  lhs_x0.push_back(0.0);
  lhs_x0.push_back(0.1);
  lhs_x0.push_back(0.2);

  vector<double> lhs_x1;
  lhs_x1.push_back(1.0);
  lhs_x1.push_back(1.1);
  lhs_x1.push_back(1.2);

  vector<vector<double>> lhs_x;
  lhs_x.push_back(lhs_x0);
  lhs_x.push_back(lhs_x1);

  double y = 15;
  assign(lhs_x, index_list(index_uni(2), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y, lhs_x[1][2]);

  test_throw(lhs_x, index_list(index_uni(0), index_uni(3)), y);
  test_throw(lhs_x, index_list(index_uni(2), index_uni(0)), y);
  test_throw(lhs_x, index_list(index_uni(10), index_uni(3)), y);
  test_throw(lhs_x, index_list(index_uni(2), index_uni(10)), y);
}

TEST(model_indexing, assign_stdvec_stdvec_min_index_max_index) {
  vector<double> lhs_x;
  for (int i = 0; i < 10; ++i) {
    lhs_x.push_back(i);
  }

  vector<double> rhs_y;
  rhs_y.push_back(8.1);
  rhs_y.push_back(9.1);

  assign(lhs_x, index_list(index_min(9)), rhs_y);
  EXPECT_FLOAT_EQ(rhs_y[0], lhs_x[8]);
  EXPECT_FLOAT_EQ(rhs_y[1], lhs_x[9]);
  test_throw_ia(lhs_x, index_list(index_min(0)), rhs_y);

  assign(lhs_x, index_list(index_max(2)), rhs_y);
  EXPECT_FLOAT_EQ(rhs_y[0], lhs_x[0]);
  EXPECT_FLOAT_EQ(rhs_y[1], lhs_x[1]);
  EXPECT_FLOAT_EQ(2, lhs_x[2]);
  test_throw_ia(lhs_x, index_list(index_max(10)), rhs_y);
}

TEST(model_indexing, assign_stdvec_stdvec_multi_index) {
  vector<double> lhs_x;
  for (int i = 0; i < 10; ++i) {
    lhs_x.push_back(i);
  }

  vector<double> rhs_y;
  rhs_y.push_back(8.1);
  rhs_y.push_back(9.1);

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(6);
  assign(lhs_x, index_list(index_multi(ns)), rhs_y);
  EXPECT_FLOAT_EQ(rhs_y[0], lhs_x[3]);
  EXPECT_FLOAT_EQ(rhs_y[1], lhs_x[5]);

  ns[0] = 0;
  test_throw(lhs_x, index_list(index_multi(ns)), rhs_y);

  ns[0] = 11;
  test_throw(lhs_x, index_list(index_multi(ns)), rhs_y);

  ns.push_back(3);
  test_throw_ia(lhs_x, index_list(index_multi(ns)), rhs_y);
}

TEST(model_indexing, assign_stdvecvec_stdvecvec_multi_min_index_max_index) {
  vector<vector<double>> lhs_x;
  for (int i = 0; i < 10; ++i) {
    vector<double> lhs_xi;
    for (int j = 0; j < 20; ++j) {
      lhs_xi.push_back(i + j / 10.0);
    }
    lhs_x.push_back(lhs_xi);
  }

  vector<vector<double>> rhs_y;
  for (int i = 0; i < 2; ++i) {
    vector<double> rhs_yi;
    for (int j = 0; j < 3; ++j) {
      rhs_yi.push_back(10 + i + j / 10.0);
    }
    rhs_y.push_back(rhs_yi);
  }

  assign(lhs_x, index_list(index_min(9), index_max(3)), rhs_y);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(rhs_y[i][j], lhs_x[8 + i][j]);
    }
  }

  test_throw_ia(lhs_x, index_list(index_min(7), index_max(3)), rhs_y);
  test_throw_ia(lhs_x, index_list(index_min(9), index_max(2)), rhs_y);
}

TEST(model_indexing, assign_stdvecvec_stdvec_index_uni_index_min_max_index) {
  vector<vector<double>> lhs_x;
  for (int i = 0; i < 10; ++i) {
    vector<double> lhs_xi;
    for (int j = 0; j < 20; ++j) {
      lhs_xi.push_back(i + j / 10.0);
    }
    lhs_x.push_back(lhs_xi);
  }

  vector<double> rhs_y;
  for (int i = 0; i < 3; ++i) {
    rhs_y.push_back(10 + i);
  }

  assign(lhs_x, index_list(index_uni(4), index_min_max(3, 5)), rhs_y);

  for (int j = 0; j < 3; ++j) {
    EXPECT_FLOAT_EQ(rhs_y[j], lhs_x[3][j + 2]);
  }

  test_throw(lhs_x, index_list(index_uni(0), index_min_max(3, 5)), rhs_y);
  test_throw(lhs_x, index_list(index_uni(11), index_min_max(3, 5)), rhs_y);
  test_throw_ia(lhs_x, index_list(index_uni(4), index_min_max(2, 5)), rhs_y);
}

TEST(model_indexing, assign_stdvecvec_stdvec_min_max_index_uni_index) {
  vector<vector<double>> lhs_x;
  for (int i = 0; i < 10; ++i) {
    vector<double> lhs_xi;
    for (int j = 0; j < 20; ++j) {
      lhs_xi.push_back(i + j / 10.0);
    }
    lhs_x.push_back(lhs_xi);
  }

  vector<double> rhs_y;
  for (int i = 0; i < 3; ++i) {
    rhs_y.push_back(10 + i);
  }

  assign(lhs_x, index_list(index_min_max(5, 7), index_uni(8)), rhs_y);

  for (int j = 0; j < 3; ++j) {
    EXPECT_FLOAT_EQ(rhs_y[j], lhs_x[j + 4][7]);
  }

  test_throw_ia(lhs_x, index_list(index_min_max(3, 6), index_uni(7)), rhs_y);
  test_throw(lhs_x, index_list(index_min_max(4, 6), index_uni(0)), rhs_y);
  test_throw(lhs_x, index_list(index_min_max(4, 6), index_uni(30)), rhs_y);
}

TEST(model_indexing, assign_eigvec_scalar_uni_index) {
  VectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  double y = 13;
  assign(lhs_x, index_list(index_uni(3)), y);
  EXPECT_FLOAT_EQ(y, lhs_x(2));

  test_throw(lhs_x, index_list(index_uni(0)), y);
  test_throw(lhs_x, index_list(index_uni(6)), y);
}

TEST(model_indexing, assign_eigrowvec_scalar_uni_index) {
  RowVectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  double y = 13;
  assign(lhs_x, index_list(index_uni(3)), y);
  EXPECT_FLOAT_EQ(y, lhs_x(2));
  test_throw(lhs_x, index_list(index_uni(0)), y);
  test_throw(lhs_x, index_list(index_uni(6)), y);
}

TEST(model_indexing, assign_eigvec_eigvec_index_min) {
  VectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  VectorXd rhs_y(3);
  rhs_y << 10, 11, 12;
  assign(lhs_x, index_list(index_min(3)), rhs_y);
  EXPECT_FLOAT_EQ(rhs_y(0), lhs_x(2));
  EXPECT_FLOAT_EQ(rhs_y(1), lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(2), lhs_x(4));
  test_throw_ia(lhs_x, index_list(index_min(0)), rhs_y);

  assign(lhs_x, index_list(index_min(3)), rhs_y.array() + 1.0);
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
  assign(lhs_x, index_list(index_multi(ns)), rhs_y);
  EXPECT_FLOAT_EQ(rhs_y(0), lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(1), lhs_x(0));
  EXPECT_FLOAT_EQ(rhs_y(2), lhs_x(2));

  assign(lhs_x, index_list(index_multi(ns)), rhs_y.array() + 4);
  EXPECT_FLOAT_EQ(rhs_y(0) + 4, lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(1) + 4, lhs_x(0));
  EXPECT_FLOAT_EQ(rhs_y(2) + 4, lhs_x(2));

  ns[ns.size() - 1] = 0;
  test_throw(lhs_x, index_list(index_multi(ns)), rhs_y);

  ns[ns.size() - 1] = 10;
  test_throw(lhs_x, index_list(index_multi(ns)), rhs_y);

  ns[ns.size() - 1] = 3;
  ns.push_back(1);
  test_throw_ia(lhs_x, index_list(index_multi(ns)), rhs_y);
}

TEST(model_indexing, assign_eigrowvec_eigrowvec_index_min) {
  RowVectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  RowVectorXd rhs_y(3);
  rhs_y << 10, 11, 12;
  assign(lhs_x, index_list(index_min(3)), rhs_y);
  EXPECT_FLOAT_EQ(rhs_y(0), lhs_x(2));
  EXPECT_FLOAT_EQ(rhs_y(1), lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(2), lhs_x(4));
  test_throw_ia(lhs_x, index_list(index_min(0)), rhs_y);

  assign(lhs_x, index_list(index_min(3)), rhs_y.array() + 1.0);
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
  assign(lhs_x, index_list(index_multi(ns)), rhs_y);
  EXPECT_FLOAT_EQ(rhs_y(0), lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(1), lhs_x(0));
  EXPECT_FLOAT_EQ(rhs_y(2), lhs_x(2));

  assign(lhs_x, index_list(index_multi(ns)), rhs_y.array() + 4);
  EXPECT_FLOAT_EQ(rhs_y(0) + 4, lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(1) + 4, lhs_x(0));
  EXPECT_FLOAT_EQ(rhs_y(2) + 4, lhs_x(2));

  ns[ns.size() - 1] = 0;
  test_throw(lhs_x, index_list(index_multi(ns)), rhs_y);

  ns[ns.size() - 1] = 10;
  test_throw(lhs_x, index_list(index_multi(ns)), rhs_y);

  ns[ns.size() - 1] = 3;
  ns.push_back(1);
  test_throw_ia(lhs_x, index_list(index_multi(ns)), rhs_y);
}

TEST(model_indexing, assign_eigmatrix_rowvec_uni_index) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(4);
  y << 10.0, 10.1, 10.2, 10.3;

  assign(x, index_list(index_uni(3)), y.array() + 3);
  for (int j = 0; j < 4; ++j)
    EXPECT_FLOAT_EQ(x(2, j), y(j) + 3);

  test_throw(x, index_list(index_uni(0)), y);
  test_throw(x, index_list(index_uni(5)), y);
}

TEST(model_indexing, assign_eigmatrix_eigmatrix_index_min) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y(2, 4);
  y << 10.0, 10.1, 10.2, 10.3, 11.0, 11.1, 11.2, 11.3;

  assign(x, index_list(index_min(2)), y);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(y(i, j), x(i + 1, j));
    }
  }
  assign(x, index_list(index_min(2)), y.transpose().transpose());
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(y(i, j), x(i + 1, j));
    }
  }
  test_throw_ia(x, index_list(index_min(1)), y);

  MatrixXd z(1, 2);
  z << 10, 20;
  test_throw_ia(x, index_list(index_min(1)), z);
  test_throw_ia(x, index_list(index_min(2)), z);
}

TEST(model_indexing, assign_eigmatrix_scalar_index_uni) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  double y = 10.12;
  assign(x, index_list(index_uni(2), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y, x(1, 2));

  test_throw(x, index_list(index_uni(0), index_uni(3)), y);
  test_throw(x, index_list(index_uni(2), index_uni(0)), y);
  test_throw(x, index_list(index_uni(4), index_uni(3)), y);
  test_throw(x, index_list(index_uni(2), index_uni(5)), y);
}

TEST(model_indexing, assign_eigmatrix_eigrowvec_uni_index_min_max_index) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(3);
  y << 10, 11, 12;

  assign(x, index_list(index_uni(2), index_min_max(2, 4)), y);
  EXPECT_FLOAT_EQ(y(0), x(1, 1));
  EXPECT_FLOAT_EQ(y(1), x(1, 2));
  EXPECT_FLOAT_EQ(y(2), x(1, 3));

  assign(x, index_list(index_uni(2), index_min_max(2, 4)), y.array() + 2);
  EXPECT_FLOAT_EQ(y(0) + 2, x(1, 1));
  EXPECT_FLOAT_EQ(y(1) + 2, x(1, 2));
  EXPECT_FLOAT_EQ(y(2) + 2, x(1, 3));

  test_throw(x, index_list(index_uni(0), index_min_max(2, 4)), y);
  test_throw(x, index_list(index_uni(5), index_min_max(2, 4)), y);
  test_throw(x, index_list(index_uni(2), index_min_max(0, 2)), y);
  test_throw_ia(x, index_list(index_uni(2), index_min_max(2, 5)), y);

  vector<int> ns;
  ns.push_back(4);
  ns.push_back(1);
  ns.push_back(3);
  assign(x, index_list(index_uni(3), index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y(0), x(2, 3));
  EXPECT_FLOAT_EQ(y(1), x(2, 0));
  EXPECT_FLOAT_EQ(y(2), x(2, 2));

  assign(x, index_list(index_uni(3), index_multi(ns)), y.array() + 2);
  EXPECT_FLOAT_EQ(y(0) + 2, x(2, 3));
  EXPECT_FLOAT_EQ(y(1) + 2, x(2, 0));
  EXPECT_FLOAT_EQ(y(2) + 2, x(2, 2));

  ns[ns.size() - 1] = 0;
  test_throw(x, index_list(index_uni(3), index_multi(ns)), y);

  ns[ns.size() - 1] = 20;
  test_throw(x, index_list(index_uni(3), index_multi(ns)), y);

  ns.push_back(2);
  test_throw_ia(x, index_list(index_uni(3), index_multi(ns)), y);
}

TEST(model_indexing, assign_eigmatrix_eigvec_min_max_index_uni_index) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  VectorXd y(2);
  y << 10, 11;

  assign(x, index_list(index_min_max(2, 3), index_uni(4)), y);
  EXPECT_FLOAT_EQ(y(0), x(1, 3));
  EXPECT_FLOAT_EQ(y(1), x(2, 3));

  assign(x, index_list(index_min_max(2, 3), index_uni(4)), y.array() + 2);
  EXPECT_FLOAT_EQ(y(0) + 2, x(1, 3));
  EXPECT_FLOAT_EQ(y(1) + 2, x(2, 3));

  test_throw(x, index_list(index_min_max(2, 3), index_uni(0)), y);
  test_throw(x, index_list(index_min_max(2, 3), index_uni(5)), y);
  test_throw(x, index_list(index_min_max(0, 1), index_uni(4)), y);
  test_throw_ia(x, index_list(index_min_max(1, 3), index_uni(4)), y);

  vector<int> ns;
  ns.push_back(3);
  ns.push_back(1);
  assign(x, index_list(index_multi(ns), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y(0), x(2, 2));
  EXPECT_FLOAT_EQ(y(1), x(0, 2));

  assign(x, index_list(index_multi(ns), index_uni(3)), y.array() + 2);
  EXPECT_FLOAT_EQ(y(0) + 2, x(2, 2));
  EXPECT_FLOAT_EQ(y(1) + 2, x(0, 2));

  ns[ns.size() - 1] = 0;
  test_throw(x, index_list(index_multi(ns), index_uni(3)), y);

  ns[ns.size() - 1] = 20;
  test_throw(x, index_list(index_multi(ns), index_uni(3)), y);

  ns.push_back(2);
  test_throw_ia(x, index_list(index_multi(ns), index_uni(3)), y);
}

TEST(model_indexing, assign_eigmatrix_eigmatrix_min_max_index_min_index) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y(2, 3);
  y << 10, 11, 12, 20, 21, 22;

  assign(x, index_list(index_min_max(2, 3), index_min(2)), y);
  EXPECT_FLOAT_EQ(y(0, 0), x(1, 1));
  EXPECT_FLOAT_EQ(y(0, 1), x(1, 2));
  EXPECT_FLOAT_EQ(y(0, 2), x(1, 3));
  EXPECT_FLOAT_EQ(y(1, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y(1, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y(1, 2), x(2, 3));

  test_throw_ia(x, index_list(index_min_max(2, 3), index_min(0)), y);
  test_throw_ia(x, index_list(index_min_max(2, 3), index_min(10)), y);
  test_throw_ia(x, index_list(index_min_max(1, 3), index_min(2)), y);
}

TEST(model_indexing, assign_eigmatrix_eigmatrix_multi_index_multi_index) {
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
  assign(x, index_list(index_multi(ms), index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y(0, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y(0, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y(0, 2), x(2, 0));
  EXPECT_FLOAT_EQ(y(1, 0), x(0, 1));
  EXPECT_FLOAT_EQ(y(1, 1), x(0, 2));
  EXPECT_FLOAT_EQ(y(1, 2), x(0, 0));

  MatrixXd y2 = y.array() + 2;
  assign(x, index_list(index_multi(ms), index_multi(ns)), y.array() + 2);
  EXPECT_FLOAT_EQ(y2(0, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y2(0, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y2(0, 2), x(2, 0));
  EXPECT_FLOAT_EQ(y2(1, 0), x(0, 1));
  EXPECT_FLOAT_EQ(y2(1, 1), x(0, 2));
  EXPECT_FLOAT_EQ(y2(1, 2), x(0, 0));

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
TEST(model_indexing, assign_double_to_var) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_omni;
  using stan::model::nil_index_list;
  using std::vector;

  vector<double> lhs_x;
  lhs_x.push_back(1);
  lhs_x.push_back(2);
  lhs_x.push_back(3);
  vector<vector<double>> lhs_xs;
  lhs_xs.push_back(lhs_x);

  vector<var> rhs_y(3);
  vector<vector<var>> rhs_ys;
  rhs_ys.push_back(rhs_y);

  assign(rhs_ys, cons_list(index_omni(), nil_index_list()), lhs_xs, "foo");

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
  assign(a, cons_list(index_multi(is), nil_index_list()), b);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(a(i + 1, j).val(), b(i, j));
    }
  }

  // case 2: double matrix with single multi-index on LHS, row vector
  // on RHS
  Matrix<var, Dynamic, Dynamic> c(4, 3);
  for (int i = 0; i < 12; ++i) {
    c(i) = -(i + 1);
  }
  Matrix<double, 1, Dynamic> d(3);
  d << 100, 101, 102;
  assign(c, cons_list(index_uni(2), nil_index_list()), d);
  for (int j = 0; j < 3; ++j) {
    EXPECT_FLOAT_EQ(c(1, j).val(), d(j));
  }
}
TEST(model_indexing, assign_result_size_neg_index) {
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

TEST(model_indexing, assign_double_to_var_simple) {
  using stan::math::var;
  using stan::model::nil_index_list;
  typedef Eigen::MatrixXd mat_d;
  typedef Eigen::Matrix<var, -1, -1> mat_v;

  mat_d a(2, 2);
  a << 1, 2, 3, 4;
  mat_v b;
  assign(b, nil_index_list(), a);
  for (int i = 0; i < a.size(); ++i) {
    EXPECT_FLOAT_EQ(a(i), b(i).val());
  }
}
