#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/lvalue_varmat.hpp>
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
      auto generate_linear_var_matrix(Eigen::Index n, Eigen::Index m, double start = 0) {
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
    }
  }
}

TEST_F(VarAssign, nil) {
  using stan::math::var_value;
  auto x = stan::model::test::generate_linear_var_vector(5);
  auto y = stan::model::test::generate_linear_var_vector(5, 1.0);
  assign(x, nil_index_list(), y);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(i + 1, x.val().coeffRef(i));
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

TEST_F(VarAssign, uni_vec) {
  test_uni_vec<true>();
}

TEST_F(VarAssign, uni_rowvec) {
  test_uni_vec<false>();
}

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
  std::conditional_t<ColVec, Eigen::VectorXd, Eigen::RowVectorXd> x_val = x.val();
  assign(x, index_list(index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[3]);
  EXPECT_FLOAT_EQ(y.val()[2], x.val()[1]);

  stan::arena_t<std::vector<int>> x_idx;
  stan::arena_t<std::vector<int>> y_idx;
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = ns.size() - 1; i >= 0; --i) {
//      std::cout << "iter: " << "(" << i << ", " << j << ")\n\n";
    if (!stan::model::internal::check_duplicate(x_idx, ns[i] - 1)) {
      y_idx.push_back(i);
      x_idx.push_back(ns[i] - 1);
    }
  }
/*
  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";
*/
  stan::math::sum(x).grad();
  /*
  std::cout << " post-grad \n";
  std::cout << "x val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";
*/
  for (int i = 0; i < x.rows(); ++i) {
    EXPECT_FLOAT_EQ(x.val()(i), x_val.coeffRef(i));
    if (stan::model::internal::check_duplicate(x_idx, i)) {
      EXPECT_FLOAT_EQ(x.adj()(i), 0) <<
       "Failed for \ni: " << i << " row_idx[i]: " << ns[i] << "\n";
    } else {
      EXPECT_FLOAT_EQ(x.adj()(i), 1) <<
      "Failed for \ni: " << i << " row_idx[i]: " << ns[i] << "\n";
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

TEST_F(VarAssign, multi_vec) {
  test_multi_vec<true>();
}

TEST_F(VarAssign, multi_rowvec) {
  test_multi_vec<false>();
}

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
/*
  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";
*/
  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj()[0], 40);
  EXPECT_FLOAT_EQ(y.adj()[1], 60);
  EXPECT_FLOAT_EQ(x.adj()[1], 0);
  EXPECT_FLOAT_EQ(x.adj()[3], 0);
}

TEST_F(VarAssign, minmax_vec) {
  test_minmax_vec<true>();
}

TEST_F(VarAssign, minmax_rowvec) {
  test_minmax_vec<false>();
}

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
/*
  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";
*/
  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj()[0], 40);
  EXPECT_FLOAT_EQ(y.adj()[1], 60);
  EXPECT_FLOAT_EQ(x.adj()[0], 0);
  EXPECT_FLOAT_EQ(x.adj()[1], 0);
}
TEST_F(VarAssign, max_vec) {
  test_max_vec<true>();
}

TEST_F(VarAssign, max_rowvec) {
  test_max_vec<false>();
}

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
/*
  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";
*/
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
/*
  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";
*/
}

TEST_F(VarAssign, omni_vec) {
  test_omni_vec<true>();
}

TEST_F(VarAssign, omni_rowvec) {
  test_omni_vec<false>();
}

template <typename Vec>
void test_eigvec_var_uni_index_seg() {
  using stan::math::var_value;
  using stan::math::var;
  using stan::math::sum;
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
  test_throw_out_of_range(lhs_x, index_list(index_uni(0)), y);
  test_throw_out_of_range(lhs_x, index_list(index_uni(6)), y);
}

TEST(model_indexing, uni_vec_segment) {
  test_eigvec_var_uni_index_seg<Eigen::VectorXd>();
}
TEST(model_indexing, uni_rowvec_segment) {
  test_eigvec_var_uni_index_seg<Eigen::RowVectorXd>();
}

template <typename Vec>
void test_uni_uni_vec_eigvec() {
  using stan::math::var_value;
  using stan::math::var;
  using stan::math::sum;

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

TEST_F(VarAssign, uniuni_std_vecvec) {
  test_uni_uni_vec_eigvec<Eigen::VectorXd>();
}

TEST_F(VarAssign, uniuni_std_vecrowvec) {
  test_uni_uni_vec_eigvec<Eigen::RowVectorXd>();
}


TEST_F(VarAssign, uni_matrix_rowvec) {
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
/*
  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";
*/
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
/*
  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";
*/
}

TEST_F(VarAssign, multi_matrix_rowvec) {
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
//      std::cout << "iter: " << "(" << i << ", " << j << ")\n\n";
    if (!stan::model::internal::check_duplicate(x_idx, col_idx[i] - 1)) {
      y_idx.push_back(i);
      x_idx.push_back(col_idx[i] - 1);
    }
  }
  /*
  std::cout << "\n before x.val(): \n" << x.val() << "\n";
  std::cout << "\n before y.val(): \n" << y.val() << "\n";
*/
  assign(x, index_list(index_uni(row_idx), index_multi(col_idx)), y);
/*
  std::cout << "\n after x.val(): \n" << x.val() << "\n";
  std::cout << "\n after y.val(): \n" << y.val() << "\n";
  */
  // We use these to check the adjoints
  for (int i = 0; i < x_idx.size(); ++i) {
    EXPECT_FLOAT_EQ(x.val()(row_idx - 1, x_idx[i]), y.val()(y_idx[i]))  <<
     "Failed for \ni: " << i << "\nx_idx[i][0]: " << x_idx[i] <<
     "\ny_idx[i]: " << y_idx[i];
  }
/*
  std::cout << "\n before x.val(): \n" << x.val() << "\n";
  std::cout << "\n before y.val(): \n" << y.val() << "\n";
*/
  stan::math::sum(x).grad();
/*
  std::cout << "\n after x.val(): \n" << x.val() << "\n";
  std::cout << "\n after y.val(): \n" << y.val() << "\n";
  std::cout << "\n after x.adj(): \n" << x.adj() << "\n";
  std::cout << "\n after y.adj(): \n" << y.adj() << "\n";
  */
  for (int j = 0; j < x.cols(); ++j) {
    EXPECT_FLOAT_EQ(x.val()(row_idx - 1, j), x_val.coeffRef(row_idx - 1, j));
    if (stan::model::internal::check_duplicate(x_idx, j)) {
      EXPECT_FLOAT_EQ(x.adj()(row_idx - 1, j), 0) <<
       "Failed for \ni: " << j << " col_idx[i]: " << col_idx[j] << "\n";
    } else {
      EXPECT_FLOAT_EQ(x.adj()(row_idx - 1, j), 1) <<
       "Failed for \ni: " << j << " col_idx[i]: " << col_idx[j] << "\n";
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
  /*
  puts("Pass1");
  row_idx.push_back(19);
  row_idx.push_back(22);
  test_out_of_range(rx, index_list(index_multi(row_idx), index_multi(col_idx)));
  row_idx.pop_back();
  row_idx.pop_back();
  col_idx.push_back(19);
  col_idx.push_back(22);
  test_out_of_range(rx, index_list(index_multi(row_idx), index_multi(col_idx)));
  */
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
  /*
  for (int j = 0; j < col_idx.size(); ++j) {
    for (int i = 0; i < row_idx.size(); ++i) {
      std::cout << "iter: " << "(" << i << ", " << j << ")" << "cell: (" << row_idx[i] << ", " << col_idx[j] << ") \n";
    }
  }
  */
  stan::arena_t<std::vector<std::array<int, 2>>> x_idx;
  stan::arena_t<std::vector<std::array<int, 2>>> y_idx;
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int j = col_idx.size() - 1; j >= 0; --j) {
    for (int i = row_idx.size() - 1; i >= 0; --i) {
//      std::cout << "iter: " << "(" << i << ", " << j << ")\n\n";
      if (!stan::model::internal::check_duplicate(x_idx, row_idx[i] - 1, col_idx[j] - 1)) {
        y_idx.push_back(std::array<int, 2>{i, j});
        x_idx.push_back(std::array<int, 2>{row_idx[i] - 1, col_idx[j] - 1});
      }
    }
  }
/*
  std::cout << "\n before x.val(): \n" << x.val() << "\n";
  std::cout << "\n before y.val(): \n" << y.val() << "\n";
*/
  assign(x, index_list(index_multi(row_idx), index_multi(col_idx)), y);
/*
  std::cout << "\n after x.val(): \n" << x.val() << "\n";
  std::cout << "\n after y.val(): \n" << y.val() << "\n";
*/
  // We use these to check the adjoints
  for (int i = 0; i < x_idx.size(); ++i) {
    EXPECT_FLOAT_EQ(x.val()(x_idx[i][0], x_idx[i][1]), y.val()(y_idx[i][0], y_idx[i][1]))  <<
     "Failed for \ni: " << i << "\nx_idx[i][0]: " << x_idx[i][0] << " x_idx[i][1]: " << x_idx[i][1] <<
     "\ny_idx[i][0]: " << y_idx[i][0] << " y_idx[i][1]: " << y_idx[i][1];
  }
/*
  std::cout << "\n before x.val(): \n" << x.val() << "\n";
  std::cout << "\n before y.val(): \n" << y.val() << "\n";
*/
  stan::math::sum(x).grad();
/*
  std::cout << "\n after x.val(): \n" << x.val() << "\n";
  std::cout << "\n after y.val(): \n" << y.val() << "\n";
  std::cout << "\n after x.adj(): \n" << x.adj() << "\n";
  std::cout << "\n after y.adj(): \n" << y.adj() << "\n";
*/
  for (int j = 0; j < x.cols(); ++j) {
    for (int i = 0; i < x.rows(); ++i) {
      EXPECT_FLOAT_EQ(x.val()(i, j), x_val.coeffRef(i, j));
      if (stan::model::internal::check_duplicate(x_idx, i, j)) {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 0) <<
         "Failed for \ni: " << i << " row_idx[i]: " << row_idx[i] << "\n" <<
         "j: " << j << " col_idx[i]: " << col_idx[j] << "\n";
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 1) <<
         "Failed for \ni: " << i << " row_idx[i]: " << row_idx[i] << "\n" <<
         "j: " << j << " col_idx[i]: " << col_idx[j] << "\n";
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
  /*
  puts("Pass1");
  row_idx.push_back(19);
  row_idx.push_back(22);
  test_out_of_range(rx, index_list(index_multi(row_idx), index_multi(col_idx)));
  row_idx.pop_back();
  row_idx.pop_back();
  col_idx.push_back(19);
  col_idx.push_back(22);
  test_out_of_range(rx, index_list(index_multi(row_idx), index_multi(col_idx)));
  */
}


TEST_F(VarAssign, uni_minmax_matrix) {
  using stan::math::var_value;
  using stan::math::sum;
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

  test_throw_out_of_range(x, index_list(index_uni(0), index_min_max(2, 4)), y);
  test_throw_out_of_range(x, index_list(index_uni(5), index_min_max(2, 4)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_min_max(0, 2)), y);
  test_throw_invalid_arg(x, index_list(index_uni(2), index_min_max(2, 5)), y);
  std::cout << "\n before x.val(): \n" << x.val() << "\n";
  std::cout << "\n before y.val(): \n" << y.val() << "\n";

  sum(x).grad();
  std::cout << "\n after x.val(): \n" << x.val() << "\n";
  std::cout << "\n after y.val(): \n" << y.val() << "\n";
  std::cout << "\n after x.adj(): \n" << x.adj() << "\n";
  std::cout << "\n after y.adj(): \n" << y.adj() << "\n";

  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      EXPECT_FLOAT_EQ(x.val()(i, j), x_val(i, j));
      if (i == 1) {
        if (j > 0 && j < 4) {
          EXPECT_FLOAT_EQ(x.adj()(i, j), 0) << "Failed for (i, j): (" << i << ", " << j << ")";
        }
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 1) << "Failed for (i, j): (" << i << ", " << j << ")";
      }
    }
  }
  for (Eigen::Index i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(y.adj()(i), 1) << "Failed for (i): (" << i << ")";
  }

}

TEST_F(VarAssign, uni_multi_matrix) {
  using stan::math::var_value;
  using stan::math::sum;
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
      EXPECT_FLOAT_EQ(x.val()(i, j), x_val(i, j)) << "Failed for (i, j): (" << i << ", " << j << ")";
      if (i == 2) {
        if (j == 0 || j == 2 || j == 3) {
          EXPECT_FLOAT_EQ(x.adj()(i, j), 0) << "Failed for (i, j): (" << i << ", " << j << ")";
        }
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), 1) << "Failed for (i, j): (" << i << ", " << j << ")";
      }
    }
  }
  for (Eigen::Index i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(y.adj()(i), 1) << "Failed for (i): (" << i << ")";
  }

}
/*
TEST_F(VarAssign, uni_matrix) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(4);
  y << 10.0, 10.1, 10.2, 10.3;

  assign(x, index_list(index_uni(3)), y);
  for (int j = 0; j < 4; ++j)
    EXPECT_FLOAT_EQ(x(2, j), y(j));

  test_throw_out_of_range(x, index_list(index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_uni(5)), y);
}

TEST_F(VarAssign, min_matrix) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  MatrixXd y(2, 4);
  y << 10.0, 10.1, 10.2, 10.3, 11.0, 11.1, 11.2, 11.3;

  assign(x, index_list(index_min(2)), y);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(y(i, j), x(i + 1, j));
  test_throw_invalid_arg(x, index_list(index_min(1)), y);

  MatrixXd z(1, 2);
  z << 10, 20;
  test_throw_invalid_arg(x, index_list(index_min(1)), z);
  test_throw_invalid_arg(x, index_list(index_min(2)), z);
}

TEST_F(VarAssign, uni_uni_matrix) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  double y = 10.12;
  assign(x, index_list(index_uni(2), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y, x(1, 2));

  test_throw_out_of_range(x, index_list(index_uni(0), index_uni(3)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_uni(4), index_uni(3)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_uni(5)), y);
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



TEST_F(VarAssign, positive_minmax_vec) {
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
  using std::vector;
  Eigen::VectorXd lhs(5);
  lhs << 1, 2, 3, 4, 5;
  Eigen::VectorXd rhs(4);
  rhs << 4, 3, 2, 1;
  assign(lhs, cons_list(index_min_max(1, 4), nil_index_list()), rhs);
  EXPECT_FLOAT_EQ(lhs(0), 4);
  EXPECT_FLOAT_EQ(lhs(1), 3);
  EXPECT_FLOAT_EQ(lhs(2), 2);
  EXPECT_FLOAT_EQ(lhs(3), 1);
  EXPECT_FLOAT_EQ(lhs(4), 5);
}

TEST_F(VarAssign, negative_minmax_vec) {
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
  using std::vector;
  Eigen::VectorXd lhs(5);
  lhs << 1, 2, 3, 4, 5;
  Eigen::VectorXd rhs(4);
  rhs << 1, 2, 3, 4;
  assign(lhs, cons_list(index_min_max(4, 1), nil_index_list()), rhs);
  EXPECT_FLOAT_EQ(lhs(0), 4);
  EXPECT_FLOAT_EQ(lhs(1), 3);
  EXPECT_FLOAT_EQ(lhs(2), 2);
  EXPECT_FLOAT_EQ(lhs(3), 1);
  EXPECT_FLOAT_EQ(lhs(4), 5);
}

TEST_F(VarAssign, positive_minmax_positive_minmax_matrix) {
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev(5, 5);
  for (int i = 0; i < x.size(); ++i) {
    x(i) = i;
    x_rev(i) = x.size() - i - 1;
  }

  for (int i = 0; i < x.rows(); ++i) {
    Eigen::MatrixXd x_colwise_rev = x_rev.block(0, 0, i + 1, i + 1);
    assign(x, index_list(index_min_max(1, i + 1), index_min_max(1, i + 1)),
           x_rev.block(0, 0, i + 1, i + 1));
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

TEST_F(VarAssign, positive_minmax_negative_minmax_matrix) {
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
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
    assign(x, index_list(index_min_max(1, i + 1), index_min_max(i + 1, 1)),
           x_rev.block(0, 0, i + 1, i + 1));
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

TEST_F(VarAssign, negative_minmax_positive_minmax_matrix) {
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
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
    assign(x, index_list(index_min_max(i + 1, 1), index_min_max(1, i + 1)),
           x_rev.block(0, 0, i + 1, i + 1));
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

TEST_F(VarAssign, negative_minmax_negative_minmax_matrix) {
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_min_max;
  using stan::model::nil_index_list;
  using std::vector;
  Eigen::Matrix<double, -1, -1> x(5, 5);
  Eigen::Matrix<double, -1, -1> x_rev(5, 5);
  for (int i = 0; i < x.size(); ++i) {
    x(i) = i;
    x_rev(i) = x.size() - i - 1;
  }

  for (int i = 0; i < x.rows(); ++i) {
    Eigen::MatrixXd x_reverse = x_rev.block(0, 0, i + 1, i + 1).reverse();
    assign(x, index_list(index_min_max(i + 1, 1), index_min_max(i + 1, 1)),
           x_rev.block(0, 0, i + 1, i + 1));
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

TEST(model_indexing, min_vec) {
  VectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  VectorXd rhs_y(3);
  rhs_y << 10, 11, 12;
  assign(lhs_x, index_list(index_min(3)), rhs_y);
  EXPECT_FLOAT_EQ(rhs_y(0), lhs_x(2));
  EXPECT_FLOAT_EQ(rhs_y(1), lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(2), lhs_x(4));
  test_throw_out_of_range(lhs_x, index_list(index_min(0)), rhs_y);

  assign(lhs_x, index_list(index_min(3)), rhs_y.array() + 1.0);
  EXPECT_FLOAT_EQ(rhs_y(0) + 1.0, lhs_x(2));
  EXPECT_FLOAT_EQ(rhs_y(1) + 1.0, lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(2) + 1.0, lhs_x(4));
}

TEST(model_indexing, min_rowvec) {
  RowVectorXd lhs_x(5);
  lhs_x << 0, 1, 2, 3, 4;
  RowVectorXd rhs_y(3);
  rhs_y << 10, 11, 12;
  assign(lhs_x, index_list(index_min(3)), rhs_y);
  EXPECT_FLOAT_EQ(rhs_y(0), lhs_x(2));
  EXPECT_FLOAT_EQ(rhs_y(1), lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(2), lhs_x(4));
  test_throw_out_of_range(lhs_x, index_list(index_min(0)), rhs_y);

  assign(lhs_x, index_list(index_min(3)), rhs_y.array() + 1.0);
  EXPECT_FLOAT_EQ(rhs_y(0) + 1.0, lhs_x(2));
  EXPECT_FLOAT_EQ(rhs_y(1) + 1.0, lhs_x(3));
  EXPECT_FLOAT_EQ(rhs_y(2) + 1.0, lhs_x(4));
}

TEST(model_indexing, uni_mat) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(4);
  y << 10.0, 10.1, 10.2, 10.3;

  assign(x, index_list(index_uni(3)), y.array() + 3);
  for (int j = 0; j < 4; ++j)
    EXPECT_FLOAT_EQ(x(2, j), y(j) + 3);

  test_throw_out_of_range(x, index_list(index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_uni(5)), y);
}

TEST(model_indexing, min_mat) {
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
  test_throw_invalid_arg(x, index_list(index_min(1)), y);

  MatrixXd z(1, 2);
  z << 10, 20;
  test_throw_invalid_arg(x, index_list(index_min(1)), z);
  test_throw_invalid_arg(x, index_list(index_min(2)), z);
}

TEST(model_indexing, uni_uni_mat) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  double y = 10.12;
  assign(x, index_list(index_uni(2), index_uni(3)), y);
  EXPECT_FLOAT_EQ(y, x(1, 2));

  test_throw_out_of_range(x, index_list(index_uni(0), index_uni(3)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_uni(0)), y);
  test_throw_out_of_range(x, index_list(index_uni(4), index_uni(3)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_uni(5)), y);
}

TEST(model_indexing, uni_minmax_mat_rowvec) {
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

  test_throw_out_of_range(x, index_list(index_uni(0), index_min_max(2, 4)), y);
  test_throw_out_of_range(x, index_list(index_uni(5), index_min_max(2, 4)), y);
  test_throw_out_of_range(x, index_list(index_uni(2), index_min_max(0, 2)), y);
  test_throw_invalid_arg(x, index_list(index_uni(2), index_min_max(2, 5)), y);

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
  test_throw_out_of_range(x, index_list(index_uni(3), index_multi(ns)), y);

  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, index_list(index_uni(3), index_multi(ns)), y);

  ns.push_back(2);
  test_throw_invalid_arg(x, index_list(index_uni(3), index_multi(ns)), y);
}

TEST(model_indexing, minmax_uni_mat) {
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

  assign(x.block(0, 0, 3, 3), index_list(index_multi(ns), index_uni(3)),
         y.array() + 2);
  EXPECT_FLOAT_EQ(y(0) + 2, x(2, 2));
  EXPECT_FLOAT_EQ(y(1) + 2, x(0, 2));

  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_list(index_multi(ns), index_uni(3)), y);

  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, index_list(index_multi(ns), index_uni(3)), y);

  ns.push_back(2);
  test_throw_invalid_arg(x, index_list(index_multi(ns), index_uni(3)), y);
}

TEST(model_indexing, minmax_min_mat) {
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

  assign(x.block(0, 0, 3, 3), index_list(index_min_max(2, 3), index_min(2)),
         y.block(0, 0, 2, 2));
  EXPECT_FLOAT_EQ(y(0, 0), x(1, 1));
  EXPECT_FLOAT_EQ(y(0, 1), x(1, 2));
  EXPECT_FLOAT_EQ(y(0, 2), x(1, 3));
  EXPECT_FLOAT_EQ(y(1, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y(1, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y(1, 2), x(2, 3));

  test_throw_invalid_arg(x, index_list(index_min_max(2, 3), index_min(0)), y);
  test_throw_invalid_arg(x, index_list(index_min_max(2, 3), index_min(10)), y);
  test_throw_invalid_arg(x, index_list(index_min_max(1, 3), index_min(2)), y);
}

*/
