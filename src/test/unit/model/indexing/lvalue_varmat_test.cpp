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

struct VarIndexing : public testing::Test {
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

TEST_F(VarIndexing, lvalueNil) {
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

TEST_F(VarIndexing, lvalueUniVec) {
  test_uni_vec<true>();
}

TEST_F(VarIndexing, lvalueUniRowVec) {
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

TEST_F(VarIndexing, lvalueMultiVec) {
  test_multi_vec<true>();
}

TEST_F(VarIndexing, lvalueMultiRowVec) {
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

TEST_F(VarIndexing, lvalueMinMaxVec) {
  test_minmax_vec<true>();
}

TEST_F(VarIndexing, lvalueMinMaxRowVec) {
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
TEST_F(VarIndexing, lvalueMaxVec) {
  test_max_vec<true>();
}

TEST_F(VarIndexing, lvalueMaxRowVec) {
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

TEST_F(VarIndexing, lvalueOmniVec) {
  test_omni_vec<true>();
}

TEST_F(VarIndexing, lvalueOmniRowVec) {
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

TEST(model_indexing, assign_eigvec_var_uni_index_segment) {
  test_eigvec_var_uni_index_seg<Eigen::VectorXd>();
}
TEST(model_indexing, assign_eigrowvec_var_uni_index_segment) {
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

TEST_F(VarIndexing, lvalueUniUniEigenVec) {
  test_uni_uni_vec_eigvec<Eigen::VectorXd>();
}

TEST_F(VarIndexing, lvalueUniUniEigenRowVec) {
  test_uni_uni_vec_eigvec<Eigen::RowVectorXd>();
}


TEST_F(VarIndexing, lvalueUniRowMat) {
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

TEST_F(VarIndexing, lvalueMatrixRowVecMulti) {
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
  std::cout << "\n before x.val(): \n" << x.val() << "\n";
  std::cout << "\n before y.val(): \n" << y.val() << "\n";
  assign(x, index_list(index_uni(row_idx), index_multi(col_idx)), y);
  std::cout << "\n after x.val(): \n" << x.val() << "\n";
  std::cout << "\n after y.val(): \n" << y.val() << "\n";
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


TEST_F(VarIndexing, lvalueMatrixMultiMulti) {
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


/*


TEST_F(VarIndexing, lvalueUniMulti) {
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

  test_throw_out_of_range(xs, index_list(index_uni(0), index_min_max(3, 5)), ys);
  test_throw_out_of_range(xs, index_list(index_uni(11), index_min_max(3, 5)), ys);
  test_throw_invalid_arg(xs, index_list(index_uni(4), index_min_max(2, 5)), ys);
}

TEST_F(VarIndexing, lvalueMultiUni) {
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

  test_throw_invalid_arg(xs, index_list(index_min_max(3, 6), index_uni(7)), ys);
  test_throw_out_of_range(xs, index_list(index_min_max(4, 6), index_uni(0)), ys);
  test_throw_out_of_range(xs, index_list(index_min_max(4, 6), index_uni(30)), ys);
}

TEST_F(VarIndexing, lvalueMatrixUni) {
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

TEST_F(VarIndexing, lvalueMatrixMulti) {
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

TEST_F(VarIndexing, lvalueMatrixUniUni) {
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

TEST_F(VarIndexing, lvalueMatrixUniMulti) {
  MatrixXd x(3, 4);
  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;

  RowVectorXd y(3);
  y << 10, 11, 12;
  assign(x, index_list(index_uni(2), index_min_max(2, 4)), y);
  EXPECT_FLOAT_EQ(y(0), x(1, 1));
  EXPECT_FLOAT_EQ(y(1), x(1, 2));
  EXPECT_FLOAT_EQ(y(2), x(1, 3));

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

  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_list(index_uni(3), index_multi(ns)), y);

  ns[ns.size() - 1] = 20;
  test_throw_out_of_range(x, index_list(index_uni(3), index_multi(ns)), y);

  ns.push_back(2);
  test_throw_invalid_arg(x, index_list(index_uni(3), index_multi(ns)), y);
}

TEST_F(VarIndexing, lvalueMatrixMultiUni) {
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

TEST_F(VarIndexing, lvalueMatrixMultiMulti) {
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

  test_throw_invalid_arg(x, index_list(index_min_max(2, 3), index_min(0)), y);
  test_throw_invalid_arg(x, index_list(index_min_max(2, 3), index_min(10)), y);
  test_throw_invalid_arg(x, index_list(index_min_max(1, 3), index_min(2)), y);

  x << 0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3;
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

  ms[ms.size() - 1] = 0;
  test_throw_out_of_range(x, index_list(index_multi(ms), index_multi(ns)), y);

  ms[ms.size() - 1] = 10;
  test_throw_out_of_range(x, index_list(index_multi(ms), index_multi(ns)), y);

  ms[ms.size() - 1] = 1;  // back to original valid value
  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_list(index_multi(ms), index_multi(ns)), y);

  ns[ns.size() - 1] = 10;
  test_throw_out_of_range(x, index_list(index_multi(ms), index_multi(ns)), y);
}
TEST_F(VarIndexing, doubleToVar) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  using stan::model::assign;
  using stan::model::cons_list;
  using stan::model::index_omni;
  using stan::model::nil_index_list;
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

  assign(yss, cons_list(index_omni(), nil_index_list()), xss, "foo");

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
  assign(a, index_list(index_multi(is)), b);
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
  assign(c, cons_list(index_uni(2), nil_index_list()), d);
  for (int j = 0; j < 3; ++j)
    EXPECT_FLOAT_EQ(c(1, j).val(), d(j));
}
TEST_F(VarIndexing, resultSizeNegIndexing) {
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

TEST_F(VarIndexing, resultSizeIndexingEigen) {
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

TEST_F(VarIndexing, resultSizeNegIndexingEigen) {
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

TEST_F(VarIndexing, resultSizePosMinMaxPosMinMaxEigenMatrix) {
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

TEST_F(VarIndexing, resultSizePosMinMaxNegMinMaxEigenMatrix) {
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

TEST_F(VarIndexing, resultSizeNigMinMaxPosMinMaxEigenMatrix) {
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

TEST_F(VarIndexing, resultSizeNegMinMaxNegMinMaxEigenMatrix) {
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

TEST(modelIndexing, doubleToVarSimple) {
  using stan::math::var;
  using stan::model::nil_index_list;
  typedef Eigen::MatrixXd mat_d;
  typedef Eigen::Matrix<var, -1, -1> mat_v;

  mat_d a(2, 2);
  a << 1, 2, 3, 4;
  mat_v b;
  assign(b, nil_index_list(), a);
  for (int i = 0; i < a.size(); ++i)
    EXPECT_FLOAT_EQ(a(i), b(i).val());
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
  test_throw_out_of_range(lhs_x, index_list(index_min(0)), rhs_y);

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
  test_throw_out_of_range(lhs_x, index_list(index_multi(ns)), rhs_y);

  ns[ns.size() - 1] = 10;
  test_throw_out_of_range(lhs_x, index_list(index_multi(ns)), rhs_y);

  ns[ns.size() - 1] = 3;
  ns.push_back(1);
  test_throw_invalid_arg(lhs_x, index_list(index_multi(ns)), rhs_y);
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
  test_throw_out_of_range(lhs_x, index_list(index_min(0)), rhs_y);

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
  test_throw_out_of_range(lhs_x, index_list(index_multi(ns)), rhs_y);

  ns[ns.size() - 1] = 10;
  test_throw_out_of_range(lhs_x, index_list(index_multi(ns)), rhs_y);

  ns[ns.size() - 1] = 3;
  ns.push_back(1);
  test_throw_invalid_arg(lhs_x, index_list(index_multi(ns)), rhs_y);
}

TEST(model_indexing, assign_densemat_rowvec_uni_index) {
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

TEST(model_indexing, assign_densemat_densemat_index_min) {
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

TEST(model_indexing, assign_densemat_scalar_index_uni) {
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

TEST(model_indexing, assign_densemat_eigrowvec_uni_index_min_max_index) {
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

TEST(model_indexing, assign_densemat_eigvec_min_max_index_uni_index) {
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

TEST(model_indexing, assign_densemat_densemat_min_max_index_min_index) {
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
  assign(x, index_list(index_multi(ms), index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y(0, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y(0, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y(0, 2), x(2, 0));
  EXPECT_FLOAT_EQ(y(1, 0), x(0, 1));
  EXPECT_FLOAT_EQ(y(1, 1), x(0, 2));
  EXPECT_FLOAT_EQ(y(1, 2), x(0, 0));

  MatrixXd y2 = y.array() + 2;
  assign(x.block(0, 0, 3, 4), index_list(index_multi(ms), index_multi(ns)),
         y.array() + 2);
  EXPECT_FLOAT_EQ(y2(0, 0), x(2, 1));
  EXPECT_FLOAT_EQ(y2(0, 1), x(2, 2));
  EXPECT_FLOAT_EQ(y2(0, 2), x(2, 0));
  EXPECT_FLOAT_EQ(y2(1, 0), x(0, 1));
  EXPECT_FLOAT_EQ(y2(1, 1), x(0, 2));
  EXPECT_FLOAT_EQ(y2(1, 2), x(0, 0));

  ms[ms.size() - 1] = 0;
  test_throw_out_of_range(x, index_list(index_multi(ms), index_multi(ns)), y);

  ms[ms.size() - 1] = 10;
  test_throw_out_of_range(x, index_list(index_multi(ms), index_multi(ns)), y);

  ms[ms.size() - 1] = 1;  // back to original valid value
  ns[ns.size() - 1] = 0;
  test_throw_out_of_range(x, index_list(index_multi(ms), index_multi(ns)), y);

  ns[ns.size() - 1] = 10;
  test_throw_out_of_range(x, index_list(index_multi(ms), index_multi(ns)), y);
}
*/
