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
void test_throw(T1& lhs, const I& idxs, const T2& rhs) {
  EXPECT_THROW(stan::model::assign(lhs, idxs, rhs), std::out_of_range);
}

template <typename T1, typename I, typename T2>
void test_throw_ia(T1& lhs, const I& idxs, const T2& rhs) {
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
      auto generate_linear_varmatrix(Eigen::Index n, Eigen::Index m, double start = 0) {
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
      auto generate_linear_varvector(Eigen::Index n, double start = 0) {
        using ret_t = stan::math::var_value<Eigen::Matrix<double, -1, 1>>;
        return ret_t(generate_linear_vector(n, start));
      }
    }
  }
}

TEST_F(VarIndexing, lvalueNil) {
  using stan::math::var_value;
  auto x = stan::model::test::generate_linear_varvector(5);
  auto y = stan::model::test::generate_linear_varvector(5, 1.0);
  assign(x, nil_index_list(), y);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(i + 1, x.val().coeffRef(i));
  }
}

TEST_F(VarIndexing, lvalueUni) {
  using stan::math::var_value;
  auto x = stan::model::test::generate_linear_varvector(5);
  stan::math::var y(18);
  assign(x, index_list(index_uni(2)), y);
  EXPECT_FLOAT_EQ(y.val(), x.val()[1]);
  y.adj() = 100;
  x.adj()[1] = 10;
  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj(), 110);
  EXPECT_FLOAT_EQ(x.adj()[0], 0);
  test_throw(x, index_list(index_uni(0)), y);
  test_throw(x, index_list(index_uni(6)), y);
}

TEST_F(VarIndexing, lvalueMultiVec) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_varvector;
  auto x = generate_linear_varvector(5);
  auto y = generate_linear_varvector(2, 10);
  vector<int> ns;
  ns.push_back(2);
  ns.push_back(4);
  assign(x, index_list(index_multi(ns)), y);
  EXPECT_FLOAT_EQ(y.val()[0], x.val()[1]);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[3]);
  y.adj()[0] = 10;
  y.adj()[1] = 20;
  x.adj()[1] = 30;
  x.adj()[3] = 40;
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
  /*
  std::cout << " post-grad \n";
  std::cout << "x val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";
  */
}
