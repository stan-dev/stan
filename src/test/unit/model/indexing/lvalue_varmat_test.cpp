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
      auto generate_linear_var_vector(Eigen::Index n, double start = 0) {
        using ret_t = stan::math::var_value<Eigen::Matrix<double, -1, 1>>;
        return ret_t(generate_linear_vector(n, start));
      }

      auto generate_linear_rowvector(Eigen::Index n, double start = 0) {
        Eigen::Matrix<double, 1, -1> A(n);
        for (Eigen::Index i = 0; i < A.size(); ++i) {
          A(i) = i + start;
        }
        return A;
      }
      auto generate_linear_var_rowvector(Eigen::Index n, double start = 0) {
        using ret_t = stan::math::var_value<Eigen::Matrix<double, 1, -1>>;
        return ret_t(generate_linear_rowvector(n, start));
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

TEST_F(VarIndexing, lvalueUni) {
  using stan::math::var_value;
  auto x = stan::model::test::generate_linear_var_vector(5);
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
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector(5);
  auto y = generate_linear_var_vector(2, 10);
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

TEST_F(VarIndexing, lvalueMinMaxVec) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector(5);
  auto y = generate_linear_var_vector(2, 10);

  assign(x, index_list(index_min_max(2, 3)), y);
  EXPECT_FLOAT_EQ(y.val()[0], x.val()[1]);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[2]);
  y.adj()[0] = 10;
  y.adj()[1] = 20;
  x.adj()[1] = 30;
  x.adj()[2] = 40;

  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";

  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj()[0], 40);
  EXPECT_FLOAT_EQ(y.adj()[1], 60);
  EXPECT_FLOAT_EQ(x.adj()[1], 0);
  EXPECT_FLOAT_EQ(x.adj()[3], 0);

}

TEST_F(VarIndexing, lvalueMinVec) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector(5);
  auto y = generate_linear_var_vector(2, 10);

  assign(x, index_list(index_min(4)), y);
  EXPECT_FLOAT_EQ(y.val()[0], x.val()[3]);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[4]);
  y.adj()[0] = 10;
  y.adj()[1] = 20;
  x.adj()[3] = 30;
  x.adj()[4] = 40;

  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";

  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj()[0], 40);
  EXPECT_FLOAT_EQ(y.adj()[1], 60);
  EXPECT_FLOAT_EQ(x.adj()[3], 0);
  EXPECT_FLOAT_EQ(x.adj()[4], 0);

}

TEST_F(VarIndexing, lvalueMaxVec) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector(5);
  auto y = generate_linear_var_vector(2, 10);

  assign(x, index_list(index_max(2)), y);
  EXPECT_FLOAT_EQ(y.val()[0], x.val()[0]);
  EXPECT_FLOAT_EQ(y.val()[1], x.val()[1]);
  y.adj()[0] = 10;
  y.adj()[1] = 20;
  x.adj()[0] = 30;
  x.adj()[1] = 40;

  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";

  stan::math::grad();
  EXPECT_FLOAT_EQ(y.adj()[0], 40);
  EXPECT_FLOAT_EQ(y.adj()[1], 60);
  EXPECT_FLOAT_EQ(x.adj()[0], 0);
  EXPECT_FLOAT_EQ(x.adj()[1], 0);

}

TEST_F(VarIndexing, lvalueOmniVec) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_vector;
  auto x = generate_linear_var_vector(5);
  auto y = generate_linear_var_vector(5, 10);

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

  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";

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

  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";

}

TEST_F(VarIndexing, lvalueUniRowMat) {
  using stan::math::var_value;
  using stan::model::test::generate_linear_var_matrix;
  using stan::model::test::generate_linear_var_rowvector;
  puts("Got1");
  auto x = generate_linear_var_matrix(5, 5);
  puts("Got2");
  auto y = generate_linear_var_rowvector(5, 10);

  puts("Got3");
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

  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";

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

  std::cout << "\n pre-grad \n";
  std::cout << "\nx val: \n" << x.val() << "\n";
  std::cout << "x adj: \n" << x.adj() << "\n";
  std::cout << "y val: \n" << y.val() << "\n";
  std::cout << "y adj: \n" << y.adj() << "\n";

}
