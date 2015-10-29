#include <Eigen/Dense>
#include <stan/model/indexing/deep_copy.hpp>
#include <vector>
#include <gtest/gtest.h>

TEST(modelIndexingDeepCopy, scalar) {
  using stan::model::deep_copy;
  EXPECT_FLOAT_EQ(2.3, deep_copy(2.3));
  EXPECT_EQ(3, deep_copy(3));
}

template <typename C>
void test_vec() {
  using stan::model::deep_copy;

  // first test size 0
  C a;
  C ac = deep_copy(a);
  EXPECT_EQ(0, ac.size());

  // something bigger than size 0
  C v(3);
  v[0] = 1; 
  v[1] = 2;
  v[2] = 3.5;

  C vc = deep_copy(v);

  // test that copies properly (no loop to avoid signed/unsigned)
  EXPECT_EQ(v.size(), vc.size());
  EXPECT_FLOAT_EQ(v[0], vc[0]);
  EXPECT_FLOAT_EQ(v[1], vc[1]);
  EXPECT_FLOAT_EQ(v[2], vc[2]);

  // modifying copy should not affect original
  vc[1] = 10;  
  EXPECT_FLOAT_EQ(10, vc[1]);
  EXPECT_FLOAT_EQ(2, v[1]);
}

TEST(modelIndexingDeepCopy, vectorDouble) {
  test_vec<Eigen::VectorXd>();
}
TEST(modelIndexingDeepCopy, rowVectorDouble) {
  test_vec<Eigen::RowVectorXd>();
}
TEST(modelIndexingDeepCopy, stdVectorDouble) {
  test_vec<std::vector<double> >();
}

TEST(modelIndexingDeepCopy, matrixDouble) {
  using stan::model::deep_copy;
  using Eigen::MatrixXd;

  // first test size 0
  MatrixXd a(0, 0);
  MatrixXd ac = deep_copy(a);
  EXPECT_EQ(0, ac.size());

  // something bigger than size 0
  MatrixXd b(2, 3);
  b << 1, 2, 3, 4, 5, 6;

  MatrixXd bc = deep_copy(b);
  EXPECT_EQ(b.rows(), bc.rows());
  EXPECT_EQ(b.cols(), bc.cols());
  for (int j = 0; j < b.cols(); ++j)
    for (int i = 0; i < b.rows(); ++i)
      EXPECT_FLOAT_EQ(b(i, j), bc(i, j));


  // modifying copy should not affect original
  bc(0, 1) = 110;
  EXPECT_FLOAT_EQ(110, bc(0, 1));
  EXPECT_FLOAT_EQ(2, b(0, 1));
}


TEST(modelIndexingDeepCopy, stdVectorStdVectorDouble) {
  using stan::model::deep_copy;
  using std::vector;
  typedef vector<double> doubles_t;
  typedef vector<doubles_t> doubless_t;
  
  doubles_t a1;
  doubles_t a2;
  for (size_t i = 0; i < 3; ++i) {
    a1.push_back(i);
    a2.push_back(i + 10);
  }
  doubless_t a;
  a.push_back(a1);
  a.push_back(a2);
  

  doubless_t ac = deep_copy(a);
  EXPECT_EQ(a.size(), ac.size());
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_EQ(a[i].size(), ac[i].size());

  for (size_t i = 0; i < a.size(); ++i)
    for (size_t j = 0; j < a[i].size(); ++j)
      EXPECT_FLOAT_EQ(a[i][j], ac[i][j]);

  ac[1][1] = 20;
  EXPECT_FLOAT_EQ(20, ac[1][1]);
  EXPECT_FLOAT_EQ(11, a[1][1]);
}
