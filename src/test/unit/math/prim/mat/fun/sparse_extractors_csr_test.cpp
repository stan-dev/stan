#include <stan/math/prim/mat/fun/assign.hpp>
#include <stan/math/prim/mat/fun/sparse_extractors_csr.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <vector>

TEST(SparseStuff, extract_csr_w) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double, Eigen::RowMajor> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();
  stan::math::vector_d result = stan::math::extract_w(a);
  EXPECT_FLOAT_EQ( 2.0, result(0));
  EXPECT_FLOAT_EQ( 4.0, result(1));
  EXPECT_FLOAT_EQ( 6.0, result(2));
  EXPECT_FLOAT_EQ( 8.0, result(3));
  EXPECT_FLOAT_EQ(10.0, result(4));
  EXPECT_FLOAT_EQ(12.0, result(5));
}

TEST(SparseStuff, extract_csr_v) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double, Eigen::RowMajor> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();
  std::vector<int> result = stan::math::extract_v(a);
  EXPECT_EQ(1, result[0]);
  EXPECT_EQ(2, result[1]);
  EXPECT_EQ(3, result[2]);
  EXPECT_EQ(1, result[3]);
  EXPECT_EQ(2, result[4]);
  EXPECT_EQ(3, result[5]);
}

TEST(SparseStuff, extract_csr_u) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double, Eigen::RowMajor> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();
  std::vector<int> result = stan::math::extract_u(a);
  EXPECT_EQ(1, result[0]);
  EXPECT_EQ(4, result[1]);
  EXPECT_EQ(7, result[2]);
}

TEST(SparseStuff, extract_csr_z) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double, Eigen::RowMajor> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();
  std::vector<int> result = stan::math::extract_z(a);
  EXPECT_EQ(3, result[0]);
  EXPECT_EQ(3, result[1]);
}

