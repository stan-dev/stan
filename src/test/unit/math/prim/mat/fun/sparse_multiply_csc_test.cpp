#include <stan/math/prim/mat/fun/assign.hpp>
#include <stan/math/prim/mat/fun/sparse_extractors_csc.hpp>
#include <stan/math/prim/mat/fun/sparse_multiply_csc.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>

// Test that dense multiplication results is correct (CSC).
TEST(SparseStuff, sparse_multiply_csc) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();

  stan::math::vector_d X_w = stan::math::extract_w(a);
  std::vector<int> X_v = stan::math::extract_v(a);
  std::vector<int> X_u = stan::math::extract_u(a);
  std::vector<int> X_z = stan::math::extract_z(a);

  stan::math::vector_d b(3);
  b << 22, 33, 44;

  stan::math::vector_d result = stan::math::sparse_multiply_csc(
      2, 3, X_w, X_v, X_u, X_z, b);

  EXPECT_FLOAT_EQ( 440.0, result(0));
  EXPECT_FLOAT_EQ(1034.0, result(1));
}

// Test that m=0 throws (CSC).
TEST(SparseStuff, sparse_multiply_csc_m0) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();

  stan::math::vector_d X_w = stan::math::extract_w(a);
  std::vector<int> X_v = stan::math::extract_v(a);
  std::vector<int> X_u = stan::math::extract_u(a);
  std::vector<int> X_z = stan::math::extract_z(a);

  stan::math::vector_d b(3);
  b << 22, 33, 44;

  EXPECT_THROW({
  stan::math::vector_d result = stan::math::sparse_multiply_csc(0, 3, X_w, X_v, X_u, X_z, b);},
  std::domain_error);
}

// Test that n=0 throws (CSC).
TEST(SparseStuff, sparse_multiply_csc_n0) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();

  stan::math::vector_d X_w = stan::math::extract_w(a);
  std::vector<int> X_v = stan::math::extract_v(a);
  std::vector<int> X_u = stan::math::extract_u(a);
  std::vector<int> X_z = stan::math::extract_z(a);

  stan::math::vector_d b(3);
  b << 22, 33, 44;

  EXPECT_THROW({
  stan::math::vector_d result = stan::math::sparse_multiply_csc(2, 0, X_w, X_v, X_u, X_z, b);},
  std::domain_error);
}

// Test that short b throws (CSC).
TEST(SparseStuff, sparse_multiply_csc_b_short) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();

  stan::math::vector_d X_w = stan::math::extract_w(a);
  std::vector<int> X_v = stan::math::extract_v(a);
  std::vector<int> X_u = stan::math::extract_u(a);
  std::vector<int> X_z = stan::math::extract_z(a);

  stan::math::vector_d b(2);  // short b
  b << 22, 33;

  EXPECT_THROW({
  stan::math::vector_d result = stan::math::sparse_multiply_csc(2, 3, X_w, X_v, X_u, X_z, b);},
  std::invalid_argument);
}

// Test that short u throws (CSC).
TEST(SparseStuff, sparse_multiply_csc_u_short) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();

  stan::math::vector_d X_w = stan::math::extract_w(a);
  std::vector<int> X_v = stan::math::extract_v(a);
  std::vector<int> X_u = stan::math::extract_u(a);
  std::vector<int> X_z = stan::math::extract_z(a);

  X_u.erase(X_u.begin());   // make a short u:

  stan::math::vector_d b(3);
  b << 22, 33, 44;

  EXPECT_THROW({
  stan::math::vector_d result = stan::math::sparse_multiply_csc(2, 3, X_w, X_v, X_u, X_z, b);},
  std::invalid_argument);
}

// Test that short z throws (CSC).
TEST(SparseStuff, sparse_multiply_csc_z_short) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();

  stan::math::vector_d X_w = stan::math::extract_w(a);
  std::vector<int> X_v = stan::math::extract_v(a);
  std::vector<int> X_u = stan::math::extract_u(a);
  std::vector<int> X_z = stan::math::extract_z(a);

  X_z.erase(X_z.begin());   // make a short z:

  stan::math::vector_d b(3);
  b << 22, 33, 44;

  EXPECT_THROW({
  stan::math::vector_d result = stan::math::sparse_multiply_csc(2, 3, X_w, X_v, X_u, X_z, b);},
  std::invalid_argument);
}

// Test that short v throws (CSC).
TEST(SparseStuff, sparse_multiply_csc_v_short) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();

  stan::math::vector_d X_w = stan::math::extract_w(a);
  std::vector<int> X_v = stan::math::extract_v(a);
  std::vector<int> X_u = stan::math::extract_u(a);
  std::vector<int> X_z = stan::math::extract_z(a);

  X_v.erase(X_v.begin()+4);   // make a short v:

  stan::math::vector_d b(3);
  b << 22, 33, 44;

  EXPECT_THROW({
  stan::math::vector_d result = stan::math::sparse_multiply_csc(2, 3, X_w, X_v, X_u, X_z, b);},
  std::invalid_argument);
}

// Test that wrong z throws (CSC).
TEST(SparseStuff, sparse_multiply_csc_z_wrong) {
  stan::math::matrix_d m(2, 3);
  Eigen::SparseMatrix<double> a;
  m << 2.0, 4.0, 6.0, 8.0, 10.0, 12.0;
  a = m.sparseView();

  stan::math::vector_d X_w = stan::math::extract_w(a);
  std::vector<int> X_v = stan::math::extract_v(a);
  std::vector<int> X_u = stan::math::extract_u(a);
  std::vector<int> X_z = stan::math::extract_z(a);

  X_z[X_z.size()-1] += 1;   // make a wrong z:

  stan::math::vector_d b(3);
  b << 22, 33, 44;

  EXPECT_THROW({
  stan::math::vector_d result = stan::math::sparse_multiply_csc(2, 3, X_w, X_v, X_u, X_z, b);},
  std::invalid_argument);
}

