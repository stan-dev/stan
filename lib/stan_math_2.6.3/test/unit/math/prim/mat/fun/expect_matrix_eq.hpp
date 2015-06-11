#ifndef TEST_MATH_MATRIX_EXPECT_MATRIX_EQ_HPP
#define TEST_MATH_MATRIX_EXPECT_MATRIX_EQ_HPP

#include <gtest/gtest.h>

void expect_matrix_eq(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& a,
                      const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& b) {
  EXPECT_EQ(a.rows(), b.rows());
  EXPECT_EQ(a.cols(), b.cols());
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j)
      EXPECT_FLOAT_EQ(a(i,j), b(i,j));
}

#endif
