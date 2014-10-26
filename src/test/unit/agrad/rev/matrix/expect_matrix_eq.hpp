#ifndef TEST_AGRAD_REV_MATRIX_EXPECT_MATRIX_EQ_HPP
#define TEST_AGRAD_REV_MATRIX_EXPECT_MATRIX_EQ_HPP

#include <stan/math/matrix.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/agrad/rev/functions/value_of.hpp>
#include <gtest/gtest.h>


template <typename T1, typename T2>
void expect_matrix_eq(const Eigen::Matrix<T1,Eigen::Dynamic,Eigen::Dynamic>& a,
                      const Eigen::Matrix<T2,Eigen::Dynamic,Eigen::Dynamic>& b) {
  EXPECT_EQ(a.rows(), b.rows());
  EXPECT_EQ(a.cols(), b.cols());
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j)
      EXPECT_FLOAT_EQ(value_of(a(i,j)), value_of(b(i,j)));
}

#endif
