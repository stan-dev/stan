#include <iostream>
#include <stan/math/matrix/append_col.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/agrad/rev/functions/exp.hpp>

using stan::math::sum;
using stan::math::append_col;
using stan::agrad::matrix_v;
using stan::agrad::row_vector_v;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;

TEST(AgradRevMatrix, append_col_matrix) {
  matrix_v a(2,2);
  matrix_v a_exp(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  AVEC x;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      x.push_back(a(i,j));
      a_exp(i, j) = stan::agrad::exp(a(i, j));
    }
  }
  
  AVAR append_col_ab = sum(append_col(a_exp, b));

  VEC g = cgradvec(append_col_ab, x);
  
  size_t idx = 0;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(std::exp(a(i, j).val()), g[idx++]);
  stan::agrad::recover_memory();
}

TEST(AgradRevMatrix, append_col_row_vector) {
  row_vector_v a(3);
  row_vector_v a_exp(3);
  RowVectorXd b(3);
  
  a << 2.0, 3.0, 9.0;
       
  b << 4.0, 3.0, 0.0;

  AVEC x;
  for (int i = 0; i < 3; ++i) {
    x.push_back(a(i));
    a_exp(i) = stan::agrad::exp(a(i));
  }
  
  AVAR append_col_ab = sum(append_col(a_exp, b));

  VEC g = cgradvec(append_col_ab, x);
  
  size_t idx = 0;
  for (int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(std::exp(a(i).val()), g[idx++]);
  stan::agrad::recover_memory();
}
