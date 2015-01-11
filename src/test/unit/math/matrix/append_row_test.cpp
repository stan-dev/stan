#include <stan/math/matrix/append_row.hpp>
#include <test/unit/math/matrix/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

using stan::math::append_row;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using std::vector;

TEST(MathMatrix, append_row) {
  MatrixXd m33(3, 3);
  m33 << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
        
  MatrixXd m32(3, 2);
  m32 << 11, 12,
         13, 14,
         15, 16;

  MatrixXd m23(2, 3);
  m23 << 21, 22, 23,
         24, 25, 26;
         
  VectorXd v3(3);
  v3 << 31, 
        32,
        33;
        
  VectorXd v3b(3);
  v3b << 34, 
         35,
         36;

  RowVectorXd rv3(3);
  rv3 << 41, 42, 43;
  
  RowVectorXd rv3b(3);
  rv3b << 44, 45, 46;

  MatrixXd mat;
  VectorXd cvec;
  
  //matrix append_row(matrix, matrix)
  mat = append_row(m33, m23);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(j, i), m33(j, i));
    for (int j = 3; j < 5; j++)
      EXPECT_EQ(mat(j, i), m23(j-3, i));
  }    
  mat = append_row(m23, m33);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(mat(j, i), m23(j, i));
    for (int j = 2; j < 5; j++)
      EXPECT_EQ(mat(j, i), m33(j-2, i));
  }

  
  MatrixXd m32b(2, 3);
  m32b = m32*1.101; //ensure some different values
  mat = append_row(m32, m32b);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(j, i), m32(j, i));
    for (int j = 3; j < 6; j++)
      EXPECT_EQ(mat(j, i), m32b(j-3, i));
  }
  mat = append_row(m32b, m32);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(j, i), m32b(j, i));
    for (int j = 3; j < 6; j++)
      EXPECT_EQ(mat(j, i), m32(j-3, i));
  }

  //matrix append_row(matrix, row_vector)
  //matrix append_row(row_vector, matrix)
  mat = append_row(m33, rv3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(j, i), m33(j, i));
    EXPECT_EQ(mat(3, i), rv3(i));
  }
  mat = append_row(rv3, m33);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(0, i), rv3(i));
    for (int j = 1; j < 4; j++)
      EXPECT_EQ(mat(j, i), m33(j-1, i));
  }
  mat = append_row(m23, rv3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(mat(j, i), m23(j, i));
    EXPECT_EQ(mat(2, i), rv3(i));
  }
  mat = append_row(rv3, m23);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(0, i), rv3(i));
    for (int j = 1; j < 3; j++)
      EXPECT_EQ(mat(j, i), m23(j-1, i));
  }
  
  //matrix append_row(row_vector, row_vector)  
  mat = append_row(rv3, rv3b);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(0, i), rv3(i));
    EXPECT_EQ(mat(1, i), rv3b(i));
  }
  mat = append_row(rv3b, rv3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(0, i), rv3b(i));
    EXPECT_EQ(mat(1, i), rv3(i));
  }
   
  //matrix append_row(vector, vector)
  cvec = append_row(v3, v3b);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(cvec(i), v3(i));
  for (int i = 3; i < 6; i++)
    EXPECT_EQ(cvec(i), v3b(i-3));
  cvec = append_row(v3b, v3);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(cvec(i), v3b(i));
  for (int i = 3; i < 6; i++)
    EXPECT_EQ(cvec(i), v3(i-3));
    
  EXPECT_THROW(append_row(m32, m33), std::domain_error);
  EXPECT_THROW(append_row(m32, m23), std::domain_error);
  EXPECT_THROW(append_row(m32, v3), std::domain_error);
  EXPECT_THROW(append_row(m32, rv3), std::domain_error);
  EXPECT_THROW(append_row(m33, m32), std::domain_error);
  EXPECT_THROW(append_row(m23, m32), std::domain_error);
  EXPECT_THROW(append_row(v3, m32), std::domain_error);
  EXPECT_THROW(append_row(rv3, m32), std::domain_error);
  
  EXPECT_THROW(append_row(v3, m33), std::domain_error);
  EXPECT_THROW(append_row(v3, m32), std::domain_error);
  EXPECT_THROW(append_row(v3, rv3), std::domain_error);
  EXPECT_THROW(append_row(m33, v3), std::domain_error);
  EXPECT_THROW(append_row(m32, v3), std::domain_error);
  EXPECT_THROW(append_row(rv3, v3), std::domain_error);
}
