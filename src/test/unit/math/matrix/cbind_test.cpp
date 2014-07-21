#include <stan/math/matrix/cbind.hpp>
#include <test/unit/math/matrix/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

using stan::math::cbind;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using std::vector;

TEST(MathMatrix, cbind) {
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
  RowVectorXd rvec;
  
  //matrix cbind(matrix, matrix)
  mat = cbind(m33, m32);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(i, j), m33(i, j));
    for (int j = 3; j < 5; j++)
      EXPECT_EQ(mat(i, j), m32(i, j-3));
  }
  mat = cbind(m32, m33);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(mat(i, j), m32(i, j));
    for (int j = 2; j < 5; j++)
      EXPECT_EQ(mat(i, j), m33(i, j-2));
  }
    
  //matrix cbind(matrix, vector)
  //matrix cbind(vector, matrix)
  mat = cbind(m33, v3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(i, j), m33(i, j));
    EXPECT_EQ(mat(i, 3), v3(i));
  }
  mat = cbind(v3, m33);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(i, 0), v3(i));
    for (int j = 1; j < 4; j++)
      EXPECT_EQ(mat(i, j), m33(i, j-1));
  }
  mat = cbind(m32, v3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(mat(i, j), m32(i, j));
    EXPECT_EQ(mat(i, 2), v3(i));
  }
  mat = cbind(v3, m32);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(i, 0), v3(i));
    for (int j = 1; j < 3; j++)
      EXPECT_EQ(mat(i, j), m32(i, j-1));
  }
  
  //matrix cbind(vector, vector)  
  mat = cbind(v3, v3b);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(i, 0), v3(i));
    EXPECT_EQ(mat(i, 1), v3b(i));
  }
  mat = cbind(v3b, v3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(i, 0), v3b(i));
    EXPECT_EQ(mat(i, 1), v3(i));
  }
   
  //matrix cbind(row_vector, row_vector)
  rvec = cbind(rv3, rv3b);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(rvec(i), rv3(i));
  for (int i = 3; i < 6; i++)
    EXPECT_EQ(rvec(i), rv3b(i-3));
  rvec = cbind(rv3b, rv3);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(rvec(i), rv3b(i));
  for (int i = 3; i < 6; i++)
    EXPECT_EQ(rvec(i), rv3(i-3));
    
  EXPECT_THROW(cbind(m23, m33), std::domain_error);
  EXPECT_THROW(cbind(m23, m32), std::domain_error);
  EXPECT_THROW(cbind(m23, v3), std::domain_error);
  EXPECT_THROW(cbind(m23, rv3), std::domain_error);
  EXPECT_THROW(cbind(m33, m23), std::domain_error);
  EXPECT_THROW(cbind(m32, m23), std::domain_error);
  EXPECT_THROW(cbind(v3, m23), std::domain_error);
  EXPECT_THROW(cbind(rv3, m23), std::domain_error);
  
  EXPECT_THROW(cbind(rv3, m33), std::domain_error);
  EXPECT_THROW(cbind(rv3, m32), std::domain_error);
  EXPECT_THROW(cbind(rv3, v3), std::domain_error);
  EXPECT_THROW(cbind(m33, rv3), std::domain_error);
  EXPECT_THROW(cbind(m32, rv3), std::domain_error);
  EXPECT_THROW(cbind(v3, rv3), std::domain_error);
}
