#include <stan/math/matrix/rbind.hpp>
#include <test/unit/math/matrix/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

using stan::math::rbind;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using std::vector;

TEST(MathMatrix, rbind) {
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
  
  //matrix rbind(matrix, matrix)
  mat = rbind(m33, m23);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(j, i), m33(j, i));
    for (int j = 3; j < 5; j++)
      EXPECT_EQ(mat(j, i), m23(j-3, i));
  }    
  mat = rbind(m23, m33);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(mat(j, i), m23(j, i));
    for (int j = 2; j < 5; j++)
      EXPECT_EQ(mat(j, i), m33(j-2, i));
  }

  //matrix rbind(matrix, row_vector)
  //matrix rbind(row_vector, matrix)
  mat = rbind(m33, rv3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(j, i), m33(j, i));
    EXPECT_EQ(mat(3, i), rv3(i));
  }
  mat = rbind(rv3, m33);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(0, i), rv3(i));
    for (int j = 1; j < 4; j++)
      EXPECT_EQ(mat(j, i), m33(j-1, i));
  }
  mat = rbind(m23, rv3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(mat(j, i), m23(j, i));
    EXPECT_EQ(mat(2, i), rv3(i));
  }
  mat = rbind(rv3, m23);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(0, i), rv3(i));
    for (int j = 1; j < 3; j++)
      EXPECT_EQ(mat(j, i), m23(j-1, i));
  }
  
  //matrix rbind(row_vector, row_vector)  
  mat = rbind(rv3, rv3b);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(0, i), rv3(i));
    EXPECT_EQ(mat(1, i), rv3b(i));
  }
  mat = rbind(rv3b, rv3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(0, i), rv3b(i));
    EXPECT_EQ(mat(1, i), rv3(i));
  }
   
  //matrix rbind(vector, vector)
  cvec = rbind(v3, v3b);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(cvec(i), v3(i));
  for (int i = 3; i < 6; i++)
    EXPECT_EQ(cvec(i), v3b(i-3));
  cvec = rbind(v3b, v3);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(cvec(i), v3b(i));
  for (int i = 3; i < 6; i++)
    EXPECT_EQ(cvec(i), v3(i-3));
    
  EXPECT_THROW(rbind(m32, m33), std::domain_error);
  EXPECT_THROW(rbind(m32, m23), std::domain_error);
  EXPECT_THROW(rbind(m32, v3), std::domain_error);
  EXPECT_THROW(rbind(m32, rv3), std::domain_error);
  EXPECT_THROW(rbind(m33, m32), std::domain_error);
  EXPECT_THROW(rbind(m23, m32), std::domain_error);
  EXPECT_THROW(rbind(v3, m32), std::domain_error);
  EXPECT_THROW(rbind(rv3, m32), std::domain_error);
  
  EXPECT_THROW(rbind(v3, m33), std::domain_error);
  EXPECT_THROW(rbind(v3, m32), std::domain_error);
  EXPECT_THROW(rbind(v3, rv3), std::domain_error);
  EXPECT_THROW(rbind(m33, v3), std::domain_error);
  EXPECT_THROW(rbind(m32, v3), std::domain_error);
  EXPECT_THROW(rbind(rv3, v3), std::domain_error);
}
