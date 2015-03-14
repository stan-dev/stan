#include <stan/math/prim/mat/fun/append_col.hpp>
#include <test/unit/math/prim/mat/fun/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

using stan::math::append_col;


template <int R, int C>
void correct_type_row_vector(const Eigen::Matrix<double,R,C>& x) {
  EXPECT_EQ(Eigen::Dynamic, C);
  EXPECT_EQ(1, R);
}

template <int R, int C>
void correct_type_matrix(const Eigen::Matrix<double,R,C>& x) {
  EXPECT_EQ(Eigen::Dynamic, C);
  EXPECT_EQ(Eigen::Dynamic, R);
}
  
TEST(MathMatrix, append_col) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using Eigen::RowVectorXd;
  using std::vector;
  
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
  
  //matrix append_col(matrix, matrix)
  mat = append_col(m33, m32);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(i, j), m33(i, j));
    for (int j = 3; j < 5; j++)
      EXPECT_EQ(mat(i, j), m32(i, j-3));
  }
  mat = append_col(m32, m33);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(mat(i, j), m32(i, j));
    for (int j = 2; j < 5; j++)
      EXPECT_EQ(mat(i, j), m33(i, j-2));
  }
  
  MatrixXd m23b(2, 3);
  m23b = m23*1.101; //ensure some different values
  mat = append_col(m23, m23b);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(i, j), m23(i, j));
    for (int j = 3; j < 6; j++)
      EXPECT_EQ(mat(i, j), m23b(i, j-3));
  }
  mat = append_col(m23b, m23);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(i, j), m23b(i, j));
    for (int j = 3; j < 6; j++)
      EXPECT_EQ(mat(i, j), m23(i, j-3));
  }
      
  //matrix append_col(matrix, vector)
  //matrix append_col(vector, matrix)
  mat = append_col(m33, v3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(mat(i, j), m33(i, j));
    EXPECT_EQ(mat(i, 3), v3(i));
  }
  mat = append_col(v3, m33);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(i, 0), v3(i));
    for (int j = 1; j < 4; j++)
      EXPECT_EQ(mat(i, j), m33(i, j-1));
  }
  mat = append_col(m32, v3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(mat(i, j), m32(i, j));
    EXPECT_EQ(mat(i, 2), v3(i));
  }
  mat = append_col(v3, m32);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(i, 0), v3(i));
    for (int j = 1; j < 3; j++)
      EXPECT_EQ(mat(i, j), m32(i, j-1));
  }
  
  //matrix append_col(vector, vector)  
  mat = append_col(v3, v3b);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(i, 0), v3(i));
    EXPECT_EQ(mat(i, 1), v3b(i));
  }
  mat = append_col(v3b, v3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(mat(i, 0), v3b(i));
    EXPECT_EQ(mat(i, 1), v3(i));
  }
   
  //matrix append_col(row_vector, row_vector)
  rvec = append_col(rv3, rv3b);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(rvec(i), rv3(i));
  for (int i = 3; i < 6; i++)
    EXPECT_EQ(rvec(i), rv3b(i-3));
  rvec = append_col(rv3b, rv3);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(rvec(i), rv3b(i));
  for (int i = 3; i < 6; i++)
    EXPECT_EQ(rvec(i), rv3(i-3));
    
  EXPECT_THROW(append_col(m23, m33), std::invalid_argument);
  EXPECT_THROW(append_col(m23, m32), std::invalid_argument);
  EXPECT_THROW(append_col(m23, v3), std::invalid_argument);
  EXPECT_THROW(append_col(m23, rv3), std::invalid_argument);
  EXPECT_THROW(append_col(m33, m23), std::invalid_argument);
  EXPECT_THROW(append_col(m32, m23), std::invalid_argument);
  EXPECT_THROW(append_col(v3, m23), std::invalid_argument);
  EXPECT_THROW(append_col(rv3, m23), std::invalid_argument);
  
  EXPECT_THROW(append_col(rv3, m33), std::invalid_argument);
  EXPECT_THROW(append_col(rv3, m32), std::invalid_argument);
  EXPECT_THROW(append_col(rv3, v3), std::invalid_argument);
  EXPECT_THROW(append_col(m33, rv3), std::invalid_argument);
  EXPECT_THROW(append_col(m32, rv3), std::invalid_argument);
  EXPECT_THROW(append_col(v3, rv3), std::invalid_argument);

  correct_type_matrix(append_col(m32, m33));
  correct_type_matrix(append_col(m33, m32));
  correct_type_matrix(append_col(m23, m23b));
  correct_type_matrix(append_col(m23b, m23));
  correct_type_matrix(append_col(m33, v3));
  correct_type_matrix(append_col(v3, m33));
  correct_type_matrix(append_col(m32, v3));
  correct_type_matrix(append_col(v3, m32));
  correct_type_matrix(append_col(v3, v3b));
  correct_type_matrix(append_col(v3b, v3));
  correct_type_row_vector(append_col(rv3, rv3b));
  correct_type_row_vector(append_col(rv3b, rv3));
}
