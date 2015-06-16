#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/rev/mat/fun/LDLT_factor.hpp>
#include <gtest/gtest.h>

TEST(AgradRevMatrix, LDLT_factor_default_constructor) {
  using stan::math::LDLT_factor;
  using stan::math::var;
  
  Eigen::Matrix<var,-1,-1> A(2,2);
  A << 2,1,1,2;
  
  LDLT_factor<var,-1,-1> ldlt_A;
  
  ASSERT_FALSE(ldlt_A.success());
  EXPECT_EQ(0U, ldlt_A.rows());
  EXPECT_EQ(0U, ldlt_A.cols());
  EXPECT_NO_THROW(ldlt_A.compute(A));
}

TEST(AgradRevMatrix, LDLT_factor_constructor_matrix) {
  using stan::math::LDLT_factor;
  using stan::math::var;
  
  Eigen::Matrix<var,-1,-1> A(2,2);
  A << 2,1,1,2;
  Eigen::Matrix<double,-1,-1> B(2,2);
  B << 2,1,1,2;
  
  LDLT_factor<var,-1,-1> ldlt_A(A);
  
  ASSERT_TRUE(ldlt_A.success());
  EXPECT_NO_THROW(ldlt_A.vectorD());
  EXPECT_NO_THROW(ldlt_A.solve(B));
  EXPECT_EQ(2U, ldlt_A.rows());
  EXPECT_EQ(2U, ldlt_A.cols());
  EXPECT_NO_THROW(ldlt_A.compute(A));
}

TEST(AgradRevMatrix, success) {
  using stan::math::LDLT_factor;
  using stan::math::var;
  
  Eigen::Matrix<var,-1,-1> A(2,2);
  A << 0,0,0,0;

  LDLT_factor<var,-1,-1> ldlt_A(A);
  EXPECT_FALSE(ldlt_A.success());

  A << 2,1,1,2;
  LDLT_factor<var,-1,-1> ldlt_A2(A);
  EXPECT_TRUE(ldlt_A2.success());
}


TEST(AgradRevMatrix, solve) {
  using stan::math::LDLT_factor;
  using stan::math::var;
  
  Eigen::Matrix<var,-1,-1> A(2,2);
  A << 2,1,1,2;
  Eigen::Matrix<double,-1,-1> B(2,2);
  B(0,0) = 3;
  B(0,1) = 1;
  B(1,0) = 2;
  B(1,1) = 2;
  
  Eigen::Matrix<double,-1,-1> expected_solve(2,2);
  expected_solve(0,0) = 1.333333333;
  expected_solve(0,1) = 0.0;
  expected_solve(1,0) = 0.333333333;
  expected_solve(1,1) = 1.0;
  Eigen::Matrix<double,-1,-1> solve;

  LDLT_factor<var,-1,-1> ldlt_A(A);
  ASSERT_TRUE(ldlt_A.success());
  EXPECT_NO_THROW(solve = ldlt_A.solve(B));
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_solve(i,j), solve(i,j));
}

TEST(AgradRevMatrix, vectorD) {
  using stan::math::LDLT_factor;
  using stan::math::var;
  
  Eigen::Matrix<var,-1,-1> A(2,2);
  A << 2,1,1,2;
  Eigen::Matrix<double,-1,-1> vectorD(2,1);
  
  Eigen::Matrix<double,-1,-1> A_double(2,2);
  A_double << 2,1,1,2;
  Eigen::LDLT<Eigen::Matrix<double,2,2> > ldlt_double(A_double);
  Eigen::Matrix<double,-1,-1> expected_vectorD(2,1);
  expected_vectorD = ldlt_double.vectorD();

  
  LDLT_factor<var,-1,-1> ldlt_A(A);
  ASSERT_TRUE(ldlt_A.success());
  EXPECT_NO_THROW(vectorD = ldlt_A.vectorD());
  for (int i = 0; i < 2; i++)
    EXPECT_FLOAT_EQ(expected_vectorD(i), vectorD(i));
}


TEST(AgradRevMatrix, rows) {
  using stan::math::LDLT_factor;
  using stan::math::var;
  
  Eigen::Matrix<var,-1,-1> A(2,2);
  A << 2,1,1,2;

  LDLT_factor<var,-1,-1> ldlt_A(A);
  ASSERT_TRUE(ldlt_A.success());
  EXPECT_EQ(2U, ldlt_A.rows());
}

TEST(AgradRevMatrix, cols) {
  using stan::math::LDLT_factor;
  using stan::math::var;
  
  Eigen::Matrix<var,-1,-1> A(2,2);
  A << 2,1,1,2;

  LDLT_factor<var,-1,-1> ldlt_A(A);
  ASSERT_TRUE(ldlt_A.success());
  EXPECT_EQ(2U, ldlt_A.cols());
}

TEST(AgradRevMatrix, compute) {
  using stan::math::LDLT_factor;
  using stan::math::var;
  
  Eigen::Matrix<var,-1,-1> A(2,2);
  A << 2,1,1,2;
  Eigen::Matrix<double,-1,-1> A_double(2,2);
  A_double << 2,1,1,2;
  
  Eigen::LDLT<Eigen::Matrix<double,-1,-1> > ldlt_double(A_double);
  Eigen::Matrix<double,-1,-1> expected_mat, mat;

  LDLT_factor<var,-1,-1> ldlt_A;

  // tests on A: [2, 1][1, 2]
  // only way to test is through side-effects.
  EXPECT_NO_THROW(ldlt_A.compute(A));
  ASSERT_TRUE(ldlt_A.success());

  EXPECT_NO_THROW(mat = ldlt_A._alloc->_ldlt.matrixLDLT());
  expected_mat = ldlt_double.matrixLDLT();
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_mat(i,j), mat(i,j)) 
        << "element (" << i << ", " << j << ")";

  
  // tests on A: [0, 0][0, 0]
  A << 0, 0, 0, 0;
  EXPECT_NO_THROW(ldlt_A.compute(A));
  ASSERT_FALSE(ldlt_A.success());

  // tests on A: [1, 2, 3][2, 3, 4]
  A.resize(2,3);
  A << 1,2,2,3,3,4;
  EXPECT_THROW(ldlt_A.compute(A), std::invalid_argument);
  ASSERT_FALSE(ldlt_A.success());
}

