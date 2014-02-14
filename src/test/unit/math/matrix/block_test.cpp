#include <stdexcept>
#include <stan/math/matrix/block.hpp>

#include <gtest/gtest.h>

TEST(MathMatrixBlock,Block1) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::block;
  
  Matrix<double,Dynamic,Dynamic> m(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i,j) = (i + 1) * (j + 1);

  
  Matrix<double,Dynamic,Dynamic> b
    = block(m,2,3,1,1);
  EXPECT_EQ(1,b.rows());
  EXPECT_EQ(1,b.cols());
  EXPECT_FLOAT_EQ(m(2-1,3-1), b(0,0));

  Matrix<double,Dynamic,Dynamic> b2
    = block(m,2,3,0,0);
  EXPECT_EQ(0,b2.rows());
  EXPECT_EQ(0,b2.cols());

  Matrix<double,Dynamic,Dynamic> b3
    = block(m,1,1,3,4);
  EXPECT_EQ(3,b3.rows());
  EXPECT_EQ(4,b3.cols());
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(m(i,j),b3(i,j));

  Matrix<double,Dynamic,Dynamic> b5
    = block(m,1,1,3,4);
  EXPECT_EQ(3,b5.rows());
  EXPECT_EQ(4,b5.cols());
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(m(i,j),b5(i,j));

  Matrix<double,Dynamic,Dynamic> b4
    = block(m,2,2,2,3);
  EXPECT_EQ(2,b4.rows());
  EXPECT_EQ(3,b4.cols());
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ(m(1 + i, 1 + j),b4(i,j));

  EXPECT_THROW(block(m,5,2,1,1), std::domain_error);
  EXPECT_THROW(block(m,1,7,1,1), std::domain_error);
  EXPECT_THROW(block(m,1,1,6,1), std::domain_error);
  EXPECT_THROW(block(m,1,1,1,6), std::domain_error);
}

