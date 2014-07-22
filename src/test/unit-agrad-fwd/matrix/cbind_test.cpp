#include <stan/math/matrix/cbind.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixCbind,fd) {
  using stan::math::cbind;
  using stan::agrad::matrix_fd;
  using Eigen::MatrixXd;

  matrix_fd a(2,2);
  MatrixXd ad(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  ad << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  a(0,0).d_ = 2.0;
  a(0,1).d_ = 3.0;
  a(1,0).d_ = 4.0;
  a(1,1).d_ = 5.0;

  matrix_fd ab_cbind = cbind(a, b);
  MatrixXd adb_cbind = cbind(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_, ab_cbind(i ,j).d_);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++)
      EXPECT_EQ(ab_cbind(i, j).val_, adb_cbind(i ,j));
}

TEST(AgradFwdRowVectorCbind,fd) {
  using stan::math::cbind;
  using stan::agrad::row_vector_fd;
  using Eigen::RowVectorXd;

  row_vector_fd a(4);
  RowVectorXd ad(4);
  RowVectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;

  row_vector_fd ab_cbind = cbind(a, b);
  RowVectorXd adb_cbind = cbind(ad, b);
  
  for (int i = 0; i < 4; i++)
      EXPECT_EQ(a(i).d_, ab_cbind(i).d_);
  
  for (int i = 0; i < 7; i++)
      EXPECT_EQ(ab_cbind(i).val_, adb_cbind(i));
}
