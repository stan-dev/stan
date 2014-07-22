#include <stan/math/matrix/rbind.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixrbind,fd) {
  using stan::math::rbind;
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

  matrix_fd ab_rbind = rbind(a, b);
  MatrixXd adb_rbind = rbind(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_, ab_rbind(i ,j).d_);
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_rbind(i, j).val_, adb_rbind(i ,j));
}

TEST(AgradFwdVectorrbind,fd) {
  using stan::math::rbind;
  using stan::agrad::vector_fd;
  using Eigen::VectorXd;

  vector_fd a(4);
  VectorXd ad(4);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;

  vector_fd ab_rbind = rbind(a, b);
  VectorXd adb_rbind = rbind(ad, b);
  
  for (int i = 0; i < 4; i++)
      EXPECT_EQ(a(i).d_, ab_rbind(i).d_);
  
  for (int i = 0; i < 7; i++)
      EXPECT_EQ(ab_rbind(i).val_, adb_rbind(i));
}
