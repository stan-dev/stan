#include <stan/math/prim/mat/fun/append_row.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixAppendRow,fd) {
  using stan::math::append_row;
  using stan::math::matrix_fd;
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

  matrix_fd ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_, ab_append_row(i ,j).d_);
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_append_row(i, j).val_, adb_append_row(i ,j));
}

TEST(AgradFwdVectorAppendRow,fd) {
  using stan::math::append_row;
  using stan::math::vector_fd;
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

  vector_fd ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 4; i++)
      EXPECT_EQ(a(i).d_, ab_append_row(i).d_);
  
  for (int i = 0; i < 7; i++)
      EXPECT_EQ(ab_append_row(i).val_, adb_append_row(i));
}

TEST(AgradFwdMatrixAppendRow,ffd) {
  using stan::math::append_row;
  using stan::math::matrix_ffd;
  using Eigen::MatrixXd;

  matrix_ffd a(2,2);
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

  matrix_ffd ab_append_row = append_row(a, b);
  MatrixXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_.val(), ab_append_row(i ,j).d_.val());
  
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(ab_append_row(i, j).val_.val(), adb_append_row(i ,j));
}

TEST(AgradFwdVectorAppendRow,ffd) {
  using stan::math::append_row;
  using stan::math::vector_ffd;
  using Eigen::VectorXd;

  vector_ffd a(4);
  VectorXd ad(4);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;

  vector_ffd ab_append_row = append_row(a, b);
  VectorXd adb_append_row = append_row(ad, b);
  
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(a(i).d_.val(), ab_append_row(i).d_.val());
  
  for (int i = 0; i < 7; i++)
    EXPECT_EQ(ab_append_row(i).val_.val(), adb_append_row(i));
}
