#include <stan/math/prim/mat/fun/append_col.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixAppendCol,fd) {
  using stan::math::append_col;
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

  matrix_fd ab_append_col = append_col(a, b);
  MatrixXd adb_append_col = append_col(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_, ab_append_col(i ,j).d_);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++)
      EXPECT_EQ(ab_append_col(i, j).val_, adb_append_col(i ,j));
}

TEST(AgradFwdRowVectorAppendCol,fd) {
  using stan::math::append_col;
  using stan::math::row_vector_fd;
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

  row_vector_fd ab_append_col = append_col(a, b);
  RowVectorXd adb_append_col = append_col(ad, b);
  
  for (int i = 0; i < 4; i++)
      EXPECT_EQ(a(i).d_, ab_append_col(i).d_);
  
  for (int i = 0; i < 7; i++)
      EXPECT_EQ(ab_append_col(i).val_, adb_append_col(i));
}

TEST(AgradFwdMatrixAppendCol,ffd) {
  using stan::math::append_col;
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

  matrix_ffd ab_append_col = append_col(a, b);
  MatrixXd adb_append_col = append_col(ad, b);
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(a(i, j).d_.val(), ab_append_col(i ,j).d_.val());
  
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++)
      EXPECT_EQ(ab_append_col(i, j).val_.val(), adb_append_col(i ,j));
}

TEST(AgradFwdRowVectorAppendCol,ffd) {
  using stan::math::append_col;
  using stan::math::row_vector_ffd;
  using Eigen::RowVectorXd;

  row_vector_ffd a(4);
  RowVectorXd ad(4);
  RowVectorXd b(3);
  
  a << 2.0, 3.0, 9.0, -1.0;
       
  ad << 2.0, 3.0, 9.0, -1.0;
       
  b << 4.0, 3.0, 0.4;

  a(0).d_ = 2.0;
  a(1).d_ = 3.0;
  a(2).d_ = 4.0;
  a(3).d_ = 5.0;

  row_vector_ffd ab_append_col = append_col(a, b);
  RowVectorXd adb_append_col = append_col(ad, b);
  
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(a(i).d_.val(), ab_append_col(i).d_.val());
  
  for (int i = 0; i < 7; i++)
    EXPECT_EQ(ab_append_col(i).val_.val(), adb_append_col(i));
}
