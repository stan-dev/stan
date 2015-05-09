#include <stan/math/fwd/mat/fun/to_fvar.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>

TEST(AgradFwdMatrixToFvar,fd_scalar) {
  using stan::math::fvar;
  double d = 5.0;
  fvar<double> v = 5.0;
   v.d_  = 1.0;
  fvar<double> fvar_x = stan::math::to_fvar(d);
  EXPECT_FLOAT_EQ(5.0, fvar_x.val_);
  EXPECT_FLOAT_EQ(0.0, fvar_x.d_);

  fvar_x = stan::math::to_fvar(v);
  EXPECT_FLOAT_EQ(5.0, fvar_x.val_);
  EXPECT_FLOAT_EQ(1.0, fvar_x.d_);
}
TEST(AgradFwdMatrixToFvar,fd_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  matrix_d m_d(2,3);
  m_d << 0, 1, 2, 3, 4, 5;
  matrix_fd m_v = stan::math::to_fvar(m_d);

  EXPECT_EQ(2, m_v.rows());
  EXPECT_EQ(3, m_v.cols());
  for (int ii = 0; ii < 2; ii++) 
    for (int jj = 0; jj < 3; jj++) {
      EXPECT_FLOAT_EQ(ii*3 + jj, m_v(ii, jj).val_);
      EXPECT_FLOAT_EQ(0.0, m_v(ii, jj).d_);
    }
}
TEST(AgradFwdMatrixToFvar,fd_vector) {
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d(5);
  vector_fd v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
  
  vector_fd out = stan::math::to_fvar(d);
  EXPECT_FLOAT_EQ(1, out(0).val_);
  EXPECT_FLOAT_EQ(2, out(1).val_);
  EXPECT_FLOAT_EQ(3, out(2).val_);
  EXPECT_FLOAT_EQ(4, out(3).val_);
  EXPECT_FLOAT_EQ(5, out(4).val_);
  EXPECT_FLOAT_EQ(0, out(0).d_);
  EXPECT_FLOAT_EQ(0, out(1).d_);
  EXPECT_FLOAT_EQ(0, out(2).d_);
  EXPECT_FLOAT_EQ(0, out(3).d_);
  EXPECT_FLOAT_EQ(0, out(4).d_);

  out = stan::math::to_fvar(v);
  EXPECT_FLOAT_EQ(1, out(0).val_);
  EXPECT_FLOAT_EQ(2, out(1).val_);
  EXPECT_FLOAT_EQ(3, out(2).val_);
  EXPECT_FLOAT_EQ(4, out(3).val_);
  EXPECT_FLOAT_EQ(5, out(4).val_);  
  EXPECT_FLOAT_EQ(1, out(0).d_);
  EXPECT_FLOAT_EQ(1, out(1).d_);
  EXPECT_FLOAT_EQ(1, out(2).d_);
  EXPECT_FLOAT_EQ(1, out(3).d_);
  EXPECT_FLOAT_EQ(1, out(4).d_);
}
TEST(AgradFwdMatrixToFvar,fd_rowvector) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d(5);
  row_vector_fd v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
  
  row_vector_fd output = stan::math::to_fvar(d);
  EXPECT_FLOAT_EQ(1, output(0).val_);
  EXPECT_FLOAT_EQ(2, output(1).val_);
  EXPECT_FLOAT_EQ(3, output(2).val_);
  EXPECT_FLOAT_EQ(4, output(3).val_);
  EXPECT_FLOAT_EQ(5, output(4).val_);
  EXPECT_FLOAT_EQ(0, output(0).d_);
  EXPECT_FLOAT_EQ(0, output(1).d_);
  EXPECT_FLOAT_EQ(0, output(2).d_);
  EXPECT_FLOAT_EQ(0, output(3).d_);
  EXPECT_FLOAT_EQ(0, output(4).d_);

  output.resize(0);
  output = stan::math::to_fvar(v);
  EXPECT_FLOAT_EQ(1, output(0).val_);
  EXPECT_FLOAT_EQ(2, output(1).val_);
  EXPECT_FLOAT_EQ(3, output(2).val_);
  EXPECT_FLOAT_EQ(4, output(3).val_);
  EXPECT_FLOAT_EQ(5, output(4).val_);
  EXPECT_FLOAT_EQ(1, output(0).d_);
  EXPECT_FLOAT_EQ(1, output(1).d_);
  EXPECT_FLOAT_EQ(1, output(2).d_);
  EXPECT_FLOAT_EQ(1, output(3).d_);
  EXPECT_FLOAT_EQ(1, output(4).d_);
}
TEST(AgradFwdMatrixToFvar,fd_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;

  matrix_d val(3,3);
  matrix_d d(3,3);
  
  val <<1,2,3,4,5,6,7,8,9;
  d <<10,11,12,13,14,15,16,17,18;
  
  matrix_fd output = stan::math::to_fvar(val,d);
  EXPECT_FLOAT_EQ(1, output(0,0).val_);
  EXPECT_FLOAT_EQ(2, output(0,1).val_);
  EXPECT_FLOAT_EQ(3, output(0,2).val_);
  EXPECT_FLOAT_EQ(4, output(1,0).val_);
  EXPECT_FLOAT_EQ(5, output(1,1).val_);
  EXPECT_FLOAT_EQ(6, output(1,2).val_);
  EXPECT_FLOAT_EQ(7, output(2,0).val_);
  EXPECT_FLOAT_EQ(8, output(2,1).val_);
  EXPECT_FLOAT_EQ(9, output(2,2).val_);
  EXPECT_FLOAT_EQ(10, output(0,0).d_);
  EXPECT_FLOAT_EQ(11, output(0,1).d_);
  EXPECT_FLOAT_EQ(12, output(0,2).d_);
  EXPECT_FLOAT_EQ(13, output(1,0).d_);
  EXPECT_FLOAT_EQ(14, output(1,1).d_);
  EXPECT_FLOAT_EQ(15, output(1,2).d_);
  EXPECT_FLOAT_EQ(16, output(2,0).d_);
  EXPECT_FLOAT_EQ(17, output(2,1).d_);
  EXPECT_FLOAT_EQ(18, output(2,2).d_);

  matrix_d val1(4,4);
  EXPECT_THROW(stan::math::to_fvar(val1, d), std::invalid_argument);
}
TEST(AgradFwdMatrixToFvar,ffd_vector) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_ffd v(5);
  
  v << 1, 2, 3, 4, 5;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;

  vector_ffd out = stan::math::to_fvar(v);
  EXPECT_FLOAT_EQ(1, out(0).val_.val());
  EXPECT_FLOAT_EQ(2, out(1).val_.val());
  EXPECT_FLOAT_EQ(3, out(2).val_.val());
  EXPECT_FLOAT_EQ(4, out(3).val_.val());
  EXPECT_FLOAT_EQ(5, out(4).val_.val());  
  EXPECT_FLOAT_EQ(1, out(0).d_.val());
  EXPECT_FLOAT_EQ(1, out(1).d_.val());
  EXPECT_FLOAT_EQ(1, out(2).d_.val());
  EXPECT_FLOAT_EQ(1, out(3).d_.val());
  EXPECT_FLOAT_EQ(1, out(4).d_.val());
}
TEST(AgradFwdMatrixToFvar,ffd_rowvector) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_ffd v(5);
  
  v << 1, 2, 3, 4, 5;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;

  row_vector_ffd output = stan::math::to_fvar(v);
  EXPECT_FLOAT_EQ(1, output(0).val_.val());
  EXPECT_FLOAT_EQ(2, output(1).val_.val());
  EXPECT_FLOAT_EQ(3, output(2).val_.val());
  EXPECT_FLOAT_EQ(4, output(3).val_.val());
  EXPECT_FLOAT_EQ(5, output(4).val_.val());
  EXPECT_FLOAT_EQ(1, output(0).d_.val());
  EXPECT_FLOAT_EQ(1, output(1).d_.val());
  EXPECT_FLOAT_EQ(1, output(2).d_.val());
  EXPECT_FLOAT_EQ(1, output(3).d_.val());
  EXPECT_FLOAT_EQ(1, output(4).d_.val());
}
TEST(AgradFwdMatrixToFvar,ffd_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::matrix_fd;

  matrix_fd val(3,3);
  matrix_fd d(3,3);
  
  val <<1,2,3,4,5,6,7,8,9;
  d <<10,11,12,13,14,15,16,17,18;
  
  matrix_ffd output = stan::math::to_fvar(val,d);
  EXPECT_FLOAT_EQ(1, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(2, output(0,1).val_.val());
  EXPECT_FLOAT_EQ(3, output(0,2).val_.val());
  EXPECT_FLOAT_EQ(4, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(5, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(6, output(1,2).val_.val());
  EXPECT_FLOAT_EQ(7, output(2,0).val_.val());
  EXPECT_FLOAT_EQ(8, output(2,1).val_.val());
  EXPECT_FLOAT_EQ(9, output(2,2).val_.val());
  EXPECT_FLOAT_EQ(10, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(11, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(12, output(0,2).d_.val());
  EXPECT_FLOAT_EQ(13, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(14, output(1,1).d_.val());
  EXPECT_FLOAT_EQ(15, output(1,2).d_.val());
  EXPECT_FLOAT_EQ(16, output(2,0).d_.val());
  EXPECT_FLOAT_EQ(17, output(2,1).d_.val());
  EXPECT_FLOAT_EQ(18, output(2,2).d_.val());
}
