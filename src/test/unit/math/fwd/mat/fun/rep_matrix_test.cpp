#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/rep_matrix.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixRepMatrix,fd_real) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fd;
  using stan::math::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  matrix_fd output;
  output = rep_matrix(a, 2,3);

  EXPECT_EQ(3,output(0,0).val_);
  EXPECT_EQ(3,output(0,1).val_);
  EXPECT_EQ(3,output(0,2).val_);
  EXPECT_EQ(3,output(1,0).val_);
  EXPECT_EQ(3,output(1,1).val_);
  EXPECT_EQ(3,output(1,2).val_);
  EXPECT_EQ(2,output(0,0).d_);
  EXPECT_EQ(2,output(0,1).d_);
  EXPECT_EQ(2,output(0,2).d_);
  EXPECT_EQ(2,output(1,0).d_);
  EXPECT_EQ(2,output(1,1).d_);
  EXPECT_EQ(2,output(1,2).d_);
}
TEST(AgradFwdMatrixRepMatrix,fd_exception_real) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fd;
  using stan::math::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_matrix(a,-2,-1), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,fd_rowvector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;
  
  row_vector_fd a(3);
  a<<3.0, 3.0, 3.0;
   a(0).d_ = 2.0;
   a(1).d_ = 2.0;
   a(2).d_ = 2.0;
  matrix_fd output;
  output = rep_matrix(a, 3);

  EXPECT_EQ(3,output(0,0).val_);
  EXPECT_EQ(3,output(0,1).val_);
  EXPECT_EQ(3,output(0,2).val_);
  EXPECT_EQ(3,output(1,0).val_);
  EXPECT_EQ(3,output(1,1).val_);
  EXPECT_EQ(3,output(1,2).val_);
  EXPECT_EQ(3,output(2,0).val_);
  EXPECT_EQ(3,output(2,1).val_);
  EXPECT_EQ(3,output(2,2).val_);
  EXPECT_EQ(2,output(0,0).d_);
  EXPECT_EQ(2,output(0,1).d_);
  EXPECT_EQ(2,output(0,2).d_);
  EXPECT_EQ(2,output(1,0).d_);
  EXPECT_EQ(2,output(1,1).d_);
  EXPECT_EQ(2,output(1,2).d_);
  EXPECT_EQ(2,output(2,0).d_);
  EXPECT_EQ(2,output(2,1).d_);
  EXPECT_EQ(2,output(2,2).d_);
}
TEST(AgradFwdMatrixRepMatrix,fd_exception_rowvector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;

  row_vector_fd a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,fd_vector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  
  vector_fd a(3);
  a<<3.0, 3.0, 3.0;
   a(0).d_ = 2.0;
   a(1).d_ = 2.0;
   a(2).d_ = 2.0;
  matrix_fd output;
  output = rep_matrix(a, 3);

  EXPECT_EQ(3,output(0,0).val_);
  EXPECT_EQ(3,output(0,1).val_);
  EXPECT_EQ(3,output(0,2).val_);
  EXPECT_EQ(3,output(1,0).val_);
  EXPECT_EQ(3,output(1,1).val_);
  EXPECT_EQ(3,output(1,2).val_);
  EXPECT_EQ(3,output(2,0).val_);
  EXPECT_EQ(3,output(2,1).val_);
  EXPECT_EQ(3,output(2,2).val_);
  EXPECT_EQ(2,output(0,0).d_);
  EXPECT_EQ(2,output(0,1).d_);
  EXPECT_EQ(2,output(0,2).d_);
  EXPECT_EQ(2,output(1,0).d_);
  EXPECT_EQ(2,output(1,1).d_);
  EXPECT_EQ(2,output(1,2).d_);
  EXPECT_EQ(2,output(2,0).d_);
  EXPECT_EQ(2,output(2,1).d_);
  EXPECT_EQ(2,output(2,2).d_);
}
TEST(AgradFwdMatrixRepMatrix,fd_exception_vector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;

  vector_fd a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,ffd_real) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffd;
  using stan::math::fvar;
  fvar<fvar<double> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  matrix_ffd output;
  output = rep_matrix(a, 2,3);

  EXPECT_EQ(3,output(0,0).val_.val());
  EXPECT_EQ(3,output(0,1).val_.val());
  EXPECT_EQ(3,output(0,2).val_.val());
  EXPECT_EQ(3,output(1,0).val_.val());
  EXPECT_EQ(3,output(1,1).val_.val());
  EXPECT_EQ(3,output(1,2).val_.val());
  EXPECT_EQ(2,output(0,0).d_.val());
  EXPECT_EQ(2,output(0,1).d_.val());
  EXPECT_EQ(2,output(0,2).d_.val());
  EXPECT_EQ(2,output(1,0).d_.val());
  EXPECT_EQ(2,output(1,1).d_.val());
  EXPECT_EQ(2,output(1,2).d_.val());
}
TEST(AgradFwdMatrixRepMatrix,ffd_exception_real) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffd;
  using stan::math::fvar;
  fvar<fvar<double> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_matrix(a,-2,-1), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,ffd_rowvector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;
  
  row_vector_ffd a(3);
  a<<3.0, 3.0, 3.0;
   a(0).d_ = 2.0;
   a(1).d_ = 2.0;
   a(2).d_ = 2.0;
  matrix_ffd output;
  output = rep_matrix(a, 3);

  EXPECT_EQ(3,output(0,0).val_.val());
  EXPECT_EQ(3,output(0,1).val_.val());
  EXPECT_EQ(3,output(0,2).val_.val());
  EXPECT_EQ(3,output(1,0).val_.val());
  EXPECT_EQ(3,output(1,1).val_.val());
  EXPECT_EQ(3,output(1,2).val_.val());
  EXPECT_EQ(3,output(2,0).val_.val());
  EXPECT_EQ(3,output(2,1).val_.val());
  EXPECT_EQ(3,output(2,2).val_.val());
  EXPECT_EQ(2,output(0,0).d_.val());
  EXPECT_EQ(2,output(0,1).d_.val());
  EXPECT_EQ(2,output(0,2).d_.val());
  EXPECT_EQ(2,output(1,0).d_.val());
  EXPECT_EQ(2,output(1,1).d_.val());
  EXPECT_EQ(2,output(1,2).d_.val());
  EXPECT_EQ(2,output(2,0).d_.val());
  EXPECT_EQ(2,output(2,1).d_.val());
  EXPECT_EQ(2,output(2,2).d_.val());
}
TEST(AgradFwdMatrixRepMatrix,ffd_exception_rowvector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;

  row_vector_ffd a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,ffd_vector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  
  vector_ffd a(3);
  a<<3.0, 3.0, 3.0;
   a(0).d_ = 2.0;
   a(1).d_ = 2.0;
   a(2).d_ = 2.0;
  matrix_ffd output;
  output = rep_matrix(a, 3);

  EXPECT_EQ(3,output(0,0).val_.val());
  EXPECT_EQ(3,output(0,1).val_.val());
  EXPECT_EQ(3,output(0,2).val_.val());
  EXPECT_EQ(3,output(1,0).val_.val());
  EXPECT_EQ(3,output(1,1).val_.val());
  EXPECT_EQ(3,output(1,2).val_.val());
  EXPECT_EQ(3,output(2,0).val_.val());
  EXPECT_EQ(3,output(2,1).val_.val());
  EXPECT_EQ(3,output(2,2).val_.val());
  EXPECT_EQ(2,output(0,0).d_.val());
  EXPECT_EQ(2,output(0,1).d_.val());
  EXPECT_EQ(2,output(0,2).d_.val());
  EXPECT_EQ(2,output(1,0).d_.val());
  EXPECT_EQ(2,output(1,1).d_.val());
  EXPECT_EQ(2,output(1,2).d_.val());
  EXPECT_EQ(2,output(2,0).d_.val());
  EXPECT_EQ(2,output(2,1).d_.val());
  EXPECT_EQ(2,output(2,2).d_.val());
}
TEST(AgradFwdMatrixRepMatrix,ffd_exception_vector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;

  vector_ffd a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
