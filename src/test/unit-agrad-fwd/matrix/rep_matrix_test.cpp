#include <gtest/gtest.h>
#include <stan/math/rep_matrix.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::var;
TEST(AgradFwdMatrixRepMatrix,fd_real) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fd;
  using stan::agrad::fvar;
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
  using stan::agrad::matrix_fd;
  using stan::agrad::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_matrix(a,-2,-1), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,fd_rowvector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fd;
  using stan::agrad::row_vector_fd;
  
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
  using stan::agrad::matrix_fd;
  using stan::agrad::row_vector_fd;

  row_vector_fd a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,fd_vector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
  
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
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;

  vector_fd a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,fv_real) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  fvar<var> a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  matrix_fv output;
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
TEST(AgradFwdMatrixRepMatrix,fv_exception_real) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  fvar<var> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_matrix(a,-2,-1), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,fv_rowvector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;
  
  row_vector_fv a(3);
  a<<3.0, 3.0, 3.0;
   a(0).d_ = 2.0;
   a(1).d_ = 2.0;
   a(2).d_ = 2.0;
  matrix_fv output;
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
TEST(AgradFwdMatrixRepMatrix,fv_exception_rowvector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;

  row_vector_fv a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,fv_vector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  
  vector_fv a(3);
  a<<3.0, 3.0, 3.0;
   a(0).d_ = 2.0;
   a(1).d_ = 2.0;
   a(2).d_ = 2.0;
  matrix_fv output;
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
TEST(AgradFwdMatrixRepMatrix,fv_exception_vector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  vector_fv a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,ffd_real) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_ffd;
  using stan::agrad::fvar;
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
  using stan::agrad::matrix_ffd;
  using stan::agrad::fvar;
  fvar<fvar<double> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_matrix(a,-2,-1), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,ffd_rowvector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_ffd;
  using stan::agrad::row_vector_ffd;
  
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
  using stan::agrad::matrix_ffd;
  using stan::agrad::row_vector_ffd;

  row_vector_ffd a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,ffd_vector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  
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
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;

  vector_ffd a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,ffv_real) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  fvar<fvar<var> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  matrix_ffv output;
  output = rep_matrix(a, 2,3);

  EXPECT_EQ(3,output(0,0).val_.val().val());
  EXPECT_EQ(3,output(0,1).val_.val().val());
  EXPECT_EQ(3,output(0,2).val_.val().val());
  EXPECT_EQ(3,output(1,0).val_.val().val());
  EXPECT_EQ(3,output(1,1).val_.val().val());
  EXPECT_EQ(3,output(1,2).val_.val().val());
  EXPECT_EQ(2,output(0,0).d_.val().val());
  EXPECT_EQ(2,output(0,1).d_.val().val());
  EXPECT_EQ(2,output(0,2).d_.val().val());
  EXPECT_EQ(2,output(1,0).d_.val().val());
  EXPECT_EQ(2,output(1,1).d_.val().val());
  EXPECT_EQ(2,output(1,2).d_.val().val());
}
TEST(AgradFwdMatrixRepMatrix,ffv_exception_real) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  fvar<fvar<var> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_matrix(a,-2,-1), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,ffv_rowvector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  
  row_vector_ffv a(3);
  a<<3.0, 3.0, 3.0;
   a(0).d_ = 2.0;
   a(1).d_ = 2.0;
   a(2).d_ = 2.0;
  matrix_ffv output;
  output = rep_matrix(a, 3);

  EXPECT_EQ(3,output(0,0).val_.val().val());
  EXPECT_EQ(3,output(0,1).val_.val().val());
  EXPECT_EQ(3,output(0,2).val_.val().val());
  EXPECT_EQ(3,output(1,0).val_.val().val());
  EXPECT_EQ(3,output(1,1).val_.val().val());
  EXPECT_EQ(3,output(1,2).val_.val().val());
  EXPECT_EQ(3,output(2,0).val_.val().val());
  EXPECT_EQ(3,output(2,1).val_.val().val());
  EXPECT_EQ(3,output(2,2).val_.val().val());
  EXPECT_EQ(2,output(0,0).d_.val().val());
  EXPECT_EQ(2,output(0,1).d_.val().val());
  EXPECT_EQ(2,output(0,2).d_.val().val());
  EXPECT_EQ(2,output(1,0).d_.val().val());
  EXPECT_EQ(2,output(1,1).d_.val().val());
  EXPECT_EQ(2,output(1,2).d_.val().val());
  EXPECT_EQ(2,output(2,0).d_.val().val());
  EXPECT_EQ(2,output(2,1).d_.val().val());
  EXPECT_EQ(2,output(2,2).d_.val().val());
}
TEST(AgradFwdMatrixRepMatrix,ffv_exception_rowvector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;

  row_vector_ffv a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrixRepMatrix,ffv_vector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
  vector_ffv a(3);
  a<<3.0, 3.0, 3.0;
   a(0).d_ = 2.0;
   a(1).d_ = 2.0;
   a(2).d_ = 2.0;
  matrix_ffv output;
  output = rep_matrix(a, 3);

  EXPECT_EQ(3,output(0,0).val_.val().val());
  EXPECT_EQ(3,output(0,1).val_.val().val());
  EXPECT_EQ(3,output(0,2).val_.val().val());
  EXPECT_EQ(3,output(1,0).val_.val().val());
  EXPECT_EQ(3,output(1,1).val_.val().val());
  EXPECT_EQ(3,output(1,2).val_.val().val());
  EXPECT_EQ(3,output(2,0).val_.val().val());
  EXPECT_EQ(3,output(2,1).val_.val().val());
  EXPECT_EQ(3,output(2,2).val_.val().val());
  EXPECT_EQ(2,output(0,0).d_.val().val());
  EXPECT_EQ(2,output(0,1).d_.val().val());
  EXPECT_EQ(2,output(0,2).d_.val().val());
  EXPECT_EQ(2,output(1,0).d_.val().val());
  EXPECT_EQ(2,output(1,1).d_.val().val());
  EXPECT_EQ(2,output(1,2).d_.val().val());
  EXPECT_EQ(2,output(2,0).d_.val().val());
  EXPECT_EQ(2,output(2,1).d_.val().val());
  EXPECT_EQ(2,output(2,2).d_.val().val());
}
TEST(AgradFwdMatrixRepMatrix,ffv_exception_vector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;

  vector_ffv a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
