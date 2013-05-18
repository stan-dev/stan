#include <gtest/gtest.h>
#include <stan/math/rep_matrix.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>

TEST(AgradFwdMatrix,rep_matrix_real) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  matrix_fv output;
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
TEST(AgradFwdMatrix,rep_matrix_exception_real) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_matrix(a,-2,-1), std::domain_error);
}
TEST(AgradFwdMatrix,rep_matrix_rowvector) {
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
TEST(AgradFwdMatrix,rep_matrix_exception_rowvector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;

  row_vector_fv a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradFwdMatrix,rep_matrix_vector) {
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
TEST(AgradFwdMatrix,rep_matrix_exception_vector) {
  using stan::math::rep_matrix;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  vector_fv a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
