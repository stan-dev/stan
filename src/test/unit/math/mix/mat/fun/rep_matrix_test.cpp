#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/rep_matrix.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>

using stan::math::var;
TEST(AgradMixMatrixRepMatrix,fv_real) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fv;
  using stan::math::fvar;
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
TEST(AgradMixMatrixRepMatrix,fv_exception_real) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fv;
  using stan::math::fvar;
  fvar<var> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_matrix(a,-2,-1), std::domain_error);
}
TEST(AgradMixMatrixRepMatrix,fv_rowvector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  
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
TEST(AgradMixMatrixRepMatrix,fv_exception_rowvector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;

  row_vector_fv a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradMixMatrixRepMatrix,fv_vector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  
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
TEST(AgradMixMatrixRepMatrix,fv_exception_vector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;

  vector_fv a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradMixMatrixRepMatrix,ffv_real) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
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
TEST(AgradMixMatrixRepMatrix,ffv_exception_real) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  fvar<fvar<var> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_matrix(a,-2,-1), std::domain_error);
}
TEST(AgradMixMatrixRepMatrix,ffv_rowvector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  
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
TEST(AgradMixMatrixRepMatrix,ffv_exception_rowvector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  row_vector_ffv a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
TEST(AgradMixMatrixRepMatrix,ffv_vector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  
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
TEST(AgradMixMatrixRepMatrix,ffv_exception_vector) {
  using stan::math::rep_matrix;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  vector_ffv a(3);
  a<<3.0, 3.0, 3.0;

  EXPECT_THROW(rep_matrix(a,-3), std::domain_error);
}
