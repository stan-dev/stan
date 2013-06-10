#include <stan/math/matrix/sd.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/sqrt.hpp>
#include <stan/agrad/fwd/operator_multiplication.hpp>
#include <stan/agrad/fwd/operator_addition.hpp>
#include <stan/agrad/fwd/operator_division.hpp>
#include <stan/agrad/fwd/operator_subtraction.hpp>
#include <stan/agrad/var.hpp>

TEST(AgradFwdMatrix, sd_vector) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  vector_d d1(6);
  vector_fv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_);
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_);
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_);
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_);
}
TEST(AgradFwdMatrix, sd_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d1;
  vector_fv v1;
  EXPECT_THROW(sd(d1), std::domain_error);
  EXPECT_THROW(sd(v1), std::domain_error);
}
TEST(AgradFwdMatrix, sd_rowvector) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));


  row_vector_d d1(6);
  row_vector_fv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_);
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_);

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_);
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_);
}
TEST(AgradFwdMatrix, sd_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d;
  row_vector_fv v;
  
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdMatrix, sd_matrix) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d v(1,1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  matrix_d d1(2, 3);
  matrix_fv v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_);
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_);

  d1.resize(1, 1);
  v1.resize(1, 1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_);
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_);
}
TEST(AgradFwdMatrix, sd_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d d;
  matrix_fv v;

  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdFvarVarMatrix, sd_vector) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_fvv;

  vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  vector_d d1(6);
  vector_fvv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdFvarVarMatrix, sd_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_fvv;

  vector_d d1;
  vector_fvv v1;
  EXPECT_THROW(sd(d1), std::domain_error);
  EXPECT_THROW(sd(v1), std::domain_error);
}
TEST(AgradFwdFvarVarMatrix, sd_rowvector) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fvv;

  row_vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));


  row_vector_d d1(6);
  row_vector_fvv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdFvarVarMatrix, sd_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fvv;

  row_vector_d d;
  row_vector_fvv v;
  
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdFvarVarMatrix, sd_matrix) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fvv;

  matrix_d v(1,1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  matrix_d d1(2, 3);
  matrix_fvv v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());

  d1.resize(1, 1);
  v1.resize(1, 1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdFvarVarMatrix, sd_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fvv;

  matrix_d d;
  matrix_fvv v;

  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdFvarFvarMatrix, sd_vector) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

  vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  vector_d d1(6);
  vector_ffv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, sd_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

  vector_d d1;
  vector_ffv v1;
  EXPECT_THROW(sd(d1), std::domain_error);
  EXPECT_THROW(sd(v1), std::domain_error);
}
TEST(AgradFwdFvarFvarMatrix, sd_rowvector) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

  row_vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));


  row_vector_d d1(6);
  row_vector_ffv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, sd_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

  row_vector_d d;
  row_vector_ffv v;
  
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdFvarFvarMatrix, sd_matrix) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

  matrix_d v(1,1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  matrix_d d1(2, 3);
  matrix_ffv v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());

  d1.resize(1, 1);
  v1.resize(1, 1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, sd_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

  matrix_d d;
  matrix_ffv v;

  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
