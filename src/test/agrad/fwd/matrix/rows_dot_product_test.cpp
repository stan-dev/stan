#include <stan/agrad/fwd/matrix/rows_dot_product.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

TEST(AgradFwdMatrix, rows_dot_product_vector_vector) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d vd_1(3), vd_2(3);
  vector_fv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
  
  vector_fv output(3);
  output = rows_dot_product(vv_1, vd_2);
  EXPECT_FLOAT_EQ( 4, output(0).val_);
  EXPECT_FLOAT_EQ(-6, output(1).val_);
  EXPECT_FLOAT_EQ( 5, output(2).val_);
  EXPECT_FLOAT_EQ( 4, output(0).d_);
  EXPECT_FLOAT_EQ(-2, output(1).d_);
  EXPECT_FLOAT_EQ(-1, output(2).d_);

  output = rows_dot_product(vd_1, vv_2);
  EXPECT_FLOAT_EQ( 4, output(0).val_);
  EXPECT_FLOAT_EQ(-6, output(1).val_);
  EXPECT_FLOAT_EQ( 5, output(2).val_);
  EXPECT_FLOAT_EQ( 1, output(0).d_);
  EXPECT_FLOAT_EQ( 3, output(1).d_);
  EXPECT_FLOAT_EQ(-5, output(2).d_);

  output = rows_dot_product(vv_1, vv_2);
  EXPECT_FLOAT_EQ( 4, output(0).val_);
  EXPECT_FLOAT_EQ(-6, output(1).val_);
  EXPECT_FLOAT_EQ( 5, output(2).val_);
  EXPECT_FLOAT_EQ( 5, output(0).d_);
  EXPECT_FLOAT_EQ( 1, output(1).d_);
  EXPECT_FLOAT_EQ(-6, output(2).d_);
}
TEST(AgradFwdMatrix, rows_dot_product_vector_vector_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  vector_d d2(2);
  vector_fv v2(4);

  EXPECT_THROW(rows_dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(rows_dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(rows_dot_product(v1, v2), std::domain_error);
}
TEST(AgradFwdMatrix, rows_dot_product_rowvector_vector) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  vector_d d2(3);
  vector_fv v2(3);

  EXPECT_THROW(rows_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(rows_dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(rows_dot_product(v1,v2), std::domain_error);
} 
TEST(AgradFwdMatrix, rows_dot_product_vector_rowvector) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  row_vector_d d2(3);
  row_vector_fv v2(3);

  EXPECT_THROW(rows_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(rows_dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(rows_dot_product(v1,v2), std::domain_error);
} 
TEST(AgradFwdMatrix, rows_dot_product_rowvector_rowvector) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3), d2(3);
  row_vector_fv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;

  row_vector_fv output;
  output = rows_dot_product(v1,d2);
  EXPECT_FLOAT_EQ( 3, output(0).val_);
  EXPECT_FLOAT_EQ( 1, output(0).d_);

  output = rows_dot_product(d1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_);
  EXPECT_FLOAT_EQ(-1, output(0).d_);

  output = rows_dot_product(v1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_);
  EXPECT_FLOAT_EQ( 0, output(0).d_);
}
TEST(AgradFwdMatrix, rows_dot_product_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_d d1(3,3), d2(3,3);
  matrix_fv v1(3,3), v2(3,3);
  
  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
   v1(1,0).d_ = 1.0;
   v1(1,1).d_ = 1.0;
   v1(1,2).d_ = 1.0;
   v1(2,0).d_ = 1.0;
   v1(2,1).d_ = 1.0;
   v1(2,2).d_ = 1.0;
  d2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
  v2 << 4, -2, -1, 4, -2, -1, 4, -2, -1;
   v2(0,0).d_ = 1.0;
   v2(0,1).d_ = 1.0;
   v2(0,2).d_ = 1.0;
   v2(1,0).d_ = 1.0;
   v2(1,1).d_ = 1.0;
   v2(1,2).d_ = 1.0;
   v2(2,0).d_ = 1.0;
   v2(2,1).d_ = 1.0;
   v2(2,2).d_ = 1.0;

  vector_fv output;
  output = rows_dot_product(v1,d2);
  EXPECT_FLOAT_EQ( 3, output(0).val_);
  EXPECT_FLOAT_EQ( 3, output(1).val_);
  EXPECT_FLOAT_EQ( 3, output(2).val_);
  EXPECT_FLOAT_EQ( 1, output(0).d_);
  EXPECT_FLOAT_EQ( 1, output(1).d_);
  EXPECT_FLOAT_EQ( 1, output(2).d_);

  output = rows_dot_product(d1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_);
  EXPECT_FLOAT_EQ( 3, output(1).val_);
  EXPECT_FLOAT_EQ( 3, output(2).val_);
  EXPECT_FLOAT_EQ(-1, output(0).d_);
  EXPECT_FLOAT_EQ(-1, output(1).d_);
  EXPECT_FLOAT_EQ(-1, output(2).d_);

  output = rows_dot_product(v1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_);
  EXPECT_FLOAT_EQ( 3, output(1).val_);
  EXPECT_FLOAT_EQ( 3, output(2).val_);
  EXPECT_FLOAT_EQ( 0, output(0).d_);
  EXPECT_FLOAT_EQ( 0, output(1).d_);
  EXPECT_FLOAT_EQ( 0, output(2).d_);
}
TEST(AgradFwdMatrix, rows_dot_product_matrix_matrix_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::rows_dot_product;

  matrix_d d1(3,3);
  matrix_d d2(3,2);
  matrix_d d3(2,3);
  matrix_fv v1(3,3);
  matrix_fv v2(3,3);
  matrix_fv v3(3,2);
  matrix_fv v4(3,2);
  matrix_fv v5(2,3);
  matrix_fv v6(2,3);

  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v2 << 4, -2, -1, 2, 1, 2, 1, 3, -5;
  v3 << 4, -2, -1, 2, 1, 2;
  v4 << 4, -2, -1, 2, 1, 2;
  v5 << 4, -2, -1, 2, 1, 2;
  v6 << 4, -2, -1, 2, 1, 2;

  EXPECT_THROW(rows_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(rows_dot_product(v1,d3), std::domain_error);
  EXPECT_THROW(rows_dot_product(v1,v3), std::domain_error);
  EXPECT_THROW(rows_dot_product(v1,v4), std::domain_error);
  EXPECT_THROW(rows_dot_product(v1,v5), std::domain_error);
  EXPECT_THROW(rows_dot_product(v1,v6), std::domain_error);

  EXPECT_THROW(rows_dot_product(d1,v3), std::domain_error);
  EXPECT_THROW(rows_dot_product(d1,v4), std::domain_error);
  EXPECT_THROW(rows_dot_product(d1,v5), std::domain_error);
  EXPECT_THROW(rows_dot_product(d1,v6), std::domain_error);

  EXPECT_THROW(rows_dot_product(d2,v1), std::domain_error);
  EXPECT_THROW(rows_dot_product(d2,v2), std::domain_error);
  EXPECT_THROW(rows_dot_product(d2,v5), std::domain_error);
  EXPECT_THROW(rows_dot_product(d2,v6), std::domain_error);

  EXPECT_THROW(rows_dot_product(d3,v1), std::domain_error);
  EXPECT_THROW(rows_dot_product(d3,v2), std::domain_error);
  EXPECT_THROW(rows_dot_product(d3,v3), std::domain_error);
  EXPECT_THROW(rows_dot_product(d3,v4), std::domain_error);
}
