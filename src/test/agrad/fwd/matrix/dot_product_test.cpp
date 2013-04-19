#include <stan/agrad/fwd/matrix/dot_product.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

TEST(AgradFwdMatrix, dot_product_vector_vector) {
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

  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(vv_1, vd_2).val_);
  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(vd_1, vv_2).val_);
  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(vv_1, vv_2).val_);  
  EXPECT_FLOAT_EQ( 1, stan::agrad::dot_product(vv_1, vd_2).d_);
  EXPECT_FLOAT_EQ(-1, stan::agrad::dot_product(vd_1, vv_2).d_);
  EXPECT_FLOAT_EQ( 0, stan::agrad::dot_product(vv_1, vv_2).d_);
}

TEST(AgradFwdMatrix, dot_product_vector_vector_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  vector_d d2(2);
  vector_fv v2(4);

  EXPECT_THROW(stan::agrad::dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(v1, v2), std::domain_error);
}
TEST(AgradFwdMatrix, dot_product_rowvector_vector) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  vector_d d2(3);
  vector_fv v2(3);
  
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

  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(v1, d2).val_);
  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(d1, v2).val_);
  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(v1, v2).val_);
  EXPECT_FLOAT_EQ( 1, stan::agrad::dot_product(v1, d2).d_);
  EXPECT_FLOAT_EQ(-1, stan::agrad::dot_product(d1, v2).d_);
  EXPECT_FLOAT_EQ( 0, stan::agrad::dot_product(v1, v2).d_);
}

TEST(AgradFwdMatrix, dot_product_rowvector_vector_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  vector_d d2(2);
  vector_fv v2(4);

  EXPECT_THROW(stan::agrad::dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(v1, v2), std::domain_error);
}

TEST(AgradFwdMatrix, dot_product_vector_rowvector) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  row_vector_d d2(3);
  row_vector_fv v2(3);
  
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

  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(v1, d2).val_);
  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(d1, v2).val_);
  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(v1, v2).val_);
  EXPECT_FLOAT_EQ( 1, stan::agrad::dot_product(v1, d2).d_);
  EXPECT_FLOAT_EQ(-1, stan::agrad::dot_product(d1, v2).d_);
  EXPECT_FLOAT_EQ( 0, stan::agrad::dot_product(v1, v2).d_);
}

TEST(AgradFwdMatrix, dot_product_vector_rowvector_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  row_vector_d d2(2);
  row_vector_fv v2(4);

  EXPECT_THROW(stan::agrad::dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(v1, v2), std::domain_error);
}

TEST(AgradFwdMatrix, dot_product_rowvector_rowvector) {
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

  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(v1, d2).val_);
  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(d1, v2).val_);
  EXPECT_FLOAT_EQ( 3, stan::agrad::dot_product(v1, v2).val_);
  EXPECT_FLOAT_EQ( 1, stan::agrad::dot_product(v1, d2).d_);
  EXPECT_FLOAT_EQ(-1, stan::agrad::dot_product(d1, v2).d_);
  EXPECT_FLOAT_EQ( 0, stan::agrad::dot_product(v1, v2).d_);
}

TEST(AgradFwdMatrix, dot_product_rowvector_rowvector_exception) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3), d2(2);
  row_vector_fv v1(3), v2(4);

  EXPECT_THROW(stan::agrad::dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(v1, v2), std::domain_error);
}

TEST(AgradFwdMatrix, dot_product_stdvector_stdvector) {
  using std::vector;
  using stan::agrad::fvar;

  vector<fvar<double> > fv1;
  vector<fvar<double> > fv2;
  vector<double> dv;

  fvar<double> a = 1.0;
  fvar<double> b = 3.0;
  fvar<double> c = 5.0;
  a.d_ = 1.0;
  b.d_ = 1.0;
  c.d_ = 1.0;

  fv1.push_back(a);
  fv1.push_back(b);
  fv1.push_back(c); 
  fv2.push_back(a);
  fv2.push_back(b);
  fv2.push_back(c);   
  dv.push_back(2.0);
  dv.push_back(4.0);
  dv.push_back(6.0);

  EXPECT_FLOAT_EQ(44.0, dot_product(fv1, dv).val_);
  EXPECT_FLOAT_EQ(44.0, dot_product(dv, fv1).val_);
  EXPECT_FLOAT_EQ(35.0, dot_product(fv1, fv2).val_);
  EXPECT_FLOAT_EQ(12.0, dot_product(fv1, dv).d_);
  EXPECT_FLOAT_EQ(12.0, dot_product(dv, fv1).d_);
  EXPECT_FLOAT_EQ(18.0, dot_product(fv1, fv2).d_);
}
TEST(AgradFwdMatrix, dot_product_matrix_matrix_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::dot_product;
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

  EXPECT_THROW(dot_product(v1,d1), std::domain_error);
  EXPECT_THROW(dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(dot_product(v1,d3), std::domain_error);
  EXPECT_THROW(dot_product(v1,v2), std::domain_error);
  EXPECT_THROW(dot_product(v1,v3), std::domain_error);
  EXPECT_THROW(dot_product(v1,v4), std::domain_error);
  EXPECT_THROW(dot_product(v1,v5), std::domain_error);
  EXPECT_THROW(dot_product(v1,v6), std::domain_error);

  EXPECT_THROW(dot_product(v2,d1), std::domain_error);
  EXPECT_THROW(dot_product(v2,d2), std::domain_error);
  EXPECT_THROW(dot_product(v2,d3), std::domain_error);
  EXPECT_THROW(dot_product(v2,v1), std::domain_error);
  EXPECT_THROW(dot_product(v2,v3), std::domain_error);
  EXPECT_THROW(dot_product(v2,v4), std::domain_error);
  EXPECT_THROW(dot_product(v2,v5), std::domain_error);
  EXPECT_THROW(dot_product(v2,v6), std::domain_error);

  EXPECT_THROW(dot_product(d1,v1), std::domain_error);
  EXPECT_THROW(dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(dot_product(d1,v3), std::domain_error);
  EXPECT_THROW(dot_product(d1,v4), std::domain_error);
  EXPECT_THROW(dot_product(d1,v5), std::domain_error);
  EXPECT_THROW(dot_product(d1,v6), std::domain_error);

  EXPECT_THROW(dot_product(d2,v1), std::domain_error);
  EXPECT_THROW(dot_product(d2,v2), std::domain_error);
  EXPECT_THROW(dot_product(d2,v3), std::domain_error);
  EXPECT_THROW(dot_product(d2,v4), std::domain_error);
  EXPECT_THROW(dot_product(d2,v5), std::domain_error);
  EXPECT_THROW(dot_product(d2,v6), std::domain_error);
}
TEST(AgradFwdMatrix, dot_product_vector_vector_length) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d vd_1(3), vd_2(3);
  vector_fv vv_1(3), vv_2(3);
  stan::agrad::size_type length = 2;
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

  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(vv_1, vd_2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(vd_1, vv_2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(vv_1, vv_2, length).val_);  
  EXPECT_FLOAT_EQ( 2, stan::agrad::dot_product(vv_1, vd_2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::agrad::dot_product(vd_1, vv_2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::agrad::dot_product(vv_1, vv_2, length).d_);
}
TEST(AgradFwdMatrix, dot_product_vector_vector_no_exception_length) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  vector_d d2(2);
  vector_fv v2(4);
  stan::agrad::size_type length = 2;

  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2;
  v2 << 4, -2, -1, 2;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;
   v2(3).d_ = 1.0;

  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::agrad::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::agrad::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::agrad::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrix, dot_product_rowvector_vector_length) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  vector_d d2(3);
  vector_fv v2(3);
  stan::agrad::size_type length = 2;

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

  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::agrad::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::agrad::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::agrad::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrix, dot_product_rowvector_vector_no_exception_length) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  vector_d d2(2);
  vector_fv v2(4);
  stan::agrad::size_type length = 2;

  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2;
  v2 << 4, -2, -1, 2;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;
   v2(3).d_ = 1.0;

  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::agrad::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::agrad::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::agrad::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrix, dot_product_vector_rowvector_length) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  row_vector_d d2(3);
  row_vector_fv v2(3);
  stan::agrad::size_type length = 2;

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

  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::agrad::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::agrad::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::agrad::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrix, dot_product_vector_rowvector_no_exception_length) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  row_vector_d d2(2);
  row_vector_fv v2(4);
  stan::agrad::size_type length = 2;

  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2;
  v2 << 4, -2, -1, 2;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;
   v2(3).d_ = 1.0;

  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::agrad::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::agrad::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::agrad::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrix, dot_product_rowvector_rowvector_length) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3), d2(3);
  row_vector_fv v1(3), v2(3);
  stan::agrad::size_type length = 2;

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

  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::agrad::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::agrad::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::agrad::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrix, dot_product_rowvector_rowvector_no_exception_length) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3), d2(2);
  row_vector_fv v1(3), v2(4);
  stan::agrad::size_type length = 2;

  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  d2 << 4, -2;
  v2 << 4, -2, -1, 2;
   v2(0).d_ = 1.0;
   v2(1).d_ = 1.0;
   v2(2).d_ = 1.0;
   v2(3).d_ = 1.0;

  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::agrad::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::agrad::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::agrad::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::agrad::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrix, dot_product_stdvector_stdvector_length) {
  using std::vector;
  using stan::agrad::fvar;

  vector<fvar<double> > fv1;
  vector<fvar<double> > fv2;
  vector<double> dv;
  stan::agrad::size_type length = 2;
  fvar<double> a = 1.0;
  fvar<double> b = 3.0;
  fvar<double> c = 5.0;
  a.d_ = 1.0;
  b.d_ = 1.0;
  c.d_ = 1.0;

  fv1.push_back(a);
  fv1.push_back(b);
  fv1.push_back(c); 
  fv2.push_back(a);
  fv2.push_back(b);
  fv2.push_back(c);   
  dv.push_back(2.0);
  dv.push_back(4.0);
  dv.push_back(6.0);

  EXPECT_FLOAT_EQ(14.0, dot_product(fv1, dv, length).val_);
  EXPECT_FLOAT_EQ(14.0, dot_product(dv, fv1, length).val_);
  EXPECT_FLOAT_EQ(10.0, dot_product(fv1, fv2, length).val_);
  EXPECT_FLOAT_EQ( 6.0, dot_product(fv1, dv, length).d_);
  EXPECT_FLOAT_EQ( 6.0, dot_product(dv, fv1, length).d_);
  EXPECT_FLOAT_EQ( 8.0, dot_product(fv1, fv2, length).d_);
}
TEST(AgradFwdMatrix, dot_product_matrix_matrix_exception_length) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::dot_product;
  stan::agrad::size_type length = 3;
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

  EXPECT_THROW(dot_product(v1,d1,length), std::domain_error);
  EXPECT_THROW(dot_product(v1,d2,length), std::domain_error);
  EXPECT_THROW(dot_product(v1,d3,length), std::domain_error);
  EXPECT_THROW(dot_product(v1,v2,length), std::domain_error);
  EXPECT_THROW(dot_product(v1,v3,length), std::domain_error);
  EXPECT_THROW(dot_product(v1,v4,length), std::domain_error);
  EXPECT_THROW(dot_product(v1,v5,length), std::domain_error);
  EXPECT_THROW(dot_product(v1,v6,length), std::domain_error);

  EXPECT_THROW(dot_product(v2,d1,length), std::domain_error);
  EXPECT_THROW(dot_product(v2,d2,length), std::domain_error);
  EXPECT_THROW(dot_product(v2,d3,length), std::domain_error);
  EXPECT_THROW(dot_product(v2,v1,length), std::domain_error);
  EXPECT_THROW(dot_product(v2,v3,length), std::domain_error);
  EXPECT_THROW(dot_product(v2,v4,length), std::domain_error);
  EXPECT_THROW(dot_product(v2,v5,length), std::domain_error);
  EXPECT_THROW(dot_product(v2,v6,length), std::domain_error);

  EXPECT_THROW(dot_product(d1,v1,length), std::domain_error);
  EXPECT_THROW(dot_product(d1,v2,length), std::domain_error);
  EXPECT_THROW(dot_product(d1,v3,length), std::domain_error);
  EXPECT_THROW(dot_product(d1,v4,length), std::domain_error);
  EXPECT_THROW(dot_product(d1,v5,length), std::domain_error);
  EXPECT_THROW(dot_product(d1,v6,length), std::domain_error);

  EXPECT_THROW(dot_product(d2,v1,length), std::domain_error);
  EXPECT_THROW(dot_product(d2,v2,length), std::domain_error);
  EXPECT_THROW(dot_product(d2,v3,length), std::domain_error);
  EXPECT_THROW(dot_product(d2,v4,length), std::domain_error);
  EXPECT_THROW(dot_product(d2,v5,length), std::domain_error);
  EXPECT_THROW(dot_product(d2,v6,length), std::domain_error);
}
