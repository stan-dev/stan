#include <stan/math/fwd/mat/fun/dot_product.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixDotProduct, vector_vector_fd) {
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d vd_1(3), vd_2(3);
  vector_fd vv_1(3), vv_2(3);
  
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

  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(vv_1, vd_2).val_);
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(vd_1, vv_2).val_);
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(vv_1, vv_2).val_);  
  EXPECT_FLOAT_EQ( 1, stan::math::dot_product(vv_1, vd_2).d_);
  EXPECT_FLOAT_EQ(-1, stan::math::dot_product(vd_1, vv_2).d_);
  EXPECT_FLOAT_EQ( 0, stan::math::dot_product(vv_1, vv_2).d_);
}

TEST(AgradFwdMatrixDotProduct, vector_vector_fd_exception) {
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d1(3);
  vector_fd v1(3);
  vector_d d2(2);
  vector_fd v2(4);

  EXPECT_THROW(stan::math::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(v1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixDotProduct, rowvector_vector_fd) {
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  vector_d d2(3);
  vector_fd v2(3);
  
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

  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, d2).val_);
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(d1, v2).val_);
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, v2).val_);
  EXPECT_FLOAT_EQ( 1, stan::math::dot_product(v1, d2).d_);
  EXPECT_FLOAT_EQ(-1, stan::math::dot_product(d1, v2).d_);
  EXPECT_FLOAT_EQ( 0, stan::math::dot_product(v1, v2).d_);
}

TEST(AgradFwdMatrixDotProduct, rowvector_vector_fd_exception) {
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  vector_d d2(2);
  vector_fd v2(4);

  EXPECT_THROW(stan::math::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(v1, v2), std::invalid_argument);
}

TEST(AgradFwdMatrixDotProduct, vector_rowvector_fd) {
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  vector_d d1(3);
  vector_fd v1(3);
  row_vector_d d2(3);
  row_vector_fd v2(3);
  
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

  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, d2).val_);
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(d1, v2).val_);
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, v2).val_);
  EXPECT_FLOAT_EQ( 1, stan::math::dot_product(v1, d2).d_);
  EXPECT_FLOAT_EQ(-1, stan::math::dot_product(d1, v2).d_);
  EXPECT_FLOAT_EQ( 0, stan::math::dot_product(v1, v2).d_);
}

TEST(AgradFwdMatrixDotProduct, vector_rowvector_fd_exception) {
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  vector_d d1(3);
  vector_fd v1(3);
  row_vector_d d2(2);
  row_vector_fd v2(4);

  EXPECT_THROW(stan::math::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(v1, v2), std::invalid_argument);
}

TEST(AgradFwdMatrixDotProduct, rowvector_rowvector_fd) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3), d2(3);
  row_vector_fd v1(3), v2(3);
  
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

  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, d2).val_);
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(d1, v2).val_);
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, v2).val_);
  EXPECT_FLOAT_EQ( 1, stan::math::dot_product(v1, d2).d_);
  EXPECT_FLOAT_EQ(-1, stan::math::dot_product(d1, v2).d_);
  EXPECT_FLOAT_EQ( 0, stan::math::dot_product(v1, v2).d_);
}

TEST(AgradFwdMatrixDotProduct, rowvector_rowvector_fd_exception) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3), d2(2);
  row_vector_fd v1(3), v2(4);

  EXPECT_THROW(stan::math::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(v1, v2), std::invalid_argument);
}

TEST(AgradFwdMatrixDotProduct, stdvector_stdvector_fd) {
  using std::vector;
  using stan::math::fvar;

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
TEST(AgradFwdMatrixDotProduct, matrix_matrix_fd_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::dot_product;
  matrix_d d1(3,3);
  matrix_d d2(3,2);
  matrix_d d3(2,3);
  matrix_fd v1(3,3);
  matrix_fd v2(3,3);
  matrix_fd v3(3,2);
  matrix_fd v4(3,2);
  matrix_fd v5(2,3);
  matrix_fd v6(2,3);

  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v2 << 4, -2, -1, 2, 1, 2, 1, 3, -5;
  v3 << 4, -2, -1, 2, 1, 2;
  v4 << 4, -2, -1, 2, 1, 2;
  v5 << 4, -2, -1, 2, 1, 2;
  v6 << 4, -2, -1, 2, 1, 2;

  EXPECT_THROW(dot_product(v1,d1), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,d2), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,d3), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v2), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v3), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v4), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v5), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v6), std::invalid_argument);

  EXPECT_THROW(dot_product(v2,d1), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,d2), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,d3), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v1), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v3), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v4), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v5), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v6), std::invalid_argument);

  EXPECT_THROW(dot_product(d1,v1), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v2), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v3), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v4), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v5), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v6), std::invalid_argument);

  EXPECT_THROW(dot_product(d2,v1), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v2), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v3), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v4), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v5), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v6), std::invalid_argument);
}
TEST(AgradFwdMatrixDotProduct, vector_vector_fd_length) {
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d vd_1(3), vd_2(3);
  vector_fd vv_1(3), vv_2(3);
  stan::math::size_type length = 2;
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

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(vv_1, vd_2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(vd_1, vv_2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(vv_1, vv_2, length).val_);  
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(vv_1, vd_2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(vd_1, vv_2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(vv_1, vv_2, length).d_);
}
TEST(AgradFwdMatrixDotProduct, vector_vector_fd_no_exception_length) {
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d1(3);
  vector_fd v1(3);
  vector_d d2(2);
  vector_fd v2(4);
  stan::math::size_type length = 2;

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

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrixDotProduct, rowvector_vector_fd_length) {
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  vector_d d2(3);
  vector_fd v2(3);
  stan::math::size_type length = 2;

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

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrixDotProduct, rowvector_vector_fd_no_exception_length) {
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  vector_d d2(2);
  vector_fd v2(4);
  stan::math::size_type length = 2;

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

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrixDotProduct, vector_rowvector_fd_length) {
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  vector_d d1(3);
  vector_fd v1(3);
  row_vector_d d2(3);
  row_vector_fd v2(3);
  stan::math::size_type length = 2;

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

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrixDotProduct, vector_rowvector_fd_no_exception_length) {
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  vector_d d1(3);
  vector_fd v1(3);
  row_vector_d d2(2);
  row_vector_fd v2(4);
  stan::math::size_type length = 2;

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

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrixDotProduct, rowvector_rowvector_fd_length) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3), d2(3);
  row_vector_fd v1(3), v2(3);
  stan::math::size_type length = 2;

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

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrixDotProduct, rowvector_rowvector_fd_no_exception_length) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3), d2(2);
  row_vector_fd v1(3), v2(4);
  stan::math::size_type length = 2;

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

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_);
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_);
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_);
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_);
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_);
}
TEST(AgradFwdMatrixDotProduct, stdvector_stdvector_fd_length) {
  using std::vector;
  using stan::math::fvar;

  vector<fvar<double> > fv1;
  vector<fvar<double> > fv2;
  vector<double> dv;
  stan::math::size_type length = 2;
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
TEST(AgradFwdMatrixDotProduct, matrix_matrix_fd_exception_length) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::dot_product;
  stan::math::size_type length = 3;
  matrix_d d1(3,3);
  matrix_d d2(3,2);
  matrix_d d3(2,3);
  matrix_fd v1(3,3);
  matrix_fd v2(3,3);
  matrix_fd v3(3,2);
  matrix_fd v4(3,2);
  matrix_fd v5(2,3);
  matrix_fd v6(2,3);

  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  v1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  v2 << 4, -2, -1, 2, 1, 2, 1, 3, -5;
  v3 << 4, -2, -1, 2, 1, 2;
  v4 << 4, -2, -1, 2, 1, 2;
  v5 << 4, -2, -1, 2, 1, 2;
  v6 << 4, -2, -1, 2, 1, 2;

  EXPECT_THROW(dot_product(v1,d1,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,d2,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,d3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v2,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v4,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v5,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v6,length), std::invalid_argument);

  EXPECT_THROW(dot_product(v2,d1,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,d2,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,d3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v1,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v4,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v5,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v6,length), std::invalid_argument);

  EXPECT_THROW(dot_product(d1,v1,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v2,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v4,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v5,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v6,length), std::invalid_argument);

  EXPECT_THROW(dot_product(d2,v1,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v2,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v4,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v5,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v6,length), std::invalid_argument);
}
TEST(AgradFwdMatrixDotProduct, vector_vector_ffd) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::fvar;

  vector_d vd_1(3), vd_2(3);
  vector_ffd vv_1(3), vv_2(3);
  
  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  vd_1 << 1, 3, -5;
  vv_1 << a,b,c;
  vd_2 << 4, -2, -1;
  vv_2 << d,e,f;

  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(vv_1, vd_2).val_.val());
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(vd_1, vv_2).val_.val());
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(vv_1, vv_2).val_.val());  
  EXPECT_FLOAT_EQ( 1, stan::math::dot_product(vv_1, vd_2).d_.val());
  EXPECT_FLOAT_EQ(-1, stan::math::dot_product(vd_1, vv_2).d_.val());
  EXPECT_FLOAT_EQ( 0, stan::math::dot_product(vv_1, vv_2).d_.val());
}

TEST(AgradFwdMatrixDotProduct, vector_vector_ffd_exception) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d d1(3),d2(2);
  vector_ffd v1(3),v2(4);

  EXPECT_THROW(stan::math::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(v1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixDotProduct, rowvector_vector_ffd) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  vector_d d2(3);
  vector_ffd v2(3);
  
  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, d2).val_.val());
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(d1, v2).val_.val());
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, v2).val_.val());
  EXPECT_FLOAT_EQ( 1, stan::math::dot_product(v1, d2).d_.val());
  EXPECT_FLOAT_EQ(-1, stan::math::dot_product(d1, v2).d_.val());
  EXPECT_FLOAT_EQ( 0, stan::math::dot_product(v1, v2).d_.val());
}

TEST(AgradFwdMatrixDotProduct, rowvector_vector_ffd_exception) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  vector_d d2(2);
  vector_ffd v2(4);

  EXPECT_THROW(stan::math::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(v1, v2), std::invalid_argument);
}

TEST(AgradFwdMatrixDotProduct, vector_rowvector_ffd) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  vector_d d1(3);
  vector_ffd v1(3);
  row_vector_d d2(3);
  row_vector_ffd v2(3);
  
  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, d2).val_.val());
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(d1, v2).val_.val());
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, v2).val_.val());
  EXPECT_FLOAT_EQ( 1, stan::math::dot_product(v1, d2).d_.val());
  EXPECT_FLOAT_EQ(-1, stan::math::dot_product(d1, v2).d_.val());
  EXPECT_FLOAT_EQ( 0, stan::math::dot_product(v1, v2).d_.val());
}

TEST(AgradFwdMatrixDotProduct, vector_rowvector_ffd_exception) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  vector_d d1(3);
  vector_ffd v1(3);
  row_vector_d d2(2);
  row_vector_ffd v2(4);

  EXPECT_THROW(stan::math::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(v1, v2), std::invalid_argument);
}

TEST(AgradFwdMatrixDotProduct, rowvector_rowvector_ffd) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  row_vector_d d1(3), d2(3);
  row_vector_ffd v1(3), v2(3);
  
  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, d2).val_.val());
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(d1, v2).val_.val());
  EXPECT_FLOAT_EQ( 3, stan::math::dot_product(v1, v2).val_.val());
  EXPECT_FLOAT_EQ( 1, stan::math::dot_product(v1, d2).d_.val());
  EXPECT_FLOAT_EQ(-1, stan::math::dot_product(d1, v2).d_.val());
  EXPECT_FLOAT_EQ( 0, stan::math::dot_product(v1, v2).d_.val());
}

TEST(AgradFwdMatrixDotProduct, rowvector_rowvector_ffd_exception) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d1(3), d2(2);
  row_vector_ffd v1(3), v2(4);

  EXPECT_THROW(stan::math::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::math::dot_product(v1, v2), std::invalid_argument);
}

TEST(AgradFwdMatrixDotProduct, stdvector_stdvector_ffd) {
  using std::vector;
  using stan::math::fvar;

  vector<fvar<fvar<double> > > fv1,fv2;
  vector<double> dv;

  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;

  fv1.push_back(a);
  fv1.push_back(b);
  fv1.push_back(c); 
  fv2.push_back(a);
  fv2.push_back(b);
  fv2.push_back(c);   
  dv.push_back(2.0);
  dv.push_back(4.0);
  dv.push_back(6.0);

  EXPECT_FLOAT_EQ(44.0, dot_product(fv1, dv).val_.val());
  EXPECT_FLOAT_EQ(44.0, dot_product(dv, fv1).val_.val());
  EXPECT_FLOAT_EQ(35.0, dot_product(fv1, fv2).val_.val());
  EXPECT_FLOAT_EQ(12.0, dot_product(fv1, dv).d_.val());
  EXPECT_FLOAT_EQ(12.0, dot_product(dv, fv1).d_.val());
  EXPECT_FLOAT_EQ(18.0, dot_product(fv1, fv2).d_.val());
}
TEST(AgradFwdMatrixDotProduct, matrix_matrix_ffd_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::dot_product;
  using stan::math::fvar;

  matrix_d d1(3,3),d2(3,2),d3(2,3);
  matrix_ffd v1(3,3),v2(3,3),v3(3,2),v4(3,2),v5(2,3),v6(2,3);

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  v1 << a,b,c,d,e,f,a,b,c;
  v2 << a,b,c,d,e,f,a,b,c;
  v3 << d,e,f,a,b,c;
  v4 << d,e,f,a,b,c;
  v5 << d,e,f,a,b,c;
  v6 << d,e,f,a,b,c;

  EXPECT_THROW(dot_product(v1,d1), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,d2), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,d3), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v2), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v3), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v4), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v5), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v6), std::invalid_argument);

  EXPECT_THROW(dot_product(v2,d1), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,d2), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,d3), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v1), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v3), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v4), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v5), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v6), std::invalid_argument);

  EXPECT_THROW(dot_product(d1,v1), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v2), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v3), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v4), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v5), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v6), std::invalid_argument);

  EXPECT_THROW(dot_product(d2,v1), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v2), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v3), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v4), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v5), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v6), std::invalid_argument);
}
TEST(AgradFwdMatrixDotProduct, vector_vector_ffd_length) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::fvar;

  vector_d vd_1(3), vd_2(3);
  vector_ffd vv_1(3), vv_2(3);
  stan::math::size_type length = 2;

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  vd_1 << 1, 3, -5;
  vv_1 << a,b,c;
  vd_2 << 4, -2, -1;
  vv_2 << d,e,f;

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(vv_1, vd_2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(vd_1, vv_2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(vv_1, vv_2, length).val_.val());  
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(vv_1, vd_2, length).d_.val());
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(vd_1, vv_2, length).d_.val());
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(vv_1, vv_2, length).d_.val());
}
TEST(AgradFwdMatrixDotProduct, vector_vector_ffd_no_exception_length) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::fvar;

  vector_d d1(3),d2(2);
  vector_ffd v1(3),v2(4);
  stan::math::size_type length = 2;

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2;
  v2 << d,e,f,-e;

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_.val());
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_.val());
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_.val());
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_.val());
}
TEST(AgradFwdMatrixDotProduct, rowvector_vector_ffd_length) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  vector_d d2(3);
  vector_ffd v2(3);
  stan::math::size_type length = 2;

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_.val());
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_.val());
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_.val());
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_.val());
}
TEST(AgradFwdMatrixDotProduct, rowvector_vector_no_exception_length) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  vector_d d2(2);
  vector_ffd v2(4);
  stan::math::size_type length = 2;

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2;
  v2 << d,e,f,-e;

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_.val());
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_.val());
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_.val());
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_.val());
}
TEST(AgradFwdMatrixDotProduct, vector_rowvector_ffd_length) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  vector_d d1(3);
  vector_ffd v1(3);
  row_vector_d d2(3);
  row_vector_ffd v2(3);
  stan::math::size_type length = 2;

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_.val());
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_.val());
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_.val());
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_.val());
}
TEST(AgradFwdMatrixDotProduct, vector_rowvector_ffd_no_exception_length) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  vector_d d1(3);
  vector_ffd v1(3);
  row_vector_d d2(2);
  row_vector_ffd v2(4);
  stan::math::size_type length = 2;

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2;
  v2 << d,e,f,-e;

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_.val());
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_.val());
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_.val());
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_.val());
}
TEST(AgradFwdMatrixDotProduct, rowvector_rowvector_ffd_length) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  row_vector_d d1(3), d2(3);
  row_vector_ffd v1(3), v2(3);
  stan::math::size_type length = 2;

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_.val());
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_.val());
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_.val());
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_.val());
}
TEST(AgradFwdMatrixDotProduct, rowvector_rowvector_ffd_no_exception_length) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  row_vector_d d1(3), d2(2);
  row_vector_ffd v1(3), v2(4);
  stan::math::size_type length = 2;

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = -2.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = -1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2;
  v2 << d,e,f,-e;

  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, d2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(d1, v2, length).val_.val());
  EXPECT_FLOAT_EQ(-2, stan::math::dot_product(v1, v2, length).val_.val());
  EXPECT_FLOAT_EQ( 2, stan::math::dot_product(v1, d2, length).d_.val());
  EXPECT_FLOAT_EQ( 4, stan::math::dot_product(d1, v2, length).d_.val());
  EXPECT_FLOAT_EQ( 6, stan::math::dot_product(v1, v2, length).d_.val());
}
TEST(AgradFwdMatrixDotProduct, stdvector_stdvector_ffd_length) {
  using std::vector;
  using stan::math::fvar;

  vector<fvar<fvar<double> > > fv1,fv2;
  vector<double> dv;
  stan::math::size_type length = 2;

  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;

  fv1.push_back(a);
  fv1.push_back(b);
  fv1.push_back(c); 
  fv2.push_back(a);
  fv2.push_back(b);
  fv2.push_back(c);   
  dv.push_back(2.0);
  dv.push_back(4.0);
  dv.push_back(6.0);

  EXPECT_FLOAT_EQ(14.0, dot_product(fv1, dv, length).val_.val());
  EXPECT_FLOAT_EQ(14.0, dot_product(dv, fv1, length).val_.val());
  EXPECT_FLOAT_EQ(10.0, dot_product(fv1, fv2, length).val_.val());
  EXPECT_FLOAT_EQ( 6.0, dot_product(fv1, dv, length).d_.val());
  EXPECT_FLOAT_EQ( 6.0, dot_product(dv, fv1, length).d_.val());
  EXPECT_FLOAT_EQ( 8.0, dot_product(fv1, fv2, length).d_.val());
}
TEST(AgradFwdMatrixDotProduct, matrix_matrix_ffd_exception_length) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::dot_product;
  using stan::math::fvar;
  stan::math::size_type length = 3;
  matrix_d d1(3,3),d2(3,2),d3(2,3);
  matrix_ffd v1(3,3),v2(3,3),v3(3,2),v4(3,2),v5(2,3),v6(2,3);

  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -5.0;
  c.d_.val_ = 1.0;

  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  v1 << a,b,c,a,b,c,a,b,c;
  v2 << a,b,c,a,b,c,a,b,c;
  v3 << a,b,c,a,b,c;
  v4 << a,b,c,a,b,c;
  v5 << a,b,c,a,b,c;
  v6 << a,b,c,a,b,c;


  EXPECT_THROW(dot_product(v1,d1,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,d2,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,d3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v2,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v4,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v5,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v1,v6,length), std::invalid_argument);

  EXPECT_THROW(dot_product(v2,d1,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,d2,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,d3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v1,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v4,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v5,length), std::invalid_argument);
  EXPECT_THROW(dot_product(v2,v6,length), std::invalid_argument);

  EXPECT_THROW(dot_product(d1,v1,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v2,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v4,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v5,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d1,v6,length), std::invalid_argument);

  EXPECT_THROW(dot_product(d2,v1,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v2,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v3,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v4,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v5,length), std::invalid_argument);
  EXPECT_THROW(dot_product(d2,v6,length), std::invalid_argument);
}
