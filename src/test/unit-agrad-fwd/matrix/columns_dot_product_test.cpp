#include <stan/agrad/fwd/matrix/columns_dot_product.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_fd) {
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

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

  EXPECT_FLOAT_EQ( 3, columns_dot_product(vv_1, vd_2)(0).val_);
  EXPECT_FLOAT_EQ( 3, columns_dot_product(vd_1, vv_2)(0).val_);   
  EXPECT_FLOAT_EQ( 3, columns_dot_product(vv_1, vv_2)(0).val_);  
  EXPECT_FLOAT_EQ( 1, columns_dot_product(vv_1, vd_2)(0).d_);
  EXPECT_FLOAT_EQ(-1, columns_dot_product(vd_1, vv_2)(0).d_);
  EXPECT_FLOAT_EQ( 0, columns_dot_product(vv_1, vv_2)(0).d_);
}
TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_fd_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  vector_d d1(3);
  vector_fd v1(3);
  vector_d d2(2);
  vector_fd v2(4);

  EXPECT_THROW(columns_dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1, v2), std::domain_error);
}
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_vector_fd) {
  using stan::math::vector_d;
  using stan::agrad::vector_fd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  vector_d d2(3);
  vector_fd v2(3);

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v2), std::domain_error);
} 
TEST(AgradFwdMatrixColumnsDotProduct, vector_rowvector_fd) {
  using stan::math::vector_d;
  using stan::agrad::vector_fd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  vector_d d1(3);
  vector_fd v1(3);
  row_vector_d d2(3);
  row_vector_fd v2(3);

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v2), std::domain_error);
} 
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_rowvector_fd) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

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

  row_vector_fd output;
  output = columns_dot_product(v1,d2);

  EXPECT_FLOAT_EQ( 4, output(0).val_);
  EXPECT_FLOAT_EQ(-6, output(1).val_);
  EXPECT_FLOAT_EQ( 5, output(2).val_);
  EXPECT_FLOAT_EQ( 4, output(0).d_);
  EXPECT_FLOAT_EQ(-2, output(1).d_);
  EXPECT_FLOAT_EQ(-1, output(2).d_);
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_fd) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
  using stan::agrad::row_vector_fd;

  matrix_d d1(3,3), d2(3,3);
  matrix_fd v1(3,3), v2(3,3);
  
  d1 << 1, 1, 1, 3, 3, 3, -5, -5, -5;
  v1 << 1, 1, 1, 3, 3, 3, -5, -5, -5;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
   v1(1,0).d_ = 1.0;
   v1(1,1).d_ = 1.0;
   v1(1,2).d_ = 1.0;
   v1(2,0).d_ = 1.0;
   v1(2,1).d_ = 1.0;
   v1(2,2).d_ = 1.0;
  d2 << 4, 4, 4, -2, -2, -2, -1, -1, -1;
  v2 << 4, 4, 4, -2, -2, -2, -1, -1, -1;
   v2(0,0).d_ = 1.0;
   v2(0,1).d_ = 1.0;
   v2(0,2).d_ = 1.0;
   v2(1,0).d_ = 1.0;
   v2(1,1).d_ = 1.0;
   v2(1,2).d_ = 1.0;
   v2(2,0).d_ = 1.0;
   v2(2,1).d_ = 1.0;
   v2(2,2).d_ = 1.0;

  row_vector_fd output;
  output = columns_dot_product(v1,d2);
  EXPECT_FLOAT_EQ( 3, output(0).val_);
  EXPECT_FLOAT_EQ( 3, output(1).val_);
  EXPECT_FLOAT_EQ( 3, output(2).val_);
  EXPECT_FLOAT_EQ( 1, output(0).d_);
  EXPECT_FLOAT_EQ( 1, output(1).d_);
  EXPECT_FLOAT_EQ( 1, output(2).d_);

  output = columns_dot_product(d1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_);
  EXPECT_FLOAT_EQ( 3, output(1).val_);
  EXPECT_FLOAT_EQ( 3, output(2).val_);
  EXPECT_FLOAT_EQ(-1, output(0).d_);
  EXPECT_FLOAT_EQ(-1, output(1).d_);
  EXPECT_FLOAT_EQ(-1, output(2).d_);

  output = columns_dot_product(v1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_);
  EXPECT_FLOAT_EQ( 3, output(1).val_);
  EXPECT_FLOAT_EQ( 3, output(2).val_);
  EXPECT_FLOAT_EQ( 0, output(0).d_);
  EXPECT_FLOAT_EQ( 0, output(1).d_);
  EXPECT_FLOAT_EQ( 0, output(2).d_);
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_fd_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
  using stan::agrad::columns_dot_product;

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

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,d3), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v4), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d1,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v4), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d2,v1), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d3,v1), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v4), std::domain_error);
}
TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_fv_1stDeriv) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(-5.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(-2.0,1.0);
  fvar<var> f(-1.0,1.0);

  vector_d vd_1(3), vd_2(3);
  vector_fv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << a,b,c;
  vd_2 << 4, -2, -1;
  vv_2 << d,e,f;

  EXPECT_FLOAT_EQ( 3, columns_dot_product(vv_1, vd_2)(0).val_.val());
  EXPECT_FLOAT_EQ( 3, columns_dot_product(vd_1, vv_2)(0).val_.val());   
  EXPECT_FLOAT_EQ( 3, columns_dot_product(vv_1, vv_2)(0).val_.val());  
  EXPECT_FLOAT_EQ( 1, columns_dot_product(vv_1, vd_2)(0).d_.val());
  EXPECT_FLOAT_EQ(-1, columns_dot_product(vd_1, vv_2)(0).d_.val());
  EXPECT_FLOAT_EQ( 0, columns_dot_product(vv_1, vv_2)(0).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  columns_dot_product(vv_1, vd_2)(0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(-2.0,h[1]);
  EXPECT_FLOAT_EQ(-1.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_fv_2ndDeriv) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(-5.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(-2.0,1.0);
  fvar<var> f(-1.0,1.0);

  vector_d vd_1(3), vd_2(3);
  vector_fv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << a,b,c;
  vd_2 << 4, -2, -1;
  vv_2 << d,e,f;

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  columns_dot_product(vv_1, vd_2)(0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_fv_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_fv v1(3);
  vector_d d2(2);
  vector_fv v2(4);

  EXPECT_THROW(columns_dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1, v2), std::domain_error);
}
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_vector_fv_exceptions) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  vector_d d2(3);
  vector_fv v2(3);

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v2), std::domain_error);
} 
TEST(AgradFwdMatrixColumnsDotProduct, vector_rowvector_fv_exceptions) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_fv v1(3);
  row_vector_d d2(3);
  row_vector_fv v2(3);

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v2), std::domain_error);
} 
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_rowvector_fv_1stDeriv) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(-5.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(-2.0,1.0);
  fvar<var> f(-1.0,1.0);

  row_vector_d d1(3), d2(3);
  row_vector_fv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  row_vector_fv output;
  output = columns_dot_product(v1,d2);

  EXPECT_FLOAT_EQ( 4, output(0).val_.val());
  EXPECT_FLOAT_EQ(-6, output(1).val_.val());
  EXPECT_FLOAT_EQ( 5, output(2).val_.val());
  EXPECT_FLOAT_EQ( 4, output(0).d_.val());
  EXPECT_FLOAT_EQ(-2, output(1).d_.val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_rowvector_fv_2ndDeriv) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(-5.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(-2.0,1.0);
  fvar<var> f(-1.0,1.0);

  row_vector_d d1(3), d2(3);
  row_vector_fv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  row_vector_fv output;
  output = columns_dot_product(v1,d2);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_fv_1stDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(-5.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(-2.0,1.0);
  fvar<var> f(-1.0,1.0);

  matrix_d d1(3,3), d2(3,3);
  matrix_fv v1(3,3), v2(3,3);
  
  d1 << 1, 1, 1, 3, 3, 3, -5, -5, -5;
  v1 << a,a,a,b,b,b,c,c,c;
  d2 << 4, 4, 4, -2, -2, -2, -1, -1, -1;
  v2 << d,d,d,e,e,e,f,f,f;

  row_vector_fv output;
  output = columns_dot_product(v1,d2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val());
  EXPECT_FLOAT_EQ( 1, output(0).d_.val());
  EXPECT_FLOAT_EQ( 1, output(1).d_.val());
  EXPECT_FLOAT_EQ( 1, output(2).d_.val());

  output = columns_dot_product(d1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val());
  EXPECT_FLOAT_EQ(-1, output(0).d_.val());
  EXPECT_FLOAT_EQ(-1, output(1).d_.val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val());

  output = columns_dot_product(v1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val());
  EXPECT_FLOAT_EQ( 0, output(0).d_.val());
  EXPECT_FLOAT_EQ( 0, output(1).d_.val());
  EXPECT_FLOAT_EQ( 0, output(2).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(-2.0,h[1]);
  EXPECT_FLOAT_EQ(-1.0,h[2]);
  EXPECT_FLOAT_EQ(1.0,h[3]);
  EXPECT_FLOAT_EQ(3.0,h[4]);
  EXPECT_FLOAT_EQ(-5.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_fv_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(-5.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(-2.0,1.0);
  fvar<var> f(-1.0,1.0);

  matrix_d d1(3,3), d2(3,3);
  matrix_fv v1(3,3), v2(3,3);
  
  d1 << 1, 1, 1, 3, 3, 3, -5, -5, -5;
  v1 << a,a,a,b,b,b,c,c,c;
  d2 << 4, 4, 4, -2, -2, -2, -1, -1, -1;
  v2 << d,d,d,e,e,e,f,f,f;

  row_vector_fv output;
  output = columns_dot_product(v1, v2);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(1.0,h[1]);
  EXPECT_FLOAT_EQ(1.0,h[2]);
  EXPECT_FLOAT_EQ(1.0,h[3]);
  EXPECT_FLOAT_EQ(1.0,h[4]);
  EXPECT_FLOAT_EQ(1.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_fv_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::columns_dot_product;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(-5.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(-2.0,1.0);
  fvar<var> f(-1.0,1.0);

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
  v1 << a,b,c,a,b,c,a,b,c;
  v2 << d,e,f,d,e,f,d,e,f;
  v3 << d,e,f,d,e,f;
  v4 << d,e,f,d,e,f;
  v5 << d,e,f,d,e,f;
  v6 << d,e,f,d,e,f;

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,d3), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v4), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d1,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v4), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d2,v1), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d3,v1), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v4), std::domain_error);
}
TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_ffd) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
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

  vector_d vd_1(3), vd_2(3);
  vector_ffd vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << a,b,c;
  vd_2 << 4, -2, -1;
  vv_2 << d,e,f;

  EXPECT_FLOAT_EQ( 3, columns_dot_product(vv_1, vd_2)(0).val_.val());
  EXPECT_FLOAT_EQ( 3, columns_dot_product(vd_1, vv_2)(0).val_.val());   
  EXPECT_FLOAT_EQ( 3, columns_dot_product(vv_1, vv_2)(0).val_.val());  
  EXPECT_FLOAT_EQ( 1, columns_dot_product(vv_1, vd_2)(0).d_.val());
  EXPECT_FLOAT_EQ(-1, columns_dot_product(vd_1, vv_2)(0).d_.val());
  EXPECT_FLOAT_EQ( 0, columns_dot_product(vv_1, vv_2)(0).d_.val());
}
TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_ffd_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

  vector_d d1(3);
  vector_ffd v1(3);
  vector_d d2(2);
  vector_ffd v2(4);

  EXPECT_THROW(columns_dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1, v2), std::domain_error);
}
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_vector_ffd) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  vector_d d2(3);
  vector_ffd v2(3);

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v2), std::domain_error);
} 
TEST(AgradFwdMatrixColumnsDotProduct, vector_rowvector_ffd) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

  vector_d d1(3);
  vector_ffd v1(3);
  row_vector_d d2(3);
  row_vector_ffd v2(3);

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v2), std::domain_error);
} 
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_rowvector_ffd) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
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

  row_vector_d d1(3), d2(3);
  row_vector_ffd v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  row_vector_ffd output;
  output = columns_dot_product(v1,d2);

  EXPECT_FLOAT_EQ( 4, output(0).val_.val());
  EXPECT_FLOAT_EQ(-6, output(1).val_.val());
  EXPECT_FLOAT_EQ( 5, output(2).val_.val());
  EXPECT_FLOAT_EQ( 4, output(0).d_.val());
  EXPECT_FLOAT_EQ(-2, output(1).d_.val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val());
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_ffd) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
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

  matrix_d d1(3,3), d2(3,3);
  matrix_ffd v1(3,3), v2(3,3);
  
  d1 << 1, 1, 1, 3, 3, 3, -5, -5, -5;
  v1 << a,a,a,b,b,b,c,c,c;
  d2 << 4, 4, 4, -2, -2, -2, -1, -1, -1;
  v2 << d,d,d,e,e,e,f,f,f;

  row_vector_ffd output;
  output = columns_dot_product(v1,d2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val());
  EXPECT_FLOAT_EQ( 1, output(0).d_.val());
  EXPECT_FLOAT_EQ( 1, output(1).d_.val());
  EXPECT_FLOAT_EQ( 1, output(2).d_.val());

  output = columns_dot_product(d1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val());
  EXPECT_FLOAT_EQ(-1, output(0).d_.val());
  EXPECT_FLOAT_EQ(-1, output(1).d_.val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val());

  output = columns_dot_product(v1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val());
  EXPECT_FLOAT_EQ( 0, output(0).d_.val());
  EXPECT_FLOAT_EQ( 0, output(1).d_.val());
  EXPECT_FLOAT_EQ( 0, output(2).d_.val());
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_ffd_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::columns_dot_product;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
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

  matrix_d d1(3,3);
  matrix_d d2(3,2);
  matrix_d d3(2,3);
  matrix_ffd v1(3,3);
  matrix_ffd v2(3,3);
  matrix_ffd v3(3,2);
  matrix_ffd v4(3,2);
  matrix_ffd v5(2,3);
  matrix_ffd v6(2,3);

  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  v1 << a,b,c,a,b,c,a,b,c;
  v2 << d,e,f,d,e,f,d,e,f;
  v3 << d,e,f,d,e,f;
  v4 << d,e,f,d,e,f;
  v5 << d,e,f,d,e,f;
  v6 << d,e,f,d,e,f;

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,d3), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v4), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d1,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v4), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d2,v1), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d3,v1), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v4), std::domain_error);
}

TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_ffv_1stDeriv) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);

  vector_d vd_1(3), vd_2(3);
  vector_ffv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << a,b,c;
  vd_2 << 4, -2, -1;
  vv_2 << d,e,f;

  EXPECT_FLOAT_EQ( 3, columns_dot_product(vv_1, vd_2)(0).val_.val().val());
  EXPECT_FLOAT_EQ( 3, columns_dot_product(vd_1, vv_2)(0).val_.val().val());   
  EXPECT_FLOAT_EQ( 3, columns_dot_product(vv_1, vv_2)(0).val_.val().val());  
  EXPECT_FLOAT_EQ( 1, columns_dot_product(vv_1, vd_2)(0).d_.val().val());
  EXPECT_FLOAT_EQ(-1, columns_dot_product(vd_1, vv_2)(0).d_.val().val());
  EXPECT_FLOAT_EQ( 0, columns_dot_product(vv_1, vv_2)(0).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  columns_dot_product(vv_1, vd_2)(0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(-2.0,h[1]);
  EXPECT_FLOAT_EQ(-1.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_ffv_2ndDeriv_1) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);

  vector_d vd_1(3), vd_2(3);
  vector_ffv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << a,b,c;
  vd_2 << 4, -2, -1;
  vv_2 << d,e,f;

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  columns_dot_product(vv_1, vd_2)(0).val_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_ffv_2ndDeriv_2) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);

  vector_d vd_1(3), vd_2(3);
  vector_ffv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << a,b,c;
  vd_2 << 4, -2, -1;
  vv_2 << d,e,f;

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  columns_dot_product(vv_1, vd_2)(0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_ffv_3rdDeriv) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;

  vector_d vd_1(3), vd_2(3);
  vector_ffv vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << a,b,c;
  vd_2 << 4, -2, -1;
  vv_2 << d,e,f;

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  columns_dot_product(vv_1, vd_2)(0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, vector_vector_ffv_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_ffv v1(3);
  vector_d d2(2);
  vector_ffv v2(4);

  EXPECT_THROW(columns_dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1, v2), std::domain_error);
}
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_vector_ffv_exceptions) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  vector_d d2(3);
  vector_ffv v2(3);

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v2), std::domain_error);
} 
TEST(AgradFwdMatrixColumnsDotProduct, vector_rowvector_ffv_exceptions) {
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_ffv v1(3);
  row_vector_d d2(3);
  row_vector_ffv v2(3);

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v2), std::domain_error);
} 
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_rowvector_ffv_1stDeriv) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);

  row_vector_d d1(3), d2(3);
  row_vector_ffv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  row_vector_ffv output;
  output = columns_dot_product(v1,d2);

  EXPECT_FLOAT_EQ( 4, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(-6, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 5, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 4, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(-2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_rowvector_ffv_2ndDeriv_1) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);

  row_vector_d d1(3), d2(3);
  row_vector_ffv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  row_vector_ffv output;
  output = columns_dot_product(v1,d2);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, rowvector_rowvector_ffv_2ndDeriv_2) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);

  row_vector_d d1(3), d2(3);
  row_vector_ffv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  row_vector_ffv output;
  output = columns_dot_product(v1,d2);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}

TEST(AgradFwdMatrixColumnsDotProduct, rowvector_rowvector_ffv_3rdDeriv) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;

  row_vector_d d1(3), d2(3);
  row_vector_ffv v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << a,b,c;
  d2 << 4, -2, -1;
  v2 << d,e,f;

  row_vector_ffv output;
  output = columns_dot_product(v1,d2);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_ffv_1stDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);

  matrix_d d1(3,3), d2(3,3);
  matrix_ffv v1(3,3), v2(3,3);
  
  d1 << 1, 1, 1, 3, 3, 3, -5, -5, -5;
  v1 << a,a,a,b,b,b,c,c,c;
  d2 << 4, 4, 4, -2, -2, -2, -1, -1, -1;
  v2 << d,d,d,e,e,e,f,f,f;

  row_vector_ffv output;
  output = columns_dot_product(v1,d2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 1, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( 1, output(1).d_.val().val());
  EXPECT_FLOAT_EQ( 1, output(2).d_.val().val());

  output = columns_dot_product(d1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(-1, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(-1, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val().val());

  output = columns_dot_product(v1, v2);
  EXPECT_FLOAT_EQ( 3, output(0).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 3, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 0, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( 0, output(1).d_.val().val());
  EXPECT_FLOAT_EQ( 0, output(2).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(-2.0,h[1]);
  EXPECT_FLOAT_EQ(-1.0,h[2]);
  EXPECT_FLOAT_EQ(1.0,h[3]);
  EXPECT_FLOAT_EQ(3.0,h[4]);
  EXPECT_FLOAT_EQ(-5.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);

  matrix_d d1(3,3), d2(3,3);
  matrix_ffv v1(3,3), v2(3,3);
  
  d1 << 1, 1, 1, 3, 3, 3, -5, -5, -5;
  v1 << a,a,a,b,b,b,c,c,c;
  d2 << 4, 4, 4, -2, -2, -2, -1, -1, -1;
  v2 << d,d,d,e,e,e,f,f,f;

  row_vector_ffv output;
  output = columns_dot_product(v1, v2);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);

  matrix_d d1(3,3), d2(3,3);
  matrix_ffv v1(3,3), v2(3,3);
  
  d1 << 1, 1, 1, 3, 3, 3, -5, -5, -5;
  v1 << a,a,a,b,b,b,c,c,c;
  d2 << 4, 4, 4, -2, -2, -2, -1, -1, -1;
  v2 << d,d,d,e,e,e,f,f,f;

  row_vector_ffv output;
  output = columns_dot_product(v1, v2);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(1.0,h[1]);
  EXPECT_FLOAT_EQ(1.0,h[2]);
  EXPECT_FLOAT_EQ(1.0,h[3]);
  EXPECT_FLOAT_EQ(1.0,h[4]);
  EXPECT_FLOAT_EQ(1.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;

  matrix_d d1(3,3), d2(3,3);
  matrix_ffv v1(3,3), v2(3,3);
  
  d1 << 1, 1, 1, 3, 3, 3, -5, -5, -5;
  v1 << a,a,a,b,b,b,c,c,c;
  d2 << 4, 4, 4, -2, -2, -2, -1, -1, -1;
  v2 << d,d,d,e,e,e,f,f,f;

  row_vector_ffv output;
  output = columns_dot_product(v1, v2);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixColumnsDotProduct, matrix_matrix_ffv_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::columns_dot_product;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(-5.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(-2.0,1.0);
  fvar<fvar<var> > f(-1.0,1.0);

  matrix_d d1(3,3);
  matrix_d d2(3,2);
  matrix_d d3(2,3);
  matrix_ffv v1(3,3);
  matrix_ffv v2(3,3);
  matrix_ffv v3(3,2);
  matrix_ffv v4(3,2);
  matrix_ffv v5(2,3);
  matrix_ffv v6(2,3);

  d1 << 1, 3, -5, 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  d2 << 1, 3, -5, 1, 3, -5;
  v1 << a,b,c,a,b,c,a,b,c;
  v2 << d,e,f,d,e,f,d,e,f;
  v3 << d,e,f,d,e,f;
  v4 << d,e,f,d,e,f;
  v5 << d,e,f,d,e,f;
  v6 << d,e,f,d,e,f;

  EXPECT_THROW(columns_dot_product(v1,d2), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,d3), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v4), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(v1,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d1,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v4), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(d1,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d2,v1), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v5), std::domain_error);
  EXPECT_THROW(columns_dot_product(d2,v6), std::domain_error);

  EXPECT_THROW(columns_dot_product(d3,v1), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v2), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v3), std::domain_error);
  EXPECT_THROW(columns_dot_product(d3,v4), std::domain_error);
}
