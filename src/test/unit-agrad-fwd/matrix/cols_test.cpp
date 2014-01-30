#include <stan/math/matrix/cols.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>

TEST(AgradFwdMatrixCols,vector_fd) {
  using stan::agrad::vector_fd;
  using stan::agrad::row_vector_fd;
  using stan::math::cols;

  vector_fd v(5);
  v << 0, 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
  EXPECT_EQ(1U, cols(v));

  v.resize(0);
  EXPECT_EQ(1U, cols(v));
}
TEST(AgradFwdMatrixCols,row_vector_fd) {
  using stan::agrad::row_vector_fd;
  using stan::math::cols;

  row_vector_fd rv(5);
  rv << 0, 1, 2, 3, 4;
   rv(0).d_ = 1.0;
   rv(1).d_ = 1.0;
   rv(2).d_ = 1.0;
   rv(3).d_ = 1.0;
   rv(4).d_ = 1.0;
  EXPECT_EQ(5U, cols(rv));
  
  rv.resize(0);
  EXPECT_EQ(0U, cols(rv));
}
TEST(AgradFwdMatrixCols,matrix_fd) {
  using stan::agrad::matrix_fd;
  using stan::math::cols;

  matrix_fd m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  m(0,0).d_ = 1.0;
  EXPECT_EQ(3U, cols(m));
  
  m.resize(5, 0);
  EXPECT_EQ(0U, cols(m));
}
TEST(AgradFwdMatrixCols,vector_fv) {
  using stan::agrad::vector_fv;
  using stan::agrad::row_vector_fv;
  using stan::math::cols;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(0.0,1.0);

  vector_fv v(5);
  v << e,a,b,c,d;
  EXPECT_EQ(1U, cols(v));

  v.resize(0);
  EXPECT_EQ(1U, cols(v));
}
TEST(AgradFwdMatrixCols,rowvector_fv) {
  using stan::agrad::row_vector_fv;
  using stan::math::cols;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(0.0,1.0);

  row_vector_fv rv(5);
  rv << e,a,b,c,d;
  EXPECT_EQ(5U, cols(rv));
  
  rv.resize(0);
  EXPECT_EQ(0U, cols(rv));
}
TEST(AgradFwdMatrixCols,matrix_fv) {
  using stan::agrad::matrix_fv;
  using stan::math::cols;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(5.0,1.0);
  fvar<var> f(0.0,1.0);
  matrix_fv m(2,3);
  m <<f,a,b,c,d,e;
  EXPECT_EQ(3U, cols(m));
  
  m.resize(5, 0);
  EXPECT_EQ(0U, cols(m));
}
TEST(AgradFwdFvarFvarMatrix,vector_ffd) {
  using stan::agrad::vector_ffd;
  using stan::agrad::row_vector_ffd;
  using stan::math::cols;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 0.0;
  e.d_.val_ = 1.0;

  vector_ffd v(5);
  v << e,a,b,c,d;
  EXPECT_EQ(1U, cols(v));

  v.resize(0);
  EXPECT_EQ(1U, cols(v));
}
TEST(AgradFwdMatrixCols,rowvector_ffd) {
  using stan::agrad::row_vector_ffd;
  using stan::math::cols;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 0.0;
  e.d_.val_ = 1.0;

  row_vector_ffd rv(5);
  rv << e,a,b,c,d;
  EXPECT_EQ(5U, cols(rv));
  
  rv.resize(0);
  EXPECT_EQ(0U, cols(rv));
}
TEST(AgradFwdMatrixCols,matrix_ffd) {
  using stan::agrad::matrix_ffd;
  using stan::math::cols;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 5.0;
  e.d_.val_ = 1.0; 
  f.val_.val_ = 0.0;
  f.d_.val_ = 1.0;

  matrix_ffd m(2,3);
  m <<f,a,b,c,d,e;
  EXPECT_EQ(3U, cols(m));
  
  m.resize(5, 0);
  EXPECT_EQ(0U, cols(m));
}
TEST(AgradFwdFvarFvarMatrix,vector_ffv) {
  using stan::agrad::vector_ffv;
  using stan::agrad::row_vector_ffv;
  using stan::math::cols;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a;
  fvar<fvar<var> > b;
  fvar<fvar<var> > c;
  fvar<fvar<var> > d;
  fvar<fvar<var> > e;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 0.0;
  e.d_.val_ = 1.0;

  vector_ffv v(5);
  v << e,a,b,c,d;
  EXPECT_EQ(1U, cols(v));

  v.resize(0);
  EXPECT_EQ(1U, cols(v));
}
TEST(AgradFwdMatrixCols,rowvector_ffv) {
  using stan::agrad::row_vector_ffv;
  using stan::math::cols;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a;
  fvar<fvar<var> > b;
  fvar<fvar<var> > c;
  fvar<fvar<var> > d;
  fvar<fvar<var> > e;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 0.0;
  e.d_.val_ = 1.0;

  row_vector_ffv rv(5);
  rv << e,a,b,c,d;
  EXPECT_EQ(5U, cols(rv));
  
  rv.resize(0);
  EXPECT_EQ(0U, cols(rv));
}
TEST(AgradFwdMatrixCols,matrix_ffv) {
  using stan::agrad::matrix_ffv;
  using stan::math::cols;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a;
  fvar<fvar<var> > b;
  fvar<fvar<var> > c;
  fvar<fvar<var> > d;
  fvar<fvar<var> > e;
  fvar<fvar<var> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 5.0;
  e.d_.val_ = 1.0; 
  f.val_.val_ = 0.0;
  f.d_.val_ = 1.0;

  matrix_ffv m(2,3);
  m <<f,a,b,c,d,e;
  EXPECT_EQ(3U, cols(m));
  
  m.resize(5, 0);
  EXPECT_EQ(0U, cols(m));
}
