#include <stan/math/prim/mat/fun/cols.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>

TEST(AgradMixMatrixCols,vector_fv) {
  using stan::math::vector_fv;
  using stan::math::row_vector_fv;
  using stan::math::cols;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixCols,rowvector_fv) {
  using stan::math::row_vector_fv;
  using stan::math::cols;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixCols,matrix_fv) {
  using stan::math::matrix_fv;
  using stan::math::cols;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixFvarFvarMatrix,vector_ffv) {
  using stan::math::vector_ffv;
  using stan::math::row_vector_ffv;
  using stan::math::cols;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixCols,rowvector_ffv) {
  using stan::math::row_vector_ffv;
  using stan::math::cols;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixCols,matrix_ffv) {
  using stan::math::matrix_ffv;
  using stan::math::cols;
  using stan::math::fvar;
  using stan::math::var;

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
