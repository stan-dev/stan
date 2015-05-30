#include <stan/math/prim/mat/fun/col.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>

TEST(AgradMixMatrixCol,matrix_fv) {
  using stan::math::col;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(5.0,1.0);
  fvar<var> f(6.0,1.0);
  matrix_fv y(2,3);
  y << a,b,c,d,e,f;

  vector_fv z = col(y,1);
  EXPECT_EQ(2,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_.val());
  EXPECT_FLOAT_EQ(4.0,z[1].val_.val());
  EXPECT_FLOAT_EQ(1.0,z[0].d_.val());
  EXPECT_FLOAT_EQ(1.0,z[1].d_.val());

  vector_fv w = col(y,2);
  EXPECT_EQ(2,w.size());
  EXPECT_EQ(2.0,w[0].val_.val());
  EXPECT_EQ(5.0,w[1].val_.val());
  EXPECT_EQ(1.0,w[0].d_.val());
  EXPECT_EQ(1.0,w[1].d_.val());
}
TEST(AgradMixMatrixCol,matrix_fv_exc0) {
  using stan::math::col;
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(5.0,1.0);
  fvar<var> f(6.0,1.0);
  matrix_fv y(2,3);
  y << a,b,c,d,e,f;

  EXPECT_THROW(col(y,0),std::out_of_range);
  EXPECT_THROW(col(y,7),std::out_of_range);
}
TEST(AgradMixMatrixCol,matrix_fv_excHigh) {
  using stan::math::col;
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(5.0,1.0);
  fvar<var> f(6.0,1.0);
  matrix_fv y(2,3);
  y << a,b,c,d,e,f;

  EXPECT_THROW(col(y,0),std::out_of_range);
  EXPECT_THROW(col(y,5),std::out_of_range);
}
TEST(AgradMixMatrixCol,matrix_ffv) {
  using stan::math::col;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
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
  f.val_.val_ = 6.0;
  f.d_.val_ = 1.0;

  matrix_ffv y(2,3);
  y << a,b,c,d,e,f;

  vector_ffv z = col(y,1);
  EXPECT_EQ(2,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_.val().val());
  EXPECT_FLOAT_EQ(4.0,z[1].val_.val().val());
  EXPECT_FLOAT_EQ(1.0,z[0].d_.val().val());
  EXPECT_FLOAT_EQ(1.0,z[1].d_.val().val());

  vector_ffv w = col(y,2);
  EXPECT_EQ(2,w.size());
  EXPECT_EQ(2.0,w[0].val_.val().val());
  EXPECT_EQ(5.0,w[1].val_.val().val());
  EXPECT_EQ(1.0,w[0].d_.val().val());
  EXPECT_EQ(1.0,w[1].d_.val().val());
}
TEST(AgradMixMatrixCol,matrix_ffv_exc0) {
  using stan::math::col;
  using stan::math::matrix_ffv;
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
  f.val_.val_ = 6.0;
  f.d_.val_ = 1.0;

  matrix_ffv y(2,3);
  y << a,b,c,d,e,f;

  EXPECT_THROW(col(y,0),std::out_of_range);
  EXPECT_THROW(col(y,7),std::out_of_range);
}
TEST(AgradMixMatrixCol,matrix_ffv_excHigh) {
  using stan::math::col;
  using stan::math::matrix_ffv;
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
  f.val_.val_ = 6.0;
  f.d_.val_ = 1.0;

  matrix_ffv y(2,3);
  y << a,b,c,d,e,f;

  EXPECT_THROW(col(y,0),std::out_of_range);
  EXPECT_THROW(col(y,5),std::out_of_range);
}
