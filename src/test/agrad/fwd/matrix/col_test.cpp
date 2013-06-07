#include <stan/math/matrix/col.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/var.hpp>

TEST(AgradFwdMatrix,col_v) {
  using stan::math::col;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  vector_fv z = col(y,1);
  EXPECT_EQ(2,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_);
  EXPECT_FLOAT_EQ(4.0,z[1].val_);
  EXPECT_FLOAT_EQ(1.0,z[0].d_);
  EXPECT_FLOAT_EQ(1.0,z[1].d_);

  vector_fv w = col(y,2);
  EXPECT_EQ(2,w.size());
  EXPECT_EQ(2.0,w[0].val_);
  EXPECT_EQ(5.0,w[1].val_);
  EXPECT_EQ(1.0,w[0].d_);
  EXPECT_EQ(1.0,w[1].d_);
}
TEST(AgradFwdMatrix,col_v_exc0) {
  using stan::math::col;
  using stan::agrad::matrix_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(col(y,0),std::domain_error);
  EXPECT_THROW(col(y,7),std::domain_error);
}
TEST(AgradFwdMatrix,col_v_excHigh) {
  using stan::math::col;
  using stan::agrad::matrix_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(col(y,0),std::domain_error);
  EXPECT_THROW(col(y,5),std::domain_error);
}
TEST(AgradFwdFvarVarMatrix,col_v) {
  using stan::math::col;
  using stan::agrad::matrix_fvv;
  using stan::agrad::vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(5.0,1.0);
  fvar<var> f(6.0,1.0);
  matrix_fvv y(2,3);
  y << a,b,c,d,e,f;

  vector_fvv z = col(y,1);
  EXPECT_EQ(2,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_.val());
  EXPECT_FLOAT_EQ(4.0,z[1].val_.val());
  EXPECT_FLOAT_EQ(1.0,z[0].d_.val());
  EXPECT_FLOAT_EQ(1.0,z[1].d_.val());

  vector_fvv w = col(y,2);
  EXPECT_EQ(2,w.size());
  EXPECT_EQ(2.0,w[0].val_.val());
  EXPECT_EQ(5.0,w[1].val_.val());
  EXPECT_EQ(1.0,w[0].d_.val());
  EXPECT_EQ(1.0,w[1].d_.val());
}
TEST(AgradFwdFvarVarMatrix,col_v_exc0) {
  using stan::math::col;
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(5.0,1.0);
  fvar<var> f(6.0,1.0);
  matrix_fvv y(2,3);
  y << a,b,c,d,e,f;

  EXPECT_THROW(col(y,0),std::domain_error);
  EXPECT_THROW(col(y,7),std::domain_error);
}
TEST(AgradFwdFvarVarMatrix,col_v_excHigh) {
  using stan::math::col;
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(4.0,1.0);
  fvar<var> e(5.0,1.0);
  fvar<var> f(6.0,1.0);
  matrix_fvv y(2,3);
  y << a,b,c,d,e,f;

  EXPECT_THROW(col(y,0),std::domain_error);
  EXPECT_THROW(col(y,5),std::domain_error);
}
TEST(AgradFwdFvarFvarMatrix,col_v) {
  using stan::math::col;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;

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
  f.val_.val_ = 6.0;
  f.d_.val_ = 1.0;

  matrix_ffv y(2,3);
  y << a,b,c,d,e,f;

  vector_ffv z = col(y,1);
  EXPECT_EQ(2,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_.val());
  EXPECT_FLOAT_EQ(4.0,z[1].val_.val());
  EXPECT_FLOAT_EQ(1.0,z[0].d_.val());
  EXPECT_FLOAT_EQ(1.0,z[1].d_.val());

  vector_ffv w = col(y,2);
  EXPECT_EQ(2,w.size());
  EXPECT_EQ(2.0,w[0].val_.val());
  EXPECT_EQ(5.0,w[1].val_.val());
  EXPECT_EQ(1.0,w[0].d_.val());
  EXPECT_EQ(1.0,w[1].d_.val());
}
TEST(AgradFwdFvarFvarMatrix,col_v_exc0) {
  using stan::math::col;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;

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
  f.val_.val_ = 6.0;
  f.d_.val_ = 1.0;

  matrix_ffv y(2,3);
  y << a,b,c,d,e,f;

  EXPECT_THROW(col(y,0),std::domain_error);
  EXPECT_THROW(col(y,7),std::domain_error);
}
TEST(AgradFwdFvarFvarMatrix,col_v_excHigh) {
  using stan::math::col;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;

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
  f.val_.val_ = 6.0;
  f.d_.val_ = 1.0;

  matrix_ffv y(2,3);
  y << a,b,c,d,e,f;

  EXPECT_THROW(col(y,0),std::domain_error);
  EXPECT_THROW(col(y,5),std::domain_error);
}
