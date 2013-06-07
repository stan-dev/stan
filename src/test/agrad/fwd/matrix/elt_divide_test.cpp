#include <stan/math/matrix/elt_divide.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/var.hpp>

TEST(AgradFwdMatrix,elt_divide_vec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::vector_fv;

  vector_fv x(2), y(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_);
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_);

}
TEST(AgradFwdMatrix,elt_divide_vec_vd) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_fv x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  vector_d y(2);
  y << 10, 100;

  vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.1,z(0).d_);
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_);
}
TEST(AgradFwdMatrix,elt_divide_vec_dv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d x(2);
  x << 2, 5;
  vector_fv y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(-0.02,z(0).d_);
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_);
}

TEST(AgradFwdMatrix,elt_divide_rowvec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_fv;

  row_vector_fv x(2),y(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  row_vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_);
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_);
}
TEST(AgradFwdMatrix,elt_divide_rowvec_vd) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_fv x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.1,z(0).d_);
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_);
}
TEST(AgradFwdMatrix,elt_divide_rowvec_dv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d x(2);
  x << 2, 5;
  row_vector_fv y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  row_vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(-0.02,z(0).d_);
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_);
}

TEST(AgradFwdMatrix,elt_divide_mat_vv) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_fv;

  matrix_fv x(2,3),y(2,3);
  x << 2, 5, 7, 13, 29, 112;
   x(0,0).d_ = 1.0;
   x(0,1).d_ = 1.0;
   x(0,2).d_ = 1.0;
   x(1,0).d_ = 1.0;
   x(1,1).d_ = 1.0;
   x(1,2).d_ = 1.0;
  y << 10, 100, 1000, 10000, 100000, 1000000;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;

  matrix_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_);
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_);
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(0.079999998,z(0,0).d_);
  EXPECT_FLOAT_EQ(0.0094999997,z(0,1).d_);
  EXPECT_FLOAT_EQ(9.9988802e-07,z(1,2).d_);
}
TEST(AgradFwdMatrix,elt_divide_mat_vd) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  
  matrix_fv x(2,3);
  x << 2, 5, 7, 13, 29, 112;
   x(0,0).d_ = 1.0;
   x(0,1).d_ = 1.0;
   x(0,2).d_ = 1.0;
   x(1,0).d_ = 1.0;
   x(1,1).d_ = 1.0;
   x(1,2).d_ = 1.0;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_);
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_);
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(0.1,z(0,0).d_);
  EXPECT_FLOAT_EQ(0.0099999998,z(0,1).d_);
  EXPECT_FLOAT_EQ(1e-06,z(1,2).d_);
}
TEST(AgradFwdMatrix,elt_divide_mat_dv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_fv y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;

  matrix_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_);
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_);
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(-0.02,z(0,0).d_);
  EXPECT_FLOAT_EQ(-0.00050000002,z(0,1).d_);
  EXPECT_FLOAT_EQ(-1.12e-10,z(1,2).d_);
}
TEST(AgradFvarVarMatrix,elt_divide_vec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);
  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  vector_fvv x(2);
  x << a,b;
  vector_fvv y(2);
  y << c,d;

  vector_fvv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val());

}
TEST(AgradFvarVarMatrix,elt_divide_vec_vd) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);

  vector_fvv x(2);
  x << a,b;
  vector_d y(2);
  y << 10, 100;

  vector_fvv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_.val());
}
TEST(AgradFvarVarMatrix,elt_divide_vec_dv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fvv;  
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  vector_d x(2);
  x << 2, 5;
  vector_fvv y(2);
  y << c,d;

  vector_fvv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_.val());
}
TEST(AgradFvarVarMatrix,elt_divide_rowvec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);
  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  row_vector_fvv x(2);
  x << a,b;
  row_vector_fvv y(2);
  y << c,d;

  row_vector_fvv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val());
}
TEST(AgradFvarVarMatrix,elt_divide_rowvec_vd) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);

  row_vector_fvv x(2);
  x << a,b;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_fvv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_.val());
}
TEST(AgradFvarVarMatrix,elt_divide_rowvec_dv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  row_vector_d x(2);
  x << 2, 5;
  row_vector_fvv y(2);
  y << c,d;
  row_vector_fvv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_.val());
}

TEST(AgradFvarVarMatrix,elt_divide_mat_vv) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);
  fvar<var> c(7.0,1.0);
  fvar<var> d(13.0,1.0);
  fvar<var> e(20.0,1.0);
  fvar<var> f(112.0,1.0);
  fvar<var> g(10.0,1.0);
  fvar<var> h(100.0,1.0);
  fvar<var> i(1000.0,1.0);
  fvar<var> j(10000.0,1.0);
  fvar<var> k(100000.0,1.0);
  fvar<var> l(1000000.0,1.0);

  matrix_fvv x(2,3);
  x << a,b,c,d,e,f;
  matrix_fvv y(2,3);
  y << g,h,i,j,k,l;

  matrix_fvv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(9.9988802e-07,z(1,2).d_.val());
}
TEST(AgradFvarVarMatrix,elt_divide_mat_vd) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);
  fvar<var> c(7.0,1.0);
  fvar<var> d(13.0,1.0);
  fvar<var> e(20.0,1.0);
  fvar<var> f(112.0,1.0);

  matrix_fvv x(2,3);
  x << a,b,c,d,e,f;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_fvv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(1e-06,z(1,2).d_.val());
}
TEST(AgradFvarVarMatrix,elt_divide_mat_dv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> g(10.0,1.0);
  fvar<var> h(100.0,1.0);
  fvar<var> i(1000.0,1.0);
  fvar<var> j(10000.0,1.0);
  fvar<var> k(100000.0,1.0);
  fvar<var> l(1000000.0,1.0);

  matrix_d x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_fvv y(2,3);
  y << g,h,i,j,k,l;

  matrix_fvv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(-1.12e-10,z(1,2).d_.val());
}
TEST(AgradFvarFvarMatrix,elt_divide_vec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 5.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 10.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 100.0;
  d.d_.val_ = 1.0;  

  vector_ffv x(2);
  x << a,b;
  vector_ffv y(2);
  y << c,d;

  vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val());

}
TEST(AgradFvarFvarMatrix,elt_divide_vec_vd) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;

  fvar<fvar<double> > a,b;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 5.0;
  b.d_.val_ = 1.0;

  vector_ffv x(2);
  x << a,b;
  vector_d y(2);
  y << 10, 100;

  vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_.val());
}
TEST(AgradFvarFvarMatrix,elt_divide_vec_dv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;  
  using stan::agrad::fvar;

  fvar<fvar<double> > c,d;
  c.val_.val_ = 10.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 100.0;
  d.d_.val_ = 1.0;  

  vector_d x(2);
  x << 2, 5;
  vector_ffv y(2);
  y << c,d;

  vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_.val());
}
TEST(AgradFvarFvarMatrix,elt_divide_rowvec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 5.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 10.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 100.0;
  d.d_.val_ = 1.0;  

  row_vector_ffv x(2), y(2);
  x << a,b;
  y << c,d;

  row_vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val());
}
TEST(AgradFvarFvarMatrix,elt_divide_rowvec_vd) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 5.0;
  b.d_.val_ = 1.0;

  row_vector_ffv x(2);
  x << a,b;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_.val());
}
TEST(AgradFvarFvarMatrix,elt_divide_rowvec_dv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;

  fvar<fvar<double> > c,d;
  c.val_.val_ = 10.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 100.0;
  d.d_.val_ = 1.0;  

  row_vector_d x(2);
  x << 2, 5;
  row_vector_ffv y(2);
  y << c,d;
  row_vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_.val());
}

TEST(AgradFvarFvarMatrix,elt_divide_mat_vv) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;

  fvar<fvar<double> > a,b,c,d,e,f,g,h,i,j,k,l;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 5.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 7.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 13.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 20.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = 112.0;
  f.d_.val_ = 1.0;
  g.val_.val_ = 10.0;
  g.d_.val_ = 1.0;  
  h.val_.val_ = 100.0;
  h.d_.val_ = 1.0;
  i.val_.val_ = 1000.0;
  i.d_.val_ = 1.0;
  j.val_.val_ = 10000.0;
  j.d_.val_ = 1.0;  
  k.val_.val_ = 100000.0;
  k.d_.val_ = 1.0;
  l.val_.val_ = 1000000.0;
  l.d_.val_ = 1.0;

  matrix_ffv x(2,3), y(2,3);
  x << a,b,c,d,e,f;
  y << g,h,i,j,k,l;

  matrix_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(9.9988802e-07,z(1,2).d_.val());
}
TEST(AgradFvarFvarMatrix,elt_divide_mat_vd) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;

  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 5.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 7.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 13.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 20.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = 112.0;
  f.d_.val_ = 1.0;

  matrix_ffv x(2,3);
  x << a,b,c,d,e,f;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(1e-06,z(1,2).d_.val());
}
TEST(AgradFvarFvarMatrix,elt_divide_mat_dv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;

  fvar<fvar<double> > g,h,i,j,k,l;
  g.val_.val_ = 10.0;
  g.d_.val_ = 1.0;  
  h.val_.val_ = 100.0;
  h.d_.val_ = 1.0;
  i.val_.val_ = 1000.0;
  i.d_.val_ = 1.0;
  j.val_.val_ = 10000.0;
  j.d_.val_ = 1.0;  
  k.val_.val_ = 100000.0;
  k.d_.val_ = 1.0;
  l.val_.val_ = 1000000.0;
  l.d_.val_ = 1.0;

  matrix_d x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_ffv y(2,3);
  y << g,h,i,j,k,l;

  matrix_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(-1.12e-10,z(1,2).d_.val());
}
