#include <gtest/gtest.h>
#include <stan/math/matrix/elt_divide.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixEltDivide,fd_vec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::vector_fd;

  vector_fd x(2), y(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  vector_fd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_);
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_);

}
TEST(AgradFwdMatrixEltDivide,fd_vec_vd) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  vector_fd x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  vector_d y(2);
  y << 10, 100;

  vector_fd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.1,z(0).d_);
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_);
}
TEST(AgradFwdMatrixEltDivide,fd_vec_dv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  vector_d x(2);
  x << 2, 5;
  vector_fd y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  vector_fd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(-0.02,z(0).d_);
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_);
}

TEST(AgradFwdMatrixEltDivide,fd_rowvec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_fd;

  row_vector_fd x(2),y(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  row_vector_fd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_);
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_);
}
TEST(AgradFwdMatrixEltDivide,fd_rowvec_vd) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  row_vector_fd x(2);
  x << 2, 5;
   x(0).d_ = 1.0;
   x(1).d_ = 1.0;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_fd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(0.1,z(0).d_);
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_);
}
TEST(AgradFwdMatrixEltDivide,fd_rowvec_dv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  row_vector_d x(2);
  x << 2, 5;
  row_vector_fd y(2);
  y << 10, 100;
   y(0).d_ = 1.0;
   y(1).d_ = 1.0;

  row_vector_fd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_);
  EXPECT_FLOAT_EQ(0.05,z(1).val_);
  EXPECT_FLOAT_EQ(-0.02,z(0).d_);
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_);
}

TEST(AgradFwdMatrixEltDivide,fd_mat_vv) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_fd;

  matrix_fd x(2,3),y(2,3);
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

  matrix_fd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_);
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_);
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(0.079999998,z(0,0).d_);
  EXPECT_FLOAT_EQ(0.0094999997,z(0,1).d_);
  EXPECT_FLOAT_EQ(9.9988802e-07,z(1,2).d_);
}
TEST(AgradFwdMatrixEltDivide,fd_mat_vd) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
  
  matrix_fd x(2,3);
  x << 2, 5, 7, 13, 29, 112;
   x(0,0).d_ = 1.0;
   x(0,1).d_ = 1.0;
   x(0,2).d_ = 1.0;
   x(1,0).d_ = 1.0;
   x(1,1).d_ = 1.0;
   x(1,2).d_ = 1.0;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_fd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_);
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_);
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(0.1,z(0,0).d_);
  EXPECT_FLOAT_EQ(0.0099999998,z(0,1).d_);
  EXPECT_FLOAT_EQ(1e-06,z(1,2).d_);
}
TEST(AgradFwdMatrixEltDivide,fd_mat_dv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;

  matrix_d x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_fd y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;

  matrix_fd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_);
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_);
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_);
  EXPECT_FLOAT_EQ(-0.02,z(0,0).d_);
  EXPECT_FLOAT_EQ(-0.00050000002,z(0,1).d_);
  EXPECT_FLOAT_EQ(-1.12e-10,z(1,2).d_);
}
TEST(AgradFwdMatrixEltDivide,fv_vec_vv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);
  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  vector_fv x(2);
  x << a,b;
  vector_fv y(2);
  y << c,d;

  vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  z(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(0.1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-.02,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,fv_vec_vv_2ndDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);
  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  vector_fv x(2);
  x << a,b;
  vector_fv y(2);
  y << c,d;

  vector_fv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  z(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-.01,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-.006,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,fv_vec_vd_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);

  vector_fv x(2);
  x << a,b;
  vector_d y(2);
  y << 10, 100;

  vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_.val());

  AVEC q = createAVEC(a.val(),b.val());
  VEC h;
  z(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(0.1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,fv_vec_vd_2ndDeriv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);

  vector_fv x(2);
  x << a,b;
  vector_d y(2);
  y << 10, 100;

  vector_fv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val(),b.val());
  VEC h;
  z(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,fv_vec_dv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;  
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  vector_d x(2);
  x << 2, 5;
  vector_fv y(2);
  y << c,d;

  vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_.val());

  AVEC q = createAVEC(c.val(),d.val());
  VEC h;
  z(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.02,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,fv_vec_dv_2ndDeriv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;  
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  vector_d x(2);
  x << 2, 5;
  vector_fv y(2);
  y << c,d;

  vector_fv z = elt_divide(x,y);

  AVEC q = createAVEC(c.val(),d.val());
  VEC h;
  z(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.004,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,fv_rowvec_vv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);
  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  row_vector_fv x(2);
  x << a,b;
  row_vector_fv y(2);
  y << c,d;

  row_vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  z(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(0.1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-.02,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,fv_rowvec_vv_2ndDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);
  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  row_vector_fv x(2);
  x << a,b;
  row_vector_fv y(2);
  y << c,d;

  row_vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  z(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-.01,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-.006,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,fv_rowvec_vd_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);

  row_vector_fv x(2);
  x << a,b;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_.val());

  AVEC q = createAVEC(a.val(),b.val());
  VEC h;
  z(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(0.1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,fv_rowvec_vd_2ndDeriv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);

  row_vector_fv x(2);
  x << a,b;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_fv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val(),b.val());
  VEC h;
  z(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,fv_rowvec_dv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  row_vector_d x(2);
  x << 2, 5;
  row_vector_fv y(2);
  y << c,d;
  row_vector_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_.val());

  AVEC q = createAVEC(c.val(),d.val());
  VEC h;
  z(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-.02,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,fv_rowvec_dv_2ndDeriv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> c(10.0,1.0);
  fvar<var> d(100.0,1.0);

  row_vector_d x(2);
  x << 2, 5;
  row_vector_fv y(2);
  y << c,d;
  row_vector_fv z = elt_divide(x,y);

  AVEC q = createAVEC(c.val(),d.val());
  VEC h;
  z(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(.004,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}

TEST(AgradFwdMatrixEltDivide,fv_mat_vv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_fv;
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

  matrix_fv x(2,3);
  x << a,b,c,d,e,f;
  matrix_fv y(2,3);
  y << g,h,i,j,k,l;

  matrix_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(9.9988802e-07,z(1,2).d_.val());

  std::vector<var> q; 
  q.push_back(a.val());
  q.push_back(b.val());
  q.push_back(c.val());
  q.push_back(d.val());
  q.push_back(e.val());
  q.push_back(f.val());
  q.push_back(g.val());
  q.push_back(h.val());
  q.push_back(i.val());
  q.push_back(j.val());
  q.push_back(k.val());
  q.push_back(l.val());
  VEC hh;
  z(0).val_.grad(q,hh);
  EXPECT_FLOAT_EQ(0.1,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
  EXPECT_FLOAT_EQ(-.02,hh[6]);
  EXPECT_FLOAT_EQ(0.0,hh[7]);
  EXPECT_FLOAT_EQ(0.0,hh[8]);
  EXPECT_FLOAT_EQ(0.0,hh[9]);
  EXPECT_FLOAT_EQ(0.0,hh[10]);
  EXPECT_FLOAT_EQ(0.0,hh[11]);
}
TEST(AgradFwdMatrixEltDivide,fv_mat_vv_2ndDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_fv;
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

  matrix_fv x(2,3);
  x << a,b,c,d,e,f;
  matrix_fv y(2,3);
  y << g,h,i,j,k,l;

  matrix_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(9.9988802e-07,z(1,2).d_.val());

  std::vector<var> q; 
  q.push_back(a.val());
  q.push_back(b.val());
  q.push_back(c.val());
  q.push_back(d.val());
  q.push_back(e.val());
  q.push_back(f.val());
  q.push_back(g.val());
  q.push_back(h.val());
  q.push_back(i.val());
  q.push_back(j.val());
  q.push_back(k.val());
  q.push_back(l.val());  
  VEC hh;
  z(0).d_.grad(q,hh);
  EXPECT_FLOAT_EQ(-.01,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
  EXPECT_FLOAT_EQ(-.006,hh[6]);
  EXPECT_FLOAT_EQ(0.0,hh[7]);
  EXPECT_FLOAT_EQ(0.0,hh[8]);
  EXPECT_FLOAT_EQ(0.0,hh[9]);
  EXPECT_FLOAT_EQ(0.0,hh[10]);
  EXPECT_FLOAT_EQ(0.0,hh[11]);
}
TEST(AgradFwdMatrixEltDivide,fv_mat_vd_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);
  fvar<var> c(7.0,1.0);
  fvar<var> d(13.0,1.0);
  fvar<var> e(20.0,1.0);
  fvar<var> f(112.0,1.0);

  matrix_fv x(2,3);
  x << a,b,c,d,e,f;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(1e-06,z(1,2).d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  z(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(0.1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixEltDivide,fv_mat_vd_2ndDeriv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(5.0,1.0);
  fvar<var> c(7.0,1.0);
  fvar<var> d(13.0,1.0);
  fvar<var> e(20.0,1.0);
  fvar<var> f(112.0,1.0);

  matrix_fv x(2,3);
  x << a,b,c,d,e,f;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_fv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  z(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixEltDivide,fv_mat_dv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
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
  matrix_fv y(2,3);
  y << g,h,i,j,k,l;

  matrix_fv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(-1.12e-10,z(1,2).d_.val());

  AVEC q = createAVEC(g.val(),h.val(),i.val(),j.val(),k.val(),l.val());
  VEC hh;
  z(0).val_.grad(q,hh);
  EXPECT_FLOAT_EQ(-.02,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
}
TEST(AgradFwdMatrixEltDivide,fv_mat_dv_2ndDeriv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
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
  matrix_fv y(2,3);
  y << g,h,i,j,k,l;

  matrix_fv z = elt_divide(x,y);

  AVEC q = createAVEC(g.val(),h.val(),i.val(),j.val(),k.val(),l.val());
  VEC hh;
  z(0).d_.grad(q,hh);
  EXPECT_FLOAT_EQ(0.004,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
}
TEST(AgradFwdMatrixEltDivide,ffd_vec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::vector_ffd;
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

  vector_ffd x(2);
  x << a,b;
  vector_ffd y(2);
  y << c,d;

  vector_ffd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val());

}
TEST(AgradFwdMatrixEltDivide,ffd_vec_vd) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

  fvar<fvar<double> > a,b;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 5.0;
  b.d_.val_ = 1.0;

  vector_ffd x(2);
  x << a,b;
  vector_d y(2);
  y << 10, 100;

  vector_ffd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_.val());
}
TEST(AgradFwdMatrixEltDivide,ffd_vec_dv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;  
  using stan::agrad::fvar;

  fvar<fvar<double> > c,d;
  c.val_.val_ = 10.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 100.0;
  d.d_.val_ = 1.0;  

  vector_d x(2);
  x << 2, 5;
  vector_ffd y(2);
  y << c,d;

  vector_ffd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_.val());
}
TEST(AgradFwdMatrixEltDivide,ffd_rowvec_vv) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_ffd;
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

  row_vector_ffd x(2), y(2);
  x << a,b;
  y << c,d;

  row_vector_ffd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val());
}
TEST(AgradFwdMatrixEltDivide,ffd_rowvec_vd) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 5.0;
  b.d_.val_ = 1.0;

  row_vector_ffd x(2);
  x << a,b;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_ffd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_.val());
}
TEST(AgradFwdMatrixEltDivide,ffd_rowvec_dv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

  fvar<fvar<double> > c,d;
  c.val_.val_ = 10.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 100.0;
  d.d_.val_ = 1.0;  

  row_vector_d x(2);
  x << 2, 5;
  row_vector_ffd y(2);
  y << c,d;
  row_vector_ffd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_.val());
}

TEST(AgradFwdMatrixEltDivide,ffd_mat_vv) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_ffd;
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

  matrix_ffd x(2,3), y(2,3);
  x << a,b,c,d,e,f;
  y << g,h,i,j,k,l;

  matrix_ffd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(0.079999998,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0094999997,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(9.9988802e-07,z(1,2).d_.val());
}
TEST(AgradFwdMatrixEltDivide,ffd_mat_vd) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
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

  matrix_ffd x(2,3);
  x << a,b,c,d,e,f;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_ffd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(0.1,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(0.0099999998,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(1e-06,z(1,2).d_.val());
}
TEST(AgradFwdMatrixEltDivide,ffd_mat_dv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
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
  matrix_ffd y(2,3);
  y << g,h,i,j,k,l;

  matrix_ffd z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val());
  EXPECT_FLOAT_EQ(-0.02,z(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(0,1).d_.val());
  EXPECT_FLOAT_EQ(-1.12e-10,z(1,2).d_.val());
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_vv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  vector_ffv x(2);
  x << a,b;
  vector_ffv y(2);
  y << c,d;

  vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val().val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val().val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  z(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-.02,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_vv_2ndDeriv_1) {
  using stan::math::elt_divide;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  vector_ffv x(2);
  x << a,b;
  vector_ffv y(2);
  y << c,d;

  vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  z(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_vv_2ndDeriv_2) {
  using stan::math::elt_divide;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  vector_ffv x(2);
  x << a,b;
  vector_ffv y(2);
  y << c,d;

  vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  z(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-.01,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-.006,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_vv_3rdDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  vector_ffv x(2);
  x << a,b;
  vector_ffv y(2);
  y << c,d;

  vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  z(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.002000000,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0027999999,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_vd_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);

  vector_ffv x(2);
  x << a,b;
  vector_d y(2);
  y << 10, 100;

  vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val().val());
  EXPECT_FLOAT_EQ(0.1,z(0).d_.val().val());
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  z(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_vd_2ndDeriv_1) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);

  vector_ffv x(2);
  x << a,b;
  vector_d y(2);
  y << 10, 100;

  vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  z(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_vd_2ndDeriv_2) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);

  vector_ffv x(2);
  x << a,b;
  vector_d y(2);
  y << 10, 100;

  vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  z(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_vd_3rdDeriv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;

  vector_ffv x(2);
  x << a,b;
  vector_d y(2);
  y << 10, 100;

  vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  z(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_dv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;  
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  vector_d x(2);
  x << 2, 5;
  vector_ffv y(2);
  y << c,d;

  vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val().val());
  EXPECT_FLOAT_EQ(-0.02,z(0).d_.val().val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_.val().val());

  AVEC q = createAVEC(c.val().val(),d.val().val());
  VEC h;
  z(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-0.02,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_dv_2ndDeriv_1) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;  
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  vector_d x(2);
  x << 2, 5;
  vector_ffv y(2);
  y << c,d;

  vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(c.val().val(),d.val().val());
  VEC h;
  z(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_dv_2ndDeriv_2) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;  
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  vector_d x(2);
  x << 2, 5;
  vector_ffv y(2);
  y << c,d;

  vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(c.val().val(),d.val().val());
  VEC h;
  z(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.004,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_vec_dv_3rdDeriv) {
  using stan::math::elt_divide;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;  
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  vector_d x(2);
  x << 2, 5;
  vector_ffv y(2);
  y << c,d;

  vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(c.val().val(),d.val().val());
  VEC h;
  z(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.001200000,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_vv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  row_vector_ffv x(2);
  x << a,b;
  row_vector_ffv y(2);
  y << c,d;

  row_vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val().val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val().val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  z(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-.02,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_vv_2ndDeriv_1) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  row_vector_ffv x(2);
  x << a,b;
  row_vector_ffv y(2);
  y << c,d;

  row_vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val().val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val().val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  z(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_vv_2ndDeriv_2) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  row_vector_ffv x(2);
  x << a,b;
  row_vector_ffv y(2);
  y << c,d;

  row_vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val().val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val().val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  z(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-.01,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-.006,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_vv_3rdDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  row_vector_ffv x(2);
  x << a,b;
  row_vector_ffv y(2);
  y << c,d;

  row_vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val().val());
  EXPECT_FLOAT_EQ(0.079999998,z(0).d_.val().val());
  EXPECT_FLOAT_EQ(0.0094999997,z(1).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  z(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.002000000,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0027999999,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_vd_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);

  row_vector_ffv x(2);
  x << a,b;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val().val());
  EXPECT_FLOAT_EQ(0.1,z(0).d_.val().val());
  EXPECT_FLOAT_EQ(0.0099999998,z(1).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  z(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_vd_2ndDeriv_1) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);

  row_vector_ffv x(2);
  x << a,b;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  z(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_vd_2ndDeriv_2) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);

  row_vector_ffv x(2);
  x << a,b;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  z(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_vd_3rdDeriv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;

  row_vector_ffv x(2);
  x << a,b;
  row_vector_d y(2);
  y << 10, 100;

  row_vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val());
  VEC h;
  z(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_dv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  row_vector_d x(2);
  x << 2, 5;
  row_vector_ffv y(2);
  y << c,d;
  row_vector_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(1).val_.val().val());
  EXPECT_FLOAT_EQ(-0.02,z(0).d_.val().val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(1).d_.val().val());

  AVEC q = createAVEC(c.val().val(),d.val().val());
  VEC h;
  z(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-.02,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_dv_2ndDeriv_1) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  row_vector_d x(2);
  x << 2, 5;
  row_vector_ffv y(2);
  y << c,d;
  row_vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(c.val().val(),d.val().val());
  VEC h;
  z(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_dv_2ndDeriv_2) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);

  row_vector_d x(2);
  x << 2, 5;
  row_vector_ffv y(2);
  y << c,d;
  row_vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(c.val().val(),d.val().val());
  VEC h;
  z(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(.004,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_rowvec_dv_3rdDeriv) {
  using stan::math::elt_divide;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  c(10.0,1.0);
  fvar<fvar<var> >  d(100.0,1.0);
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  row_vector_d x(2);
  x << 2, 5;
  row_vector_ffv y(2);
  y << c,d;
  row_vector_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(c.val().val(),d.val().val());
  VEC h;
  z(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.0012000001,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_vv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(7.0,1.0);
  fvar<fvar<var> >  d(13.0,1.0);
  fvar<fvar<var> >  e(20.0,1.0);
  fvar<fvar<var> >  f(112.0,1.0);
  fvar<fvar<var> >  g(10.0,1.0);
  fvar<fvar<var> >  h(100.0,1.0);
  fvar<fvar<var> >  i(1000.0,1.0);
  fvar<fvar<var> >  j(10000.0,1.0);
  fvar<fvar<var> >  k(100000.0,1.0);
  fvar<fvar<var> >  l(1000000.0,1.0);

  matrix_ffv x(2,3);
  x << a,b,c,d,e,f;
  matrix_ffv y(2,3);
  y << g,h,i,j,k,l;

  matrix_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val().val());
  EXPECT_FLOAT_EQ(0.079999998,z(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(0.0094999997,z(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(9.9988802e-07,z(1,2).d_.val().val());

  std::vector<var> q; 
  q.push_back(a.val().val());
  q.push_back(b.val().val());
  q.push_back(c.val().val());
  q.push_back(d.val().val());
  q.push_back(e.val().val());
  q.push_back(f.val().val());
  q.push_back(g.val().val());
  q.push_back(h.val().val());
  q.push_back(i.val().val());
  q.push_back(j.val().val());
  q.push_back(k.val().val());
  q.push_back(l.val().val());
  VEC hh;
  z(0).val_.val().grad(q,hh);
  EXPECT_FLOAT_EQ(0.1,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
  EXPECT_FLOAT_EQ(-.02,hh[6]);
  EXPECT_FLOAT_EQ(0.0,hh[7]);
  EXPECT_FLOAT_EQ(0.0,hh[8]);
  EXPECT_FLOAT_EQ(0.0,hh[9]);
  EXPECT_FLOAT_EQ(0.0,hh[10]);
  EXPECT_FLOAT_EQ(0.0,hh[11]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_vv_2ndDeriv_1) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(7.0,1.0);
  fvar<fvar<var> >  d(13.0,1.0);
  fvar<fvar<var> >  e(20.0,1.0);
  fvar<fvar<var> >  f(112.0,1.0);
  fvar<fvar<var> >  g(10.0,1.0);
  fvar<fvar<var> >  h(100.0,1.0);
  fvar<fvar<var> >  i(1000.0,1.0);
  fvar<fvar<var> >  j(10000.0,1.0);
  fvar<fvar<var> >  k(100000.0,1.0);
  fvar<fvar<var> >  l(1000000.0,1.0);

  matrix_ffv x(2,3);
  x << a,b,c,d,e,f;
  matrix_ffv y(2,3);
  y << g,h,i,j,k,l;

  matrix_ffv z = elt_divide(x,y);

  std::vector<var> q; 
  q.push_back(a.val().val());
  q.push_back(b.val().val());
  q.push_back(c.val().val());
  q.push_back(d.val().val());
  q.push_back(e.val().val());
  q.push_back(f.val().val());
  q.push_back(g.val().val());
  q.push_back(h.val().val());
  q.push_back(i.val().val());
  q.push_back(j.val().val());
  q.push_back(k.val().val());
  q.push_back(l.val().val());  
  VEC hh;
  z(0).val().d_.grad(q,hh);
  EXPECT_FLOAT_EQ(0,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
  EXPECT_FLOAT_EQ(0,hh[6]);
  EXPECT_FLOAT_EQ(0.0,hh[7]);
  EXPECT_FLOAT_EQ(0.0,hh[8]);
  EXPECT_FLOAT_EQ(0.0,hh[9]);
  EXPECT_FLOAT_EQ(0.0,hh[10]);
  EXPECT_FLOAT_EQ(0.0,hh[11]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_vv_2ndDeriv_2) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(7.0,1.0);
  fvar<fvar<var> >  d(13.0,1.0);
  fvar<fvar<var> >  e(20.0,1.0);
  fvar<fvar<var> >  f(112.0,1.0);
  fvar<fvar<var> >  g(10.0,1.0);
  fvar<fvar<var> >  h(100.0,1.0);
  fvar<fvar<var> >  i(1000.0,1.0);
  fvar<fvar<var> >  j(10000.0,1.0);
  fvar<fvar<var> >  k(100000.0,1.0);
  fvar<fvar<var> >  l(1000000.0,1.0);

  matrix_ffv x(2,3);
  x << a,b,c,d,e,f;
  matrix_ffv y(2,3);
  y << g,h,i,j,k,l;

  matrix_ffv z = elt_divide(x,y);

  std::vector<var> q; 
  q.push_back(a.val().val());
  q.push_back(b.val().val());
  q.push_back(c.val().val());
  q.push_back(d.val().val());
  q.push_back(e.val().val());
  q.push_back(f.val().val());
  q.push_back(g.val().val());
  q.push_back(h.val().val());
  q.push_back(i.val().val());
  q.push_back(j.val().val());
  q.push_back(k.val().val());
  q.push_back(l.val().val());  
  VEC hh;
  z(0).d_.val().grad(q,hh);
  EXPECT_FLOAT_EQ(-.01,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
  EXPECT_FLOAT_EQ(-.006,hh[6]);
  EXPECT_FLOAT_EQ(0.0,hh[7]);
  EXPECT_FLOAT_EQ(0.0,hh[8]);
  EXPECT_FLOAT_EQ(0.0,hh[9]);
  EXPECT_FLOAT_EQ(0.0,hh[10]);
  EXPECT_FLOAT_EQ(0.0,hh[11]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_vv_3rdDeriv) {
  using stan::math::elt_divide;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(7.0,1.0);
  fvar<fvar<var> >  d(13.0,1.0);
  fvar<fvar<var> >  e(20.0,1.0);
  fvar<fvar<var> >  f(112.0,1.0);
  fvar<fvar<var> >  g(10.0,1.0);
  fvar<fvar<var> >  h(100.0,1.0);
  fvar<fvar<var> >  i(1000.0,1.0);
  fvar<fvar<var> >  j(10000.0,1.0);
  fvar<fvar<var> >  k(100000.0,1.0);
  fvar<fvar<var> >  l(1000000.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;
  g.val_.d_ = 1.0;
  h.val_.d_ = 1.0;
  i.val_.d_ = 1.0;
  j.val_.d_ = 1.0;
  k.val_.d_ = 1.0;
  l.val_.d_ = 1.0;

  matrix_ffv x(2,3);
  x << a,b,c,d,e,f;
  matrix_ffv y(2,3);
  y << g,h,i,j,k,l;

  matrix_ffv z = elt_divide(x,y);

  std::vector<var> q; 
  q.push_back(a.val().val());
  q.push_back(b.val().val());
  q.push_back(c.val().val());
  q.push_back(d.val().val());
  q.push_back(e.val().val());
  q.push_back(f.val().val());
  q.push_back(g.val().val());
  q.push_back(h.val().val());
  q.push_back(i.val().val());
  q.push_back(j.val().val());
  q.push_back(k.val().val());
  q.push_back(l.val().val());  
  VEC hh;
  z(0).d_.d_.grad(q,hh);
  EXPECT_FLOAT_EQ(0.002000000,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
  EXPECT_FLOAT_EQ(0.0027999999,hh[6]);
  EXPECT_FLOAT_EQ(0.0,hh[7]);
  EXPECT_FLOAT_EQ(0.0,hh[8]);
  EXPECT_FLOAT_EQ(0.0,hh[9]);
  EXPECT_FLOAT_EQ(0.0,hh[10]);
  EXPECT_FLOAT_EQ(0.0,hh[11]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_vd_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(7.0,1.0);
  fvar<fvar<var> >  d(13.0,1.0);
  fvar<fvar<var> >  e(20.0,1.0);
  fvar<fvar<var> >  f(112.0,1.0);

  matrix_ffv x(2,3);
  x << a,b,c,d,e,f;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val().val());
  EXPECT_FLOAT_EQ(0.1,z(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(0.0099999998,z(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(1e-06,z(1,2).d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  z(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.1,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_vd_2ndDeriv_1) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(7.0,1.0);
  fvar<fvar<var> >  d(13.0,1.0);
  fvar<fvar<var> >  e(20.0,1.0);
  fvar<fvar<var> >  f(112.0,1.0);

  matrix_ffv x(2,3);
  x << a,b,c,d,e,f;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  z(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_vd_2ndDeriv_2) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(7.0,1.0);
  fvar<fvar<var> >  d(13.0,1.0);
  fvar<fvar<var> >  e(20.0,1.0);
  fvar<fvar<var> >  f(112.0,1.0);

  matrix_ffv x(2,3);
  x << a,b,c,d,e,f;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  z(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_vd_3rdDeriv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(2.0,1.0);
  fvar<fvar<var> >  b(5.0,1.0);
  fvar<fvar<var> >  c(7.0,1.0);
  fvar<fvar<var> >  d(13.0,1.0);
  fvar<fvar<var> >  e(20.0,1.0);
  fvar<fvar<var> >  f(112.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;

  matrix_ffv x(2,3);
  x << a,b,c,d,e,f;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;

  matrix_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  z(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_dv_1stDeriv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  g(10.0,1.0);
  fvar<fvar<var> >  h(100.0,1.0);
  fvar<fvar<var> >  i(1000.0,1.0);
  fvar<fvar<var> >  j(10000.0,1.0);
  fvar<fvar<var> >  k(100000.0,1.0);
  fvar<fvar<var> >  l(1000000.0,1.0);

  matrix_d x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_ffv y(2,3);
  y << g,h,i,j,k,l;

  matrix_ffv z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val_.val().val());
  EXPECT_FLOAT_EQ(-0.02,z(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(-0.00050000002,z(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(-1.12e-10,z(1,2).d_.val().val());

  AVEC q = createAVEC(g.val().val(),h.val().val(),i.val().val(),j.val().val(),k.val().val(),l.val().val());
  VEC hh;
  z(0).val_.val().grad(q,hh);
  EXPECT_FLOAT_EQ(-.02,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_dv_2ndDeriv_1) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  g(10.0,1.0);
  fvar<fvar<var> >  h(100.0,1.0);
  fvar<fvar<var> >  i(1000.0,1.0);
  fvar<fvar<var> >  j(10000.0,1.0);
  fvar<fvar<var> >  k(100000.0,1.0);
  fvar<fvar<var> >  l(1000000.0,1.0);

  matrix_d x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_ffv y(2,3);
  y << g,h,i,j,k,l;

  matrix_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(g.val().val(),h.val().val(),i.val().val(),j.val().val(),k.val().val(),l.val().val());
  VEC hh;
  z(0).val().d_.grad(q,hh);
  EXPECT_FLOAT_EQ(0.0,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_dv_2ndDeriv_2) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  g(10.0,1.0);
  fvar<fvar<var> >  h(100.0,1.0);
  fvar<fvar<var> >  i(1000.0,1.0);
  fvar<fvar<var> >  j(10000.0,1.0);
  fvar<fvar<var> >  k(100000.0,1.0);
  fvar<fvar<var> >  l(1000000.0,1.0);

  matrix_d x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_ffv y(2,3);
  y << g,h,i,j,k,l;

  matrix_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(g.val().val(),h.val().val(),i.val().val(),j.val().val(),k.val().val(),l.val().val());
  VEC hh;
  z(0).d_.val().grad(q,hh);
  EXPECT_FLOAT_EQ(0.004,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
}
TEST(AgradFwdMatrixEltDivide,ffv_mat_dv_3rdDeriv) {
  using stan::math::elt_divide;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  g(10.0,1.0);
  fvar<fvar<var> >  h(100.0,1.0);
  fvar<fvar<var> >  i(1000.0,1.0);
  fvar<fvar<var> >  j(10000.0,1.0);
  fvar<fvar<var> >  k(100000.0,1.0);
  fvar<fvar<var> >  l(1000000.0,1.0);
  g.val_.d_ = 1.0;
  h.val_.d_ = 1.0;
  i.val_.d_ = 1.0;
  j.val_.d_ = 1.0;
  k.val_.d_ = 1.0;
  l.val_.d_ = 1.0;

  matrix_d x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_ffv y(2,3);
  y << g,h,i,j,k,l;

  matrix_ffv z = elt_divide(x,y);

  AVEC q = createAVEC(g.val().val(),h.val().val(),i.val().val(),j.val().val(),k.val().val(),l.val().val());
  VEC hh;
  z(0).d_.d_.grad(q,hh);
  EXPECT_FLOAT_EQ(-0.001200000,hh[0]);
  EXPECT_FLOAT_EQ(0.0,hh[1]);
  EXPECT_FLOAT_EQ(0.0,hh[2]);
  EXPECT_FLOAT_EQ(0.0,hh[3]);
  EXPECT_FLOAT_EQ(0.0,hh[4]);
  EXPECT_FLOAT_EQ(0.0,hh[5]);
}
